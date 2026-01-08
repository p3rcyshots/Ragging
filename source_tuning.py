#Created by Prashant Saxena - https://github.com/p3rcyshots

import os
import argparse
import logging
import sys
import torch
import pandas as pd
import sqlite3
import re

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

# --- Constants ---
DB_DIR = "db"
VECTOR_DB_PATH = "vector_db"
STRUCTURED_DB_PATH = os.path.join(DB_DIR, "structured_data.db")
SUPPORTED_UNSTRUCTURED_TYPES = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
}
SUPPORTED_STRUCTURED_TYPES = [".xlsx", ".xls"]
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Ensure DB directory exists
os.makedirs(DB_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
log = logging.getLogger(__name__)


def clear_vector_db():
    """Clears the vector database if it exists."""
    if os.path.exists(VECTOR_DB_PATH):
        log.info(f"Clearing vector database at: {VECTOR_DB_PATH}")
        try:
            import shutil
            shutil.rmtree(VECTOR_DB_PATH)
            log.info("Vector database cleared successfully.")
        except Exception as e:
            log.error(f"Error clearing vector database: {e}")
            sys.exit(1)
    else:
        log.info("Vector database does not exist. Nothing to clear.")

# --- NEW FUNCTION to clear the structured database ---
def clear_structured_db():
    """Clears the structured SQLite database if it exists."""
    if os.path.exists(STRUCTURED_DB_PATH):
        log.info(f"Clearing structured database at: {STRUCTURED_DB_PATH}")
        try:
            os.remove(STRUCTURED_DB_PATH)
            log.info("Structured database cleared successfully.")
        except Exception as e:
            log.error(f"Error clearing structured database: {e}")
            sys.exit(1)
    else:
        log.info("Structured database does not exist. Nothing to clear.")

def process_structured_file(file_path):
    """Reads an Excel file and loads its data into a SQLite table."""
    log.info(f"Processing structured file: {file_path}")
    try:
        df = pd.read_excel(file_path)
        
        original_columns = df.columns
        df.columns = df.columns.str.replace(' ', '_', regex=False).str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
        sanitized_columns = df.columns
        log.info(f"Sanitized column names: {list(zip(original_columns, sanitized_columns))}")

        base_name = os.path.basename(file_path)
        table_name = os.path.splitext(base_name)[0]
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        
        with sqlite3.connect(STRUCTURED_DB_PATH) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        log.info(f"Successfully loaded data from '{base_name}' into table '{table_name}' in {STRUCTURED_DB_PATH}")
        return True
    except Exception as e:
        log.error(f"Failed to process Excel file {file_path}. Error: {e}")
        return False

def load_and_process_unstructured_files(file_paths, chunk_size=1000, chunk_overlap=200):
    """Loads and processes a list of unstructured files, splitting them into chunks."""
    documents = []
    total_files = len(file_paths)
    log.info(f"Found {total_files} unstructured files to process.")

    for i, file_path in enumerate(file_paths):
        file_extension = os.path.splitext(file_path)[1].lower()
        loader_class = SUPPORTED_UNSTRUCTURED_TYPES.get(file_extension)

        if not loader_class:
            log.warning(f"Skipping unsupported file type for vectorization: {file_path}")
            continue

        try:
            log.info(f"Processing file {i + 1}/{total_files}: {file_path}")
            loader = loader_class(file_path)
            if file_extension == ".txt":
                loader.encoding = "utf-8"
            
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
        except Exception as e:
            log.error(f"Failed to load or process {file_path}. Error: {e}")

    if not documents:
        log.warning("No unstructured documents were loaded.")
        return None

    log.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = text_splitter.split_documents(documents)
    log.info(f"Successfully split documents into {len(chunked_docs)} chunks.")
    return chunked_docs


def create_and_store_embeddings(docs):
    """Creates embeddings using a local model and stores them in ChromaDB."""
    if not docs:
        log.error("No document chunks to process. Aborting embedding creation.")
        return

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        log.info(f"Using device: CUDA ({gpu_name}) for embedding generation.")
    else:
        log.info("Using device: CPU for embedding generation.")

    log.info("Initializing local embedding model...")
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-mpnet-base-v2",
        model_kwargs={'device': device}
    )

    log.info(f"Creating and storing embeddings in ChromaDB at: {VECTOR_DB_PATH}")
    log.info("This may take some time depending on the number of documents...")

    try:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embedding_function,
            persist_directory=VECTOR_DB_PATH
        )
        log.info("Embeddings created and stored successfully!")
        log.info(f"Total documents in vector store: {vector_store._collection.count()}")
    except Exception as e:
        log.error(f"An error occurred during embedding creation: {e}")
        sys.exit(1)


def main():
    """Main function to parse arguments and orchestrate the tuning process."""
    parser = argparse.ArgumentParser(description="Source Tuning for RAG Project")
    parser.add_argument("-d", "--directory", type=str, help="Directory to process files from.")
    parser.add_argument("-f", "--files", type=str, nargs="+", help="List of files to process.")
    parser.add_argument("-c", "--clear", action="store_true", help="Clear all vector and structured databases.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="The size of each text chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="The overlap between text chunks.")

    args = parser.parse_args()

    try:
        if args.clear:
            # --- FIX: Updated the warning message for clarity ---
            print("WARNING: This will delete all tuned data in BOTH the vector DB and the structured (Excel) DB. Are you sure? (yes/no): ", end='', flush=True)
            confirm = sys.stdin.readline().strip()

            if confirm.lower() == 'yes':
                # --- FIX: Call both clearing functions ---
                clear_vector_db()
                clear_structured_db()
                if not args.directory and not args.files:
                    print("Clear operation complete. The program will now exit.")
                    sys.exit(0)
            else:
                print("Operation cancelled by user.")
                sys.exit(0)

        file_paths = []
        if args.directory:
            if not os.path.isdir(args.directory):
                print(f"ERROR: Directory not found: {args.directory}")
                sys.exit(1)
            for root, _, files in os.walk(args.directory):
                for file in files:
                    if file.startswith('.') or file.startswith('~$'):
                        log.info(f"Skipping hidden or temporary file: {file}")
                        continue
                    file_paths.append(os.path.join(root, file))
        elif args.files:
            file_paths = []
            for f in args.files:
                 base_name = os.path.basename(f)
                 if base_name.startswith('.') or base_name.startswith('~$'):
                     log.info(f"Skipping hidden or temporary file: {f}")
                     continue
                 file_paths.append(f)
        else:
            print("ERROR: You must provide a source with -d (directory) or -f (files), or use -c to clear the database.")
            parser.print_help()
            sys.exit(1)
        
        if not file_paths:
            log.warning(f"No valid files found to process in the specified source.")
            sys.exit(0)
        
        unstructured_files = []
        structured_files = []
        for fp in file_paths:
            ext = os.path.splitext(fp)[1].lower()
            if ext in SUPPORTED_STRUCTURED_TYPES:
                structured_files.append(fp)
            elif ext in SUPPORTED_UNSTRUCTURED_TYPES:
                unstructured_files.append(fp)
            else:
                log.warning(f"Skipping file with unsupported extension: {fp}")

        if structured_files:
            log.info(f"Found {len(structured_files)} structured files to process.")
            for fp in structured_files:
                process_structured_file(fp)
        else:
            log.info("No new structured files found to process.")

        if unstructured_files:
            chunked_docs = load_and_process_unstructured_files(unstructured_files, args.chunk_size, args.chunk_overlap)
            if chunked_docs:
                create_and_store_embeddings(chunked_docs)
        else:
            log.info("No new unstructured files found for vectorization.")


    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected critical error occurred: {e}")
        logging.error("An unexpected critical error occurred:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()