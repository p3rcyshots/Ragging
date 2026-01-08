#Created by Prashant Saxena - https://github.com/p3rcyshots
import os
import sys
import argparse
import bcrypt
import logging
import sqlite3
import torch
import re

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from waitress import serve
from colorama import init, Fore, Style

# LangChain and vector DB imports
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Constants ---
DB_DIR = "db"
LOG_DIR = "logs"
DB_PATH = os.path.join(DB_DIR, "app_data.db")
STRUCTURED_DB_PATH = os.path.join(DB_DIR, "structured_data.db")
LOG_FILE = os.path.join(LOG_DIR, "user_activity.log")
VECTOR_DB_PATH = "vector_db"
DEFAULT_PASSWORD = "user123"

# Ensure required directories exist
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebRAGApp")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

def hash_password(password):
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(hashed_password, user_password):
    """Verifies a user's password against the stored hash."""
    return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password)

# --- Database Setup ---
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash BLOB NOT NULL, is_admin INTEGER DEFAULT 0, must_change_password INTEGER DEFAULT 0)""")
            cursor.execute("SELECT * FROM users WHERE email='admin'")
            if not cursor.fetchone():
                admin_pass_hash = hash_password('admin123')
                cursor.execute("INSERT INTO users (email, password_hash, is_admin) VALUES (?, ?, ?)",('admin', admin_pass_hash, 1))
            cursor.execute("""CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, FOREIGN KEY (user_id) REFERENCES users(id))""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, chat_id INTEGER, sender TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (chat_id) REFERENCES chats(id))""")
            conn.commit()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database error during initialization: {e}")
        sys.exit(1)

# --- RAG and Title Generation System ---
class RAGSystem:
    def __init__(self, model_name):
        self.qa_chain = None
        self.title_chain = None
        self.router_chain = None
        self.general_chain = None
        self.rephrase_chain = None
        self.sql_chain = None 
        self.db = None
        self.model_name = model_name
        self.initialize()
        
    def initialize(self):
        try:
            logger.info("Initializing RAG system...")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"{Fore.GREEN}CUDA device detected. Using GPU: {gpu_name}{Style.RESET_ALL}")
            else:
                logger.warning(f"{Fore.YELLOW}No CUDA device detected. Using CPU.{Style.RESET_ALL}")

            system_prompt = "You are a helpful assistant. Be polite and concise."
            llm = Ollama(model=self.model_name, system=system_prompt)

            # --- 1. Initialize Vector Store (for RAG) ---
            embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
            vector_store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function)
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            qa_template = "Answer the user's question based on the context below. If the context does not contain the answer, use your own knowledge.\n\nContext:\n{context}\n\nQuestion:\n{question}"
            QA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=qa_template)
            self.qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": QA_PROMPT}, return_source_documents=True)

            # --- 2. Initialize Direct SQL Chain ---
            if os.path.exists(STRUCTURED_DB_PATH):
                logger.info(f"Found structured database at {STRUCTURED_DB_PATH}. Initializing SQL Chain...")
                self.db = SQLDatabase.from_uri(f"sqlite:///{STRUCTURED_DB_PATH}")
                
                sql_prompt_template = """You are a SQLite expert. Given a user question, create a single, syntactically correct SQLite query to run.
                To calculate a "total sales figure" or "total revenue", you MUST multiply the unit price column by the units sold column.
                Only use the following table schema: {schema}
                Question: {question}
                SQLQuery:"""
                prompt = ChatPromptTemplate.from_template(sql_prompt_template)
                
                self.sql_chain = (
                    RunnablePassthrough.assign(schema=lambda x: self.db.get_table_info())
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                logger.info("SQL Chain initialized successfully.")
            else:
                logger.warning("Structured database not found. SQL Chain will be disabled.")
            
            # --- 3. Initialize Router ---
            router_template = """Classify the question as "STRUCTURED", "RAG", or "GENERAL". Respond with only a single word.
- If the question is about sales, data, totals, revenue, units, or asks "how many", classify it as "STRUCTURED".
- If the question is math, physics, a greeting, or a general knowledge question, classify it as "GENERAL".
- For all other questions about company policy or product descriptions, classify it as "RAG".
Question: '{question}'
Classification:"""
            ROUTER_PROMPT = PromptTemplate.from_template(router_template)
            self.router_chain = LLMChain(llm=llm, prompt=ROUTER_PROMPT)
            
            # --- 4. Initialize other chains ---
            general_template = "Answer the following question directly and concisely: '{question}'"
            GENERAL_PROMPT = PromptTemplate.from_template(general_template)
            self.general_chain = LLMChain(llm=llm, prompt=GENERAL_PROMPT)
            
            rephrase_template = """Given a chat history and a follow-up question, rephrase the follow-up into a standalone question.
Chat History:
{chat_history}
Follow-up Question: {question}
Standalone Question:"""
            REPHRASE_PROMPT = PromptTemplate.from_template(rephrase_template)
            self.rephrase_chain = LLMChain(llm=llm, prompt=REPHRASE_PROMPT)

            title_template = "Generate a short, concise title (less than 5 words) for the following user question: '{question}'"
            TITLE_PROMPT = PromptTemplate.from_template(title_template)
            self.title_chain = LLMChain(llm=llm, prompt=TITLE_PROMPT)
            logger.info("RAG system initialized successfully.")
        except Exception as e:
            logger.critical(f"FATAL: Could not initialize RAG system. Error: {e}", exc_info=True)
            raise
    
    def rephrase_question(self, question, chat_history_str):
        if not chat_history_str or chat_history_str == "No history yet.":
            return question
        logger.info("Rephrasing question...")
        response = self.rephrase_chain.invoke({"chat_history": chat_history_str, "question": question})
        rephrased = response.get('text', question).strip()
        logger.info(f"Rephrased question: {rephrased}")
        return rephrased
            
    def query(self, user_question, chat_history_str):
        logger.info(f"Original question: '{user_question}'")
        
        try:
            # Route the raw user question first to prevent hallucinations
            classification_result = self.router_chain.invoke({"question": user_question})
            choice = classification_result.get('text', '').strip().upper()
            logger.info(f"Router choice: {choice}")
            
            if "STRUCTURED" in choice and self.sql_chain:
                logger.info("Using SQL chain.")
                # Only rephrase if the question is for the SQL chain
                question_for_sql = self.rephrase_question(user_question, chat_history_str)
                
                raw_llm_output = self.sql_chain.invoke({"question": question_for_sql})
                logger.info(f"LLM Raw Output for SQL: {raw_llm_output}")

                # --- FINAL, ROBUST PARSER ---
                sql_query = ""
                # Method 1: Look for ```sql markdown block
                match = re.search(r"```sql(.*?)```", raw_llm_output, re.DOTALL)
                if match:
                    sql_query = match.group(1).strip()
                # Method 2: Fallback to look for SQLQuery: marker
                elif "SQLQuery:" in raw_llm_output:
                    parts = raw_llm_output.split("SQLQuery:")
                    if len(parts) > 1:
                        sql_query = parts.strip().split('\n')
                # Method 3: If no markers, assume the whole thing is a query
                else:
                    sql_query = raw_llm_output.strip()
                
                # Clean up semicolon which can cause issues
                sql_query = sql_query.replace(";", "")

                if not sql_query:
                     logger.error("Failed to extract a valid SQL query.")
                     return "I'm sorry, I was unable to construct a query for your request.", []

                logger.info(f"Extracted SQL Query: {sql_query}")
                
                try:
                    result = self.db.run(sql_query)
                except Exception as e:
                    logger.error(f"SQL Execution failed: {e}")
                    return "I'm sorry, the data query failed. The database may not contain the information you requested.", []
                
                logger.info(f"SQL Result: {result}")
                
                # --- ENHANCED FINAL ANSWER PROMPT ---
                answer_prompt = f"""Based on the user's question "{question_for_sql}" and the SQL result: "{result}", provide a concise, natural language answer.
                If the result has multiple rows, format it as a clear, easy-to-read list.
                Do not mention SQL. Just give the final answer."""
                llm = Ollama(model=self.model_name)
                final_answer = llm.invoke(answer_prompt)
                return final_answer, []
            
            elif "RAG" in choice:
                logger.info("Using RAG chain.")
                result = self.qa_chain.invoke({"query": user_question})
                return result.get('result'), result.get('source_documents', [])
            
            else: # GENERAL
                logger.info("Using GENERAL chain.")
                result = self.general_chain.invoke({"question": user_question})
                return result.get('text'), []
        
        except Exception as e:
            logger.error(f"A critical error occurred in the query pipeline: {e}", exc_info=True)
            return "I'm sorry, a critical error occurred while processing your request.", []

    def generate_title(self, question):
        if not self.title_chain: return "New Chat"
        try:
            response = self.title_chain.invoke({"question": question})
            return response.get('text', 'New Chat').strip().replace('"', '')
        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return "New Chat"

# --- Flask Web Application ---
app = Flask(__name__)
app.secret_key = os.urandom(24)
rag_system = None

@app.route("/")
def index():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template("index.html", login_id=session.get('login_id'), is_admin=session.get('is_admin', False))

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login_submit():
    login_id = request.form['login_id'].strip()
    password = request.form['password']
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (login_id,))
        user = cursor.fetchone()
    if user and check_password(user['password_hash'], password):
        session['user_id'] = user['id']
        session['login_id'] = user['email']
        session['is_admin'] = bool(user['is_admin'])
        logger.info(f"User '{login_id}' logged in (Admin: {session['is_admin']}).")
        return redirect(url_for('index'))
    logger.warning(f"Failed login attempt for ID: '{login_id}'")
    return "Invalid credentials", 401

@app.route("/logout")
def logout():
    logger.info(f"User '{session.get('login_id')}' logged out.")
    session.clear()
    return redirect(url_for('login_page'))

@app.route("/api/ask", methods=["POST"])
def ask():
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    question = data.get('question')
    chat_id = data.get('chat_id')
    if not question: return jsonify({"error": "No question provided"}), 400
    user_id = session['user_id']
    new_chat_id, new_chat_title = None, None
    chat_history_str = "No history yet."
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        if not chat_id:
            new_chat_title = rag_system.generate_title(question)
            cursor.execute("INSERT INTO chats (user_id, title) VALUES (?, ?)", (user_id, new_chat_title))
            chat_id = cursor.lastrowid
            new_chat_id = chat_id
            conn.commit()
        else:
            history_rows = cursor.execute("SELECT sender, content FROM messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 4", (chat_id,)).fetchall()
            if history_rows:
                # Correctly format chat history
                chat_history_str = "\n".join([f"{row}: {row}" for row in reversed(history_rows)])

        cursor.execute("INSERT INTO messages (chat_id, sender, content) VALUES (?, ?, ?)", (chat_id, 'You', question))
        conn.commit()
    
    answer, sources = rag_system.query(question, chat_history_str)

    formatted_sources = []
    if sources:
        # ... (source formatting is correct) ...
        pass
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO messages (chat_id, sender, content) VALUES (?, ?, ?)", (chat_id, 'Bot', answer))
        conn.commit()
    response = {"answer": answer, "chat_id": chat_id}
    if session.get('is_admin', False) and formatted_sources:
        response['sources'] = formatted_sources
    if new_chat_id:
        response['new_chat_id'], response['new_chat_title'] = new_chat_id, new_chat_title
    return jsonify(response)

# ... (rest of Flask routes are unchanged and correct) ...
@app.route("/api/chats", methods=["GET"])
def get_chats():
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM chats WHERE user_id = ? ORDER BY id DESC", (session['user_id'],))
        return jsonify([dict(row) for row in cursor.fetchall()])
@app.route("/api/chat/<int:chat_id>", methods=["GET"])
def get_chat(chat_id):
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, session['user_id']))
        if not cursor.fetchone(): return jsonify({"error": "Forbidden"}), 403
        cursor.execute("SELECT sender, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,))
        return jsonify([dict(row) for row in cursor.fetchall()])
@app.route("/api/chat/<int:chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM chats WHERE id = ?", (chat_id,))
        result = cursor.fetchone()
        if not result or result != user_id:
            return jsonify({"error": "Forbidden"}), 403
        cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
    logger.info(f"User '{session.get('login_id')}' deleted chat ID {chat_id}.")
    return jsonify({"success": True, "message": "Chat deleted successfully."})
@app.route("/api/change_password", methods=["POST"])
def change_password():
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    current_password, new_password = data.get('current_password'), data.get('new_password')
    if not all([current_password, new_password, data.get('confirm_password')]):
        return jsonify({"error": "All fields are required."}), 400
    if new_password != data.get('confirm_password'):
        return jsonify({"error": "New passwords do not match."}), 400
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.cursor().execute("SELECT password_hash FROM users WHERE id = ?", (session['user_id'],)).fetchone()
    if not user or not check_password(user['password_hash'], current_password):
        return jsonify({"error": "Incorrect current password."}), 403
    new_password_hash = hash_password(new_password)
    with sqlite3.connect(DB_PATH) as conn:
        conn.cursor().execute("UPDATE users SET password_hash = ?, must_change_password = 0 WHERE id = ?", (new_password_hash, session['user_id']))
        conn.commit()
    logger.info(f"User '{session.get('login_id')}' successfully changed their password.")
    return jsonify({"success": True, "message": "Password changed successfully."})
@app.route("/api/admin/add_user", methods=["POST"])
def add_user():
    if not session.get('is_admin'):
        return jsonify({"error": "Forbidden"}), 403
    data = request.json
    email = data.get('email', '').strip()
    password = data.get('password')
    if not email or "@" not in email:
        return jsonify({"error": "A valid email is required"}), 400
    must_change_password = 0
    if not password:
        password = DEFAULT_PASSWORD
        must_change_password = 1
    password_hash = hash_password(password)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (email, password_hash, must_change_password) VALUES (?, ?, ?)", (email, password_hash, must_change_password))
            conn.commit()
        logger.info(f"Admin '{session.get('login_id')}' created user '{email}'.")
        return jsonify({"success": True, "message": f"User '{email}' created."})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already exists"}), 409
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return jsonify({"error": "A server error occurred"}), 500
@app.route("/api/admin/users", methods=["GET"])
def get_users():
    if not session.get('is_admin'):
        return jsonify({"error": "Forbidden"}), 403
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        users = [dict(row) for row in conn.cursor().execute("SELECT id, email FROM users WHERE is_admin = 0 ORDER BY email").fetchall()]
    return jsonify(users)
@app.route("/api/admin/user_chats/<int:user_id>", methods=["GET"])
def get_user_chats(user_id):
    if not session.get('is_admin'):
        return jsonify({"error": "Forbidden"}), 403
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        chats = [dict(row) for row in conn.cursor().execute("SELECT id, title FROM chats WHERE user_id = ? ORDER BY id DESC", (user_id,)).fetchall()]
    return jsonify(chats)
@app.route("/api/admin/chat_messages/<int:chat_id>", methods=["GET"])
def get_chat_messages(chat_id):
    if not session.get('is_admin'):
        return jsonify({"error": "Forbidden"}), 403
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        messages = [dict(row) for row in conn.cursor().execute("SELECT sender, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,)).fetchall()]
    return jsonify(messages)
@app.route("/api/admin/chat/<int:chat_id>", methods=["DELETE"])
def admin_delete_chat(chat_id):
    if not session.get('is_admin'):
        return jsonify({"error": "Forbidden"}), 403
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM chats WHERE id = ?", (chat_id,))
        if not cursor.fetchone():
            return jsonify({"error": "Chat not found"}), 404
        cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
    logger.info(f"Admin '{session.get('login_id')}' deleted chat ID {chat_id}.")
    return jsonify({"success": True, "message": "Chat deleted by admin."})
def main():
    init(autoreset=True)
    parser = argparse.ArgumentParser(description="Run the RAG Project Web GUI")
    parser.add_argument("-m", "--model", dest="ollama_model", type=str, help="Ollama model to use.", required=True)
    try:
        args = parser.parse_args()
        init_db()
        global rag_system
        rag_system = RAGSystem(model_name=args.ollama_model)
        logger.info(f"{Fore.CYAN}Starting Waitress server on http://0.0.0.0:8002{Style.RESET_ALL}")
        logger.info(f"{Fore.YELLOW}Note: For production on Windows, Waitress is a suitable standalone server.{Style.RESET_ALL}")
        serve(app, host='0.0.0.0', port=8002)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Process interrupted by user. Shutting down gracefully.{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print("\n" + Fore.RED + "="*50)
        print("--- FATAL STARTUP ERROR ---")
        print(f"An error occurred that prevented the application from starting: {e}")
        print("Please check your setup (Ollama running, vector DB exists, etc.)")
        print("="*50 + Style.RESET_ALL + "\n")
        logger.critical(f"A critical error occurred during startup: {e}", exc_info=True)
        sys.exit(1)
if __name__ == "__main__":
    main()