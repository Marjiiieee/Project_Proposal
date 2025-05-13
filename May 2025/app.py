import os
import mysql.connector
import time
import re
import PyPDF2
import pdfplumber
import camelot
import uuid
from flask import Flask, request, jsonify, session, g, render_template, redirect, url_for, flash, send_file, send_from_directory
from flask_session import Session
from datetime import timedelta
from db import get_db_connection
from functools import wraps
from werkzeug.utils import secure_filename
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from keybert import KeyBERT
from difflib import SequenceMatcher
import ssl
import certifi
from sentence_transformers import SentenceTransformer, util
from difflib import ndiff
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from datetime import datetime
from typing import List
import spacy
from nltk.tokenize import sent_tokenize
import nltk
import traceback
import base64
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import jiwer
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, models
# Import enhanced context similarity module
from enhanced_context_similarity import check_thesis_similarity_route_handler, extract_introduction

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# BASE PATH for MODELS
BASE_PATH = os.path.dirname(__file__)

# Initialize context similarity model
context_model_path = os.path.join(BASE_PATH, "models", "contextsim_model")
if os.path.exists(context_model_path):
    try:
        print("Wrapping Hugging Face model for SentenceTransformer...")
        word_embedding_model = models.Transformer(context_model_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        context_sim_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print(f"Custom context similarity model loaded on: {'GPU' if device.type == 'cuda' else 'CPU'}")
    except Exception as e:
        print(f"Error wrapping Hugging Face model: {e}")
        context_sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        print("Fallback context model loaded instead.")
else:
    print(f"Context similarity model not found at {context_model_path}, using fallback")
    context_sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    print(f"Fallback context model loaded on: {'GPU' if device.type == 'cuda' else 'CPU'}")

sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
try:
    custom_model_path = os.path.join(BASE_PATH, "models", "similarity_model")
    print(f"Loading custom similarity model from {custom_model_path}")
    custom_similarity_model = SentenceTransformer(custom_model_path)
    print("Custom similarity model loaded successfully")
except Exception as e:
    print(f"Error loading custom similarity model: {e}")
    custom_similarity_model = sentence_model  # fallback

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Initialize spaCy - with fallback to simpler analysis if not available
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    nlp = None

# fine_tuned_path = r"C:\NOT_MINE\MARCH 2025\flan-t5-title-finetuned"
# title_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
# title_model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_path)

# Initialize title generation models
try:
    print("Loading title generation models...")
    title_tokenizer = AutoTokenizer.from_pretrained("EngLip/flan-t5-sentence-generator")
    title_model = AutoModelForSeq2SeqLM.from_pretrained("EngLip/flan-t5-sentence-generator")
    # Move model to GPU if available
    title_model = title_model.to(device)
    title_generator = pipeline("text2text-generation", model=title_model, tokenizer=title_tokenizer,
                              device=0 if device.type == "cuda" else -1)
    print("Title generation models loaded successfully")
except Exception as e:
    print(f"Error loading title generation models: {e}")
    # Fallback to simpler models
    try:
        print("Attempting to load fallback title models...")
        title_tokenizer = AutoTokenizer.from_pretrained("t5-base")
        title_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        # Move model to GPU if available
        title_model = title_model.to(device)
        title_generator = pipeline("text2text-generation", model=title_model, tokenizer=title_tokenizer,
                                  device=0 if device.type == "cuda" else -1)
        print("Fallback title models loaded successfully")
    except Exception as e:
        print(f"Error loading fallback title models: {e}")
        title_tokenizer = None
        title_model = None
        title_generator = None


# Configure SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Initialize models with error handling
try:
    print("Loading sentence transformer model...")
    # Initialize sentence transformer model
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    print(f"Sentence transformer model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading sentence transformer model: {e}")
    # Fallback to a simpler model
    try:
        print("Attempting to load fallback sentence transformer model...")
        sentence_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=device)
        print(f"Fallback sentence transformer model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading fallback sentence transformer model: {e}")
        print("WARNING: No sentence transformer model available. Using dummy model.")
        # Create a dummy model that returns random embeddings
        class DummyModel:
            def encode(self, text, **kwargs):
                import numpy as np
                # Return a random embedding of the right size
                return np.random.rand(384)  # Standard size for many models
        sentence_model = DummyModel()

# Initialize grammar correction model with fallback
try:
    grammar_tokenizer = T5Tokenizer.from_pretrained(
        "vennify/t5-base-grammar-correction",
        legacy=True,
        local_files_only=False
    )
    grammar_model = T5ForConditionalGeneration.from_pretrained(
        "vennify/t5-base-grammar-correction",
        local_files_only=False
    )
    # Move model to GPU if available
    grammar_model = grammar_model.to(device)
    print(f"Grammar correction model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading grammar correction model: {e}")
    try:
        # Fallback to t5-small
        grammar_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
        grammar_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        # Move model to GPU if available
        grammar_model = grammar_model.to(device)
        print(f"Fallback grammar model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading fallback grammar model: {e}")
        grammar_tokenizer = None
        grammar_model = None

# Note: title_tokenizer is already initialized above

# Initialize KeyBERT with fallback
try:
    print("Loading KeyBERT model...")
    # KeyBERT will use the device from the underlying model
    kw_model = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")
    print("KeyBERT model loaded successfully")
except Exception as e:
    print(f"Error loading KeyBERT model: {e}")
    # Create a dummy KeyBERT model
    class DummyKeyBERT:
        def extract_keywords(self, text, **kwargs):
            # Extract simple keywords based on frequency
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 4 and word not in ['about', 'these', 'those', 'their', 'there']:
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Get the most frequent words
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            top_n = kwargs.get('top_n', 5)
            return [(word, 1.0) for word, _ in sorted_words[:top_n]]
    kw_model = DummyKeyBERT()
USE_EMOJIS_IN_LOGS = False  # Set to True if you want emojis in your debug logs

# Cache for thesis dataset
# Global cache for thesis dataset and embeddings
cached_thesis_data = None
cached_embeddings = None

# ✅ Load dataset and compute embeddings without modifying DataFrame
def load_thesis_dataset_with_embeddings(model):
    try:
        # Try to load from the main dataset path
        dataset_path = os.path.join(BASE_PATH, 'Dataset - thesis.csv')
        print(f"Attempting to load dataset from: {dataset_path}")

        if not os.path.exists(dataset_path):
            print(f"Main dataset not found at: {dataset_path}")

            # Try alternative locations
            alt_paths = [
                os.path.join(BASE_PATH, 'Project-main', 'Dataset - thesis.csv'),
                os.path.join(BASE_PATH, 'data', 'Dataset - thesis.csv'),
                os.path.join(BASE_PATH, 'PropEase_Dataset.xlsx')
            ]

            for alt_path in alt_paths:
                print(f"Trying alternative path: {alt_path}")
                if os.path.exists(alt_path):
                    dataset_path = alt_path
                    print(f"Found dataset at alternative path: {dataset_path}")
                    break
            else:
                # If we get here, none of the paths worked
                print("Could not find dataset in any of the expected locations")

                # Create a minimal dataset as a last resort
                print("Creating a minimal dataset as fallback")
                minimal_data = {
                    'title': [
                        "Web-Based Student Information System",
                        "Mobile Application for Healthcare Monitoring",
                        "Automated Attendance System Using Facial Recognition"
                    ],
                    'abstract': [
                        "This thesis presents the development of a web-based student information system designed to streamline administrative processes in educational institutions.",
                        "This research focuses on the development of a mobile application for real-time healthcare monitoring using IoT sensors.",
                        "This thesis presents an automated attendance system using facial recognition technology and deep learning algorithms."
                    ]
                }
                df = pd.DataFrame(minimal_data)
                df['content'] = df['title'] + ' ' + df['abstract']
                print(f"Created minimal dataset with {len(df)} rows")

                # Encode the minimal dataset
                try:
                    embeddings = model.encode(df['content'].tolist(), convert_to_tensor=True)
                    return df, embeddings
                except Exception as e:
                    print(f"Error encoding minimal dataset: {e}")
                    return df, None

        # If we found a dataset, load it
        print(f"Loading dataset from: {dataset_path}")

        # Handle Excel files
        if dataset_path.endswith('.xlsx'):
            df = pd.read_excel(dataset_path).fillna('')
        else:
            df = pd.read_csv(dataset_path, encoding='utf-8', on_bad_lines='skip').fillna('')

        # Ensure required columns exist
        if 'title' not in df.columns or 'abstract' not in df.columns:
            print(f"Warning: Required columns missing. Available columns: {df.columns.tolist()}")

            # Try to identify appropriate columns
            title_candidates = [col for col in df.columns if 'title' in col.lower()]
            abstract_candidates = [col for col in df.columns if any(term in col.lower() for term in ['abstract', 'description', 'content', 'text'])]

            if title_candidates and abstract_candidates:
                print(f"Using {title_candidates[0]} as title and {abstract_candidates[0]} as abstract")
                df = df.rename(columns={
                    title_candidates[0]: 'title',
                    abstract_candidates[0]: 'abstract'
                })
            else:
                # Create dummy columns if needed
                if 'title' not in df.columns:
                    print("Creating dummy 'title' column")
                    df['title'] = df.iloc[:, 0].astype(str)

                if 'abstract' not in df.columns:
                    print("Creating dummy 'abstract' column")
                    # Use the second column or concatenate all columns if there's no obvious abstract
                    if len(df.columns) > 1:
                        df['abstract'] = df.iloc[:, 1].astype(str)
                    else:
                        df['abstract'] = df.iloc[:, 0].astype(str)

        # Create content column for embedding
        df['content'] = (
            df['title'].astype(str) + ' ' +
            df['abstract'].astype(str)
        )

        print(f"Successfully loaded dataset with {len(df)} rows")
        print("Encoding thesis content...")

        # Encode in smaller batches to avoid memory issues
        try:
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(df), batch_size):
                batch = df['content'].iloc[i:i+batch_size].tolist()
                batch_embeddings = model.encode(batch, convert_to_tensor=True)
                all_embeddings.append(batch_embeddings)
                print(f"Encoded batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")

            # Combine all batches
            if all_embeddings:
                embeddings = torch.cat(all_embeddings, dim=0)
            else:
                embeddings = torch.tensor([])

            return df, embeddings

        except Exception as e:
            print(f"Error during batch encoding: {e}")
            # Return the dataframe even if encoding failed
            return df, None

    except Exception as e:
        print(f"❌ Error loading thesis dataset: {e}")
        traceback.print_exc()

        # Create an empty DataFrame with the required columns as a last resort
        print("Creating empty dataset with required columns")
        df = pd.DataFrame(columns=['title', 'abstract', 'content'])
        return df, None

def initialize_thesis_data():
    global cached_thesis_data, cached_embeddings
    try:
        print("Initializing thesis dataset and embeddings...")
        cached_thesis_data, cached_embeddings = load_thesis_dataset_with_embeddings(custom_similarity_model)
        if cached_thesis_data is not None and not cached_thesis_data.empty:
            print(f"✅ Cached {len(cached_thesis_data)} thesis entries")
        else:
            print("⚠️ Thesis dataset is empty or None")

            # Try to create a minimal dataset as fallback
            print("Creating minimal dataset as fallback")
            minimal_data = {
                'title': [
                    "Web-Based Student Information System",
                    "Mobile Application for Healthcare Monitoring",
                    "Automated Attendance System Using Facial Recognition"
                ],
                'abstract': [
                    "This thesis presents the development of a web-based student information system designed to streamline administrative processes in educational institutions.",
                    "This research focuses on the development of a mobile application for real-time healthcare monitoring using IoT sensors.",
                    "This thesis presents an automated attendance system using facial recognition technology and deep learning algorithms."
                ]
            }
            cached_thesis_data = pd.DataFrame(minimal_data)
            cached_thesis_data['content'] = cached_thesis_data['title'] + ' ' + cached_thesis_data['abstract']
            print(f"Created minimal dataset with {len(cached_thesis_data)} rows")

            # Try to encode the minimal dataset
            try:
                if custom_similarity_model is not None:
                    cached_embeddings = custom_similarity_model.encode(cached_thesis_data['content'].tolist(), convert_to_tensor=True)
                    print("Successfully encoded minimal dataset")
                else:
                    print("No similarity model available for encoding")
                    cached_embeddings = None
            except Exception as e:
                print(f"Error encoding minimal dataset: {e}")
                cached_embeddings = None
    except Exception as e:
        print(f"❌ Error caching thesis dataset: {e}")
        traceback.print_exc()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")  # Use environment variable for security
app.config['SESSION_TYPE'] = 'filesystem'  # Store session data on the server
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file upload limit
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
app.config['SESSION_PERMANENT'] = True

# Initialize Flask-Session correctly before running the app
Session(app)

# Cache for extracted text
extracted_texts = {}

# Initialize thesis data at startup
print("Initializing thesis data at application startup...")
initialize_thesis_data()

ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database connection using Flask `g`
def get_db():
    if 'db' not in g:
        g.db = get_db_connection()
    return g.db

@app.teardown_appcontext
def close_db(_=None):  # Using _ for unused parameter
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template("signin.html")

    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required"}), 400

    # Validate email domain
    valid_domains = ['gmail.com', 'yahoo.com', 'edu.ph']
    email_domain = email.lower().split('@')[-1]
    if email_domain not in valid_domains:
        return jsonify({"status": "error", "message": "Please use a valid email domain"}), 400

    # Validate password
    if len(password) < 8:
        return jsonify({"status": "error", "message": "Password must be at least 8 characters long"}), 400

    if len(re.findall(r'\d', password)) < 2:
        return jsonify({"status": "error", "message": "Password must contain at least 2 numbers"}), 400

    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return jsonify({"status": "error", "message": "Password must contain at least 1 special character"}), 400

    try:
        conn = get_db()
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute("SELECT email FROM registration WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({"status": "error", "message": "Email already registered"}), 400

        # Insert user data into the database
        cursor.execute("INSERT INTO registration (email, password) VALUES (%s, %s)", (email, password))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"status": "success", "message": "Registration successful!"})

    except mysql.connector.Error as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Login Required
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            flash("You must be logged in to access this page", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def main():
    return render_template("main.html")

# LOG-IN
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email:
            return jsonify({
                "status": "error",
                "field": "email",
                "message": "Please enter your email"
            })

        if not password:
            return jsonify({
                "status": "error",
                "field": "password",
                "message": "Please enter your password"
            })

        try:
            conn = get_db()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM registration WHERE email = %s", (email,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if not user:
                return jsonify({
                    "status": "error",
                    "field": "email",
                    "message": "Email not found. Check your email or Sign up."
                })

            if user["password"] != password:  # Note: You should use proper password hashing
                return jsonify({
                    "status": "error",
                    "field": "password",
                    "message": "Incorrect password. Please try again."
                })

            # Success case
            session.permanent = True
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            return jsonify({
                "status": "success",
                "message": "Login successful!",
                "redirect": url_for('home')
            })

        except Exception as e:
            print(f"Login error: {str(e)}")
            return jsonify({
                "status": "error",
                "field": "email",
                "message": "An error occurred. Please try again later."
            })

    return render_template("main.html")

# HOME
@app.route('/home')
@login_required
def home():
    return render_template("home.html")

# LOG-OUT
@app.route('/logout')
def logout():
    session.clear()  # Clear session data
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# PROPOSAL UPLOAD
@app.route('/proposal_upload', methods=['GET', 'POST'])
@login_required
def proposal_upload():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"})

        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"})

        if file and allowed_file(file.filename):
            try:
                # Keep original filename but make it secure
                original_filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}{os.path.splitext(original_filename)[1]}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                # Save the file
                file.save(file_path)

                # Extract text using both pdfplumber and PyPDF2 for better results
                extracted_text = ""

                # Try pdfplumber first
                try:
                   with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text(layout=True)
                        if text:
                            extracted_text += text + "\n"
                except Exception as e:
                    print(f"pdfplumber extraction error: {e}")

                # If pdfplumber fails or extracts no text, try PyPDF2
                if not extracted_text.strip():
                    try:
                        with open(file_path, "rb") as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text:
                                    extracted_text += text + "\n"
                    except Exception as e:
                        print(f"PyPDF2 extraction error: {e}")

                # Always try to extract tables using camelot
                try:
                    tables = camelot.read_pdf(file_path, pages='all')
                    table_text = "\n\n".join(table.df.to_string() for table in tables)
                    if table_text.strip():
                        extracted_text += "\n\nTables:\n" + table_text
                except Exception as e:
                    print(f"Camelot extraction error: {e}")

                # Verification after all extraction attempts
                if not extracted_text.strip():
                    print("Warning: No text could be extracted from the PDF")
                    extracted_text = "No text could be extracted from this PDF."

                # Save to database with extracted text
                cursor.execute("""
                    INSERT INTO files (user_email, file_name, file_path, extracted_text, archived)
                    VALUES (%s, %s, %s, %s, %s)
                """, (session['user_email'], original_filename, file_path, extracted_text, False))

                file_id = cursor.lastrowid
                conn.commit()

                return jsonify({
                    "status": "success",
                    "message": "File uploaded successfully",
                    "file_id": file_id,
                    "original_filename": original_filename,
                    "text_length": len(extracted_text)
                })

            except Exception as e:
                print(f"Error during file processing: {str(e)}")
                return jsonify({"status": "error", "message": f"Error processing file: {str(e)}"}), 500
            finally:
                cursor.close()
                conn.close()

        return jsonify({"status": "error", "message": "Invalid file type"}), 400

    # GET request - display the upload page
    cursor.execute("""
        SELECT id, file_name, file_path
        FROM files
        WHERE user_email = %s AND (archived = FALSE OR archived IS NULL)
    """, (session['user_email'],))
    files = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('proposal_upload.html', files=files)

@app.route('/uploaded_file')
@login_required
def uploaded_file():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # Explicitly only fetch non-archived files
    cursor.execute("""
        SELECT id, file_name, file_path, archived
        FROM files
        WHERE user_email = %s AND archived = FALSE
    """, (session['user_email'],))

    files = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify(files)

# PROPOSAL VIEW
@app.route('/view_proposal/<int:file_id>')
@login_required
def view_proposal(file_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT file_path, extracted_text, analysis_json
        FROM files
        WHERE id = %s AND user_email = %s
    """, (file_id, session['user_email']))

    file_record = cursor.fetchone()
    cursor.close()
    conn.close()

    if file_record:
        # Get file path from record
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(file_record['file_path']))
        absolute_path = os.path.abspath(full_path)

        # Check file existence and encode for rendering
        if not os.path.exists(absolute_path):
            flash("PDF file not found on the server.", "error")
            return redirect(url_for('proposal_upload'))

        with open(absolute_path, 'rb') as f:
            encoded_pdf = base64.b64encode(f.read()).decode('utf-8')

        return render_template(
            'proposal_view.html',
            encoded_pdf=encoded_pdf,
            extracted_text=file_record['extracted_text'],
            analysis_json=file_record['analysis_json'] or '{}'
        )

    else:
        flash("File not found or Access Denied", "error")
        return redirect(url_for('proposal_upload'))


def extract_enhanced_keywords(text, top_n=10, ignore_phrases=None):
    """
    Enhanced keyword extraction that splits text into paragraphs,
    processes each paragraph, and combines results while ignoring specified phrases.
    Avoids repetition of similar keywords.

    Args:
        text (str): The text to extract keywords from
        top_n (int): Number of keywords to extract
        ignore_phrases (list): List of phrases to ignore

    Returns:
        list: List of extracted keywords
    """
    if ignore_phrases is None:
        ignore_phrases = [
            "laguna state polytechnic university",
            "province of laguna",
            "college of computer studies",
            "lspu", "santa cruz", "los baños"
        ]

    # Additional domain-specific stopwords for academic papers
    domain_stopwords = [
        "chapter", "introduction", "conclusion", "abstract", "references",
        "methodology", "results", "discussion", "figure", "table", "appendix",
        "et al", "et", "al", "i.e", "e.g", "etc", "university", "college",
        "study", "research", "paper", "thesis", "dissertation", "project",
        "student", "professor", "faculty", "department", "school", "institute"
    ]

    try:
        print(f"Extracting enhanced keywords from text of length {len(text)}")

        # Clean text by removing ignored phrases
        clean_text = text.lower()
        for phrase in ignore_phrases:
            clean_text = clean_text.replace(phrase.lower(), "")

        # Split into paragraphs
        paragraphs = clean_text.split("\n")
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]

        # If we have too many paragraphs, focus on the first few, middle, and last few
        if len(paragraphs) > 10:
            print(f"Text has {len(paragraphs)} paragraphs, focusing on key sections")
            middle_idx = len(paragraphs) // 2
            paragraphs = paragraphs[:4] + paragraphs[middle_idx-1:middle_idx+1] + paragraphs[-3:]

        # Process each paragraph
        all_keywords = []

        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 10000:
                paragraph = paragraph[:10000]  # Limit paragraph length

            print(f"Processing paragraph {i+1}/{len(paragraphs)}, length: {len(paragraph)}")

            try:
                # Extract keywords with different n-gram ranges
                # First try 1-2 word phrases
                keywords_1_2 = kw_model.extract_keywords(
                    paragraph,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    use_mmr=True,  # Use Maximal Marginal Relevance for diversity
                    diversity=0.5,
                    top_n=5
                )

                # Then try 2-3 word phrases for more specific terms
                keywords_2_3 = kw_model.extract_keywords(
                    paragraph,
                    keyphrase_ngram_range=(2, 3),
                    stop_words='english',
                    use_mmr=True,
                    diversity=0.5,
                    top_n=3
                )

                # Combine results
                keywords = keywords_1_2 + keywords_2_3

                # Filter out domain-specific stopwords
                keywords = [(kw, score) for kw, score in keywords
                           if not any(stopword == kw.lower() for stopword in domain_stopwords)]

                # Add to our collection
                all_keywords.extend(keywords)
                print(f"Extracted {len(keywords)} keywords from paragraph {i+1}")

            except Exception as e:
                print(f"Error extracting keywords from paragraph {i+1}: {e}")
                continue

        # Sort by score and remove duplicates
        all_keywords.sort(key=lambda x: x[1], reverse=True)

        # Remove duplicates and similar keywords while preserving order
        seen = set()
        unique_keywords = []
        for kw, score in all_keywords:
            kw_lower = kw.lower()
            # Skip if we've seen this exact keyword
            if kw_lower in seen:
                continue

            # Skip if we've seen a similar keyword (using a stricter threshold)
            if any(is_similar(kw_lower, seen_kw, 0.85) for seen_kw in seen):
                continue

            seen.add(kw_lower)
            unique_keywords.append((kw, score))

        # Get top N keywords
        result = [kw.lower() for kw, _ in unique_keywords[:top_n]]

        # Post-processing: Try to match with common technical terms
        tech_terms = set([
            "machine learning", "artificial intelligence", "deep learning",
            "neural network", "data mining", "big data", "cloud computing",
            "internet of things", "iot", "blockchain", "cybersecurity",
            "virtual reality", "augmented reality", "mobile application",
            "web application", "database", "algorithm", "data science",
            "natural language processing", "computer vision", "robotics",
            "automation", "software engineering", "network security",
            "information system", "data analytics", "user interface",
            "user experience", "ui/ux", "mobile development", "web development"
        ])

        # Check if any technical terms appear in the text
        for term in tech_terms:
            if term in clean_text and term not in result and len(result) < top_n:
                result.append(term)

        print(f"Final enhanced keywords: {result}")
        return result

    except Exception as e:
        print(f"Error in extract_enhanced_keywords: {e}")
        traceback.print_exc()
        # Return default keywords in case of any error
        default_keywords = ["system", "data", "analysis", "research", "technology"]
        print(f"Returning default keywords: {default_keywords}")
        return default_keywords

def summarize_discrepancies(paper_text, speech_text):
    """
    Summarize discrepancies between paper and speech by extracting keywords
    from the technical description of the paper and comparing with speech.

    Args:
        paper_text (str): The text extracted from the paper
        speech_text (str): The text from the speech/presentation

    Returns:
        tuple: (missed_in_speech, added_in_speech) - keywords missed or added in speech
    """
    try:
        print("Summarizing discrepancies between paper and speech...")

        # Extract technical description from the paper
        technical_description = extract_technical_description(paper_text)
        print(f"Extracted technical description for keyword analysis ({len(technical_description)} chars)")

        # Extract keypoints with enhanced method
        try:
            print("Extracting paper keypoints from technical description...")
            paper_keywords = extract_enhanced_keywords(technical_description, top_n=10)
            print(f"Extracted {len(paper_keywords)} paper keypoints")
        except Exception as e:
            print(f"Error extracting paper keypoints: {e}")
            paper_keywords = ["system", "data", "analysis", "research", "technology"]

        try:
            print("Extracting speech keypoints...")
            speech_keywords = extract_enhanced_keywords(speech_text, top_n=10)
            print(f"Extracted {len(speech_keywords)} speech keypoints")
        except Exception as e:
            print(f"Error extracting speech keypoints: {e}")
            speech_keywords = ["presentation", "overview", "summary", "explanation", "discussion"]

        # Find discrepancies (avoid repetition)
        missed_in_speech = []
        for kw in paper_keywords:
            # Check if this keyword or a similar one is in speech_keywords
            if kw not in speech_keywords and not any(is_similar(kw, sk, 0.8) for sk in speech_keywords):
                # Also check it's not already in our missed list or similar to something there
                if not any(is_similar(kw, mk, 0.8) for mk in missed_in_speech):
                    missed_in_speech.append(kw)

        added_in_speech = []
        for kw in speech_keywords:
            # Check if this keyword or a similar one is in paper_keywords
            if kw not in paper_keywords and not any(is_similar(kw, pk, 0.8) for pk in paper_keywords):
                # Also check it's not already in our added list or similar to something there
                if not any(is_similar(kw, ak, 0.8) for ak in added_in_speech):
                    added_in_speech.append(kw)

        print(f"Found {len(missed_in_speech)} unique missed keypoints and {len(added_in_speech)} unique added keypoints")

        return missed_in_speech[:5], added_in_speech[:5]
    except Exception as e:
        print(f"Error in summarize_discrepancies: {e}")
        traceback.print_exc()
        return ["Error analyzing paper"], ["Error analyzing speech"]


def extract_technical_description(text):
    """Extract the technical description section from the concept paper"""
    try:
        print(f"Extracting technical description from text of length {len(text)}")

        # Look for sections specific to concept papers - expanded patterns
        patterns = [
            # Standard technical sections
            r'(?i)technical\s+description.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)methodology.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)implementation.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)system\s+design.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)proposed\s+solution.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)approach.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)system\s+architecture.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)design.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)development.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)algorithm.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)framework.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)technology.*?(?=\n\s*\n\s*[A-Z]|\Z)',

            # Additional patterns for concept papers
            r'(?i)system\s+implementation.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)system\s+development.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)proposed\s+methodology.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)proposed\s+framework.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)technical\s+approach.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)development\s+methodology.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)software\s+architecture.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)hardware\s+requirements.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)software\s+requirements.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)system\s+requirements.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)functional\s+requirements.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)non-functional\s+requirements.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)technical\s+framework.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)implementation\s+details.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)development\s+process.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)technical\s+solution.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)solution\s+architecture.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)design\s+and\s+implementation.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)research\s+methodology.*?(?=\n\s*\n\s*[A-Z]|\Z)',

            # Try to match section headers with numbers
            r'(?i)(?:3|III|3\.0)[\.\s]+(?:methodology|implementation|system design).*?(?=\n\s*\n\s*[0-9]+[\.\s]+[A-Z]|\Z)',
            r'(?i)(?:4|IV|4\.0)[\.\s]+(?:methodology|implementation|system design).*?(?=\n\s*\n\s*[0-9]+[\.\s]+[A-Z]|\Z)',
            r'(?i)(?:5|V|5\.0)[\.\s]+(?:methodology|implementation|system design).*?(?=\n\s*\n\s*[0-9]+[\.\s]+[A-Z]|\Z)'
        ]

        # Try each pattern
        for pattern in patterns:
            matches = re.search(pattern, text, re.DOTALL)
            if matches:
                extracted = matches.group(0).strip()
                print(f"Found technical section using pattern: {pattern[:30]}...")
                print(f"Extracted {len(extracted)} characters")
                return extracted

        # Try to find sections with technical keywords - expanded list
        tech_keywords = [
            # General technical terms
            "algorithm", "system", "database", "interface", "module",
            "function", "API", "architecture", "framework", "technology",
            "implementation", "code", "software", "hardware", "network",
            "protocol", "data", "model", "neural", "machine learning",

            # More specific technical terms
            "frontend", "backend", "middleware", "microservice", "REST API",
            "SOAP", "JSON", "XML", "HTTP", "HTTPS", "TCP/IP", "UDP",
            "encryption", "authentication", "authorization", "security",
            "cloud", "server", "client", "web service", "mobile app",
            "desktop application", "responsive design", "user interface",
            "user experience", "UX", "UI", "database schema", "SQL",
            "NoSQL", "MongoDB", "MySQL", "PostgreSQL", "Oracle",
            "data structure", "algorithm complexity", "time complexity",
            "space complexity", "big O notation", "neural network",
            "convolutional", "recurrent", "transformer", "BERT", "GPT",
            "natural language processing", "computer vision", "image recognition",
            "object detection", "sentiment analysis", "classification",
            "regression", "clustering", "dimensionality reduction",
            "feature extraction", "feature engineering", "preprocessing",
            "postprocessing", "validation", "testing", "unit test",
            "integration test", "system test", "acceptance test",
            "continuous integration", "continuous deployment", "CI/CD",
            "version control", "git", "agile", "scrum", "waterfall",
            "sprint", "backlog", "user story", "use case", "requirement",
            "specification", "documentation", "API documentation",
            "technical documentation", "user manual", "administrator guide",
            "developer guide", "installation guide", "configuration guide",
            "troubleshooting guide", "maintenance guide", "support guide"
        ]

        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        technical_paragraphs = []

        # Score each paragraph based on technical keyword density
        paragraph_scores = []
        for i, para in enumerate(paragraphs):
            if len(para.strip()) < 50:  # Skip very short paragraphs
                continue

            # Count technical keywords in this paragraph
            keyword_count = sum(1 for keyword in tech_keywords if keyword.lower() in para.lower())
            # Calculate density (keywords per 100 words)
            word_count = len(para.split())
            if word_count > 0:
                density = (keyword_count * 100) / word_count
                paragraph_scores.append((i, density, para))

        # Sort paragraphs by technical keyword density (highest first)
        paragraph_scores.sort(key=lambda x: x[1], reverse=True)

        # Take the top 5 paragraphs with highest technical density
        top_paragraphs = paragraph_scores[:5]

        # If we found technical paragraphs, combine them
        if top_paragraphs:
            # Sort by original order to maintain document flow
            top_paragraphs.sort(key=lambda x: x[0])
            technical_paragraphs = [p[2] for p in top_paragraphs]
            combined = "\n\n".join(technical_paragraphs)
            print(f"Found {len(technical_paragraphs)} paragraphs with high technical keyword density")
            print(f"Combined length: {len(combined)} characters")
            return combined

        # If no specific section found, use the middle portion of the document
        # (often contains technical details)
        lines = text.split('\n')
        if len(lines) > 10:
            start_idx = len(lines) // 3
            end_idx = (len(lines) * 2) // 3
            middle_portion = '\n'.join(lines[start_idx:end_idx])
            print(f"Using middle portion of document: {len(middle_portion)} characters")
            return middle_portion

        print(f"Using full text as technical description: {len(text)} characters")
        return text  # Return full text if it's short
    except Exception as e:
        print(f"Error extracting technical description: {e}")
        traceback.print_exc()
        return text  # Return full text on error

def extract_abstract(text):
    """Extract the abstract section from a thesis"""
    try:
        # Look for abstract section
        patterns = [
            r'(?i)abstract.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)summary.*?(?=\n\s*\n\s*[A-Z]|\Z)',
            r'(?i)overview.*?(?=\n\s*\n\s*[A-Z]|\Z)'
        ]

        for pattern in patterns:
            matches = re.search(pattern, text, re.DOTALL)
            if matches:
                return matches.group(0).strip()

        # If no abstract found, use the beginning of the document
        lines = text.split('\n')
        if len(lines) > 10:
            return '\n'.join(lines[:min(15, len(lines))])

        return text  # Return full text if it's short
    except Exception as e:
        print(f"Error extracting abstract: {e}")
        return text  # Return full text on error

def get_thesis_data_from_csv():
    """Load thesis data from the CSV file instead of the database"""
    try:
        csv_path = os.path.join(BASE_PATH, 'Dataset - thesis.csv')
        print(f"Loading thesis data from CSV: {csv_path}")

        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return []

        # Read the CSV file
        import pandas as pd
        df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
        print(f"Successfully loaded {len(df)} thesis records from CSV")

        # Convert DataFrame to list of dictionaries
        thesis_data = []
        for _, row in df.iterrows():
            try:
                title = row.get('title', '')
                abstract = row.get('abstract', '')

                if pd.isna(title) or pd.isna(abstract):
                    continue

                thesis_data.append({
                    'title': str(title).strip(),
                    'text': str(abstract).strip()
                })
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        print(f"Processed {len(thesis_data)} valid thesis records")
        return thesis_data

    except Exception as e:
        print(f"Error loading thesis data from CSV: {e}")
        traceback.print_exc()
        return []  # Return empty list on error

@app.route('/analyze_content/<int:file_id>', methods=['POST'])
@login_required
def analyze_content(file_id):
    print(f"=== STARTING NEW ANALYSIS FOR FILE ID: {file_id} ===")

    # Set default fallback results
    fallback_results = {
        'suggested_titles': [
            "Automated System for Data Analysis and Processing",
            "Intelligent Framework for Information Management",
            "Digital Solution for Streamlined Workflow Optimization",
            "Integrated Platform for Enhanced User Experience",
            "Smart Application for Efficient Resource Utilization"
        ],
        'missed_keypoints': ["methodology", "implementation", "results", "analysis", "conclusion"],
        'added_keypoints': ["overview", "introduction", "background", "summary", "future work"]
    }

    try:
        # Get the data from the request
        data = request.json
        extracted_text = data.get('extracted_text', '')
        speech_text = data.get('speech_text', '')

        # Validate inputs
        if not extracted_text:
            return jsonify({'status': 'error', 'message': 'No extracted text provided'}), 400

        print(f"Received extracted text ({len(extracted_text)} chars) and speech text ({len(speech_text)} chars)")

        # --- Speech Similarity ---
        if speech_text.strip():
            try:
                print("Calculating speech similarity...")

                # ========== SEMANTIC SIMILARITY (Proposal vs Speech) ==========
                proposal_embedding = sentence_model.encode(extracted_text)
                speech_embedding = sentence_model.encode(speech_text)
                # Calculate raw cosine similarity
                raw_similarity = float(util.pytorch_cos_sim(
                    proposal_embedding.reshape(1, -1),
                    speech_embedding.reshape(1, -1)
                )[0][0] * 100)
                # Apply scaling factor to reduce similarity (multiply by 0.6)
                speech_similarity = raw_similarity * 0.6
                speech_similarity = round(speech_similarity, 2)
                print(f"Speech similarity (scaled): {speech_similarity}% (raw: {round(raw_similarity, 2)}%)")

                speech_is_similar = speech_similarity >= 60.0
                print(f"Speech is considered {'similar' if speech_is_similar else 'not similar'} (threshold: 60%)")

            except Exception as e:
                print(f"Error calculating speech similarity: {e}")
                traceback.print_exc()
                speech_similarity = None
                speech_is_similar = None
        else:
            print("No speech input provided, skipping similarity check.")
            speech_similarity = None
            speech_is_similar = None

        # --- Thesis Similarity ---
        try:
            print("Calculating thesis similarity...")

            # Extract technical description from the concept paper
            technical_description = extract_technical_description(extracted_text)
            print(f"Extracted technical description ({len(technical_description)} chars)")

            # Get thesis data from CSV instead of database
            if cached_thesis_data is not None and not cached_thesis_data.empty:
                thesis_data = [
                    {'title': row['title'], 'text': row['abstract']}
                    for _, row in cached_thesis_data.iterrows()
                    if pd.notna(row['title']) and pd.notna(row['abstract'])
                ]
            else:
                print("Cached thesis data is None or empty, trying to load directly")
                df, _ = load_thesis_dataset_with_embeddings(custom_similarity_model)

                if df is not None and not df.empty:
                    print(f"Successfully loaded thesis dataset with {len(df)} rows")
                    thesis_data = [
                        {'title': row['title'], 'text': row['abstract']}
                        for _, row in df.iterrows()
                        if pd.notna(row['title']) and pd.notna(row['abstract'])
                    ]
                else:
                    print("Could not load thesis dataset, creating minimal fallback data")
                    # Create minimal fallback data
                    thesis_data = [
                        {
                            'title': "Web-Based Student Information System",
                            'text': "This thesis presents the development of a web-based student information system designed to streamline administrative processes in educational institutions."
                        },
                        {
                            'title': "Mobile Application for Healthcare Monitoring",
                            'text': "This research focuses on the development of a mobile application for real-time healthcare monitoring using IoT sensors."
                        },
                        {
                            'title': "Automated Attendance System Using Facial Recognition",
                            'text': "This thesis presents an automated attendance system using facial recognition technology and deep learning algorithms."
                        }
                    ]
                    print(f"Created minimal fallback dataset with {len(thesis_data)} entries")

            if not thesis_data:
                print("No thesis data available after all attempts, using empty list")
                thesis_similarity = 0
                similar_titles_by_abstract = []
                print("WARNING: No thesis data available, but continuing with analysis")

            # Use context similarity model if available, otherwise use sentence_model
            similarity_model = context_sim_model if context_sim_model is not None else sentence_model
            print(f"Using {'context similarity' if context_sim_model is not None else 'sentence transformer'} model")

            # Encode the technical description once
            tech_desc_embedding = similarity_model.encode(technical_description)

            # Calculate similarity with each thesis
            print("Calculating similarity with thesis abstracts...")
            similarities = []
            similar_titles_by_abstract = []
            all_similarities = []

            if cached_thesis_data is not None and cached_embeddings is not None:
                thesis_titles = cached_thesis_data['title'].tolist()
                thesis_abstracts = cached_thesis_data['abstract'].tolist()

                # Calculate similarities between technical description and all thesis embeddings
                raw_similarities = util.cos_sim(tech_desc_embedding.reshape(1, -1), cached_embeddings)[0]
                # Apply scaling factor to reduce similarity (multiply by 0.6)
                # This reduces the similarity to reflect that we're only comparing parts of the papers
                scaled_similarities = [float(score * 0.6) for score in raw_similarities]
                all_similarities = [float(score * 100) for score in scaled_similarities]

                print(f"Applied scaling factor (0.6) to all similarity scores")

                similar_titles_by_abstract = [
                    {
                        'title': thesis_titles[i],
                        'similarity': round(float(scaled_similarities[i] * 100), 2),
                        'abstract': thesis_abstracts[i][:200] + '...' if len(thesis_abstracts[i]) > 200 else thesis_abstracts[i]
                    }
                    for i, score in enumerate(raw_similarities) if score > 0.1  # still use raw similarity for threshold
                ]

                # Sort and keep top 5
                similar_titles_by_abstract = sorted(similar_titles_by_abstract, key=lambda x: x['similarity'], reverse=True)[:5]
                thesis_similarity = similar_titles_by_abstract[0]['similarity'] if similar_titles_by_abstract else 0
            else:
                similar_titles_by_abstract = []
                thesis_similarity = 0

            # Print similarity statistics
            if all_similarities:
                print(f"Abstract similarity statistics: min={min(all_similarities):.1f}%, max={max(all_similarities):.1f}%, avg={sum(all_similarities)/len(all_similarities):.1f}%")
                print(f"Found {len(similar_titles_by_abstract)} similar titles by abstract above threshold")
            else:
                print("No similarities calculated - all abstracts may have been skipped")

            # Get the highest similarity score
            print(f"Highest thesis similarity: {thesis_similarity:.2f}%")

            # Sort similar titles by similarity (highest first)
            similar_titles_by_abstract = sorted(similar_titles_by_abstract, key=lambda x: x['similarity'], reverse=True)[:5]
            print(f"Found {len(similar_titles_by_abstract)} similar titles")

        except Exception as e:
            print(f"Error calculating thesis similarity: {e}")
            traceback.print_exc()
            thesis_similarity = 0
            similar_titles_by_abstract = []

        # --- Discrepancies ---
        try:
            missed_keypoints, added_keypoints = summarize_discrepancies(extracted_text, speech_text)
        except Exception as e:
            print(f"Error extracting discrepancies: {e}")
            traceback.print_exc()
            missed_keypoints = fallback_results['missed_keypoints']
            added_keypoints = fallback_results['added_keypoints']

        # --- Keyword Extraction ---
        keyword_info = {
            "method": "enhanced",
            "keywords": []
        }
        try:
            # Use the enhanced keyword extraction for combined text
            combined_text = extracted_text + ' ' + speech_text
            keywords = extract_enhanced_keywords(
                combined_text,
                top_n=8,
                ignore_phrases=[
                    "Laguna State Polytechnic University",
                    "Province of Laguna",
                    "College of Computer Studies"
                ]
            )

            keyword_info["method"] = "enhanced_keybert"
            keyword_info["keywords"] = keywords

        except Exception as e:
            print(f"Keyword extraction error: {e}")
            traceback.print_exc()
            keywords = ["system", "data", "analysis", "research", "technology"]
            keyword_info["method"] = "fallback"
            keyword_info["keywords"] = keywords

        # --- Title Generation ---
        try:
            suggested_titles = generate_titles(extracted_text, keywords)
        except Exception as e:
            print(f"Title generation error: {e}")
            traceback.print_exc()
            suggested_titles = fallback_results['suggested_titles']

        if not suggested_titles:
            suggested_titles = fallback_results['suggested_titles']

        # --- Title Similarity Check ---
        similar_titles_by_title = []
        try:
            # Extract title from the proposal by looking for "Project Title:" or similar patterns
            import re

            # Try to find the project title using various patterns
            title_patterns = [
                r'(?i)project\s+title\s*[:]\s*([^\n]+)',
                r'(?i)research\s+title\s*[:]\s*([^\n]+)',
                r'(?i)title\s*[:]\s*([^\n]+)',
                r'(?i)^[\s\*]*([A-Z][^.!?]*(?:[.!?]|$))'  # Capitalized first line
            ]

            proposal_title = "Untitled Proposal"  # Default

            for pattern in title_patterns:
                matches = re.search(pattern, extracted_text)
                if matches:
                    candidate_title = matches.group(1).strip()
                    # Make sure it's a reasonable length for a title
                    if 10 <= len(candidate_title) <= 200:
                        proposal_title = candidate_title
                        print(f"Found title using pattern '{pattern}': {proposal_title}")
                        break

            print(f"Extracted proposal title: {proposal_title}")

            # Check for similar titles
            # Define the check_title_similarity function
            def check_title_similarity(title, proposal_text=None):
                try:
                    # Load thesis dataset
                    df = cached_thesis_data
                    if df is None:
                        print("Cached thesis data is None, trying to load directly")
                        df, _ = load_thesis_dataset_with_embeddings(custom_similarity_model)

                    if df is None or df.empty:
                        print("Main dataset is empty, trying preprocessed CSV")
                        # Try to load preprocessed CSV as fallback
                        preprocessed_path = os.path.join(BASE_PATH, 'preprocessed.csv')
                        if os.path.exists(preprocessed_path):
                            df = pd.read_csv(preprocessed_path).fillna('')
                            print(f"Loaded preprocessed dataset with {len(df)} rows")
                        else:
                            print(f"Preprocessed dataset not found at {preprocessed_path}")
                            # Create minimal fallback data
                            print("Creating minimal fallback dataset")
                            minimal_data = {
                                'title': [
                                    "Web-Based Student Information System",
                                    "Mobile Application for Healthcare Monitoring",
                                    "Automated Attendance System Using Facial Recognition"
                                ],
                                'abstract': [
                                    "This thesis presents the development of a web-based student information system designed to streamline administrative processes in educational institutions.",
                                    "This research focuses on the development of a mobile application for real-time healthcare monitoring using IoT sensors.",
                                    "This thesis presents an automated attendance system using facial recognition technology and deep learning algorithms."
                                ]
                            }
                            df = pd.DataFrame(minimal_data)
                            df['content'] = df['title'] + ' ' + df['abstract']
                            print(f"Created minimal fallback dataset with {len(df)} rows")

                    # Check if 'Title' column exists (case-insensitive check)
                    title_column = None
                    for col in df.columns:
                        if col.lower() == 'title':
                            title_column = col
                            break

                    if not title_column:
                        print("No 'title' column found in dataset. Available columns:", df.columns.tolist())
                        # Try to use a different column that might contain titles
                        potential_title_columns = ['Title', 'title', 'TITLE', 'project_title', 'research_title', 'name']
                        for potential_col in potential_title_columns:
                            if potential_col in df.columns:
                                title_column = potential_col
                                print(f"Using '{title_column}' as title column")
                                break

                        if not title_column:
                            return []

                    print(f"Using '{title_column}' column for title similarity")

                    # Encode the input title
                    title_embedding = custom_similarity_model.encode(title, convert_to_tensor=True)

                    # Extract technical description from proposal text if available
                    technical_description = None
                    tech_desc_embedding = None
                    if proposal_text:
                        # Extract technical description from concept paper
                        try:
                            # Look for sections specific to concept papers
                            concept_paper_patterns = [
                                r'(?i)technical\s+description.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)methodology.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)proposed\s+system.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)system\s+architecture.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)implementation\s+plan.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)technical\s+specifications.*?(?=\n\s*\n\s*[A-Z]|\Z)'
                            ]

                            for pattern in concept_paper_patterns:
                                matches = re.search(pattern, proposal_text, re.DOTALL)
                                if matches:
                                    technical_description = matches.group(0).strip()
                                    print(f"Found technical section using pattern: {pattern}")
                                    break

                            # If no specific section found, use the middle portion of the document
                            # (concept papers often have technical details in the middle)
                            if not technical_description:
                                print("No specific technical section found, using middle portion of document")
                                lines = proposal_text.split('\n')
                                if len(lines) > 10:
                                    start_idx = len(lines) // 3
                                    end_idx = (len(lines) * 2) // 3
                                    technical_description = '\n'.join(lines[start_idx:end_idx])
                                else:
                                    technical_description = proposal_text  # Use full text if it's short

                            print(f"Extracted technical description ({len(technical_description)} chars)")

                            # Encode the technical description if we have a context model
                            if context_sim_model and technical_description:
                                tech_desc_embedding = context_sim_model.encode(technical_description, convert_to_tensor=True)
                                print("Technical description encoded successfully")
                        except Exception as e:
                            print(f"Error extracting technical description: {e}")
                            technical_description = None
                        # Encode the technical description if available
                        if technical_description:
                            tech_desc_embedding = context_sim_model.encode(technical_description, convert_to_tensor=True)
                            print("Technical description encoded successfully")

                    # Calculate similarity with existing titles
                    similar_titles = []
                    for _, row in df.iterrows():
                        # Skip if the title column value is empty
                        if not row[title_column] or pd.isna(row[title_column]):
                            continue

                        # Title similarity
                        title_similarity = float(util.pytorch_cos_sim(
                            title_embedding.reshape(1, -1),
                            custom_similarity_model.encode(str(row[title_column]), convert_to_tensor=True).reshape(1, -1)
                        )[0][0] * 100)

                        # Context similarity if we have technical description and abstract
                        context_similarity = 0
                        if technical_description and 'abstract' in row and row['abstract'] and context_sim_model:
                            abstract_embedding = context_sim_model.encode(str(row['abstract']), convert_to_tensor=True)
                            # Calculate raw cosine similarity
                            raw_similarity = float(util.pytorch_cos_sim(
                                tech_desc_embedding.reshape(1, -1),
                                abstract_embedding.reshape(1, -1)
                            )[0][0] * 100)
                            # Apply scaling factor to reduce similarity (multiply by 0.6)
                            # This reduces the similarity to reflect that we're only comparing parts of the papers
                            context_similarity = raw_similarity * 0.6
                            print(f"Context similarity for '{row[title_column][:30]}...': {context_similarity:.1f}% (raw: {raw_similarity:.1f}%)")

                        # Combined similarity score (weighted average)
                        combined_similarity = title_similarity * 0.6 + context_similarity * 0.4 if context_similarity > 0 else title_similarity

                        # Add to results if similarity is above threshold (e.g., 70%)
                        if combined_similarity >= 50.0 or title_similarity >= 50.0 or context_similarity >= 50.0:
                            similar_titles.append({
                                'title': str(row[title_column]),
                                'title_similarity': round(title_similarity, 1),
                                'context_similarity': round(context_similarity, 1) if context_similarity > 0 else None,
                                'combined_similarity': round(combined_similarity, 1),
                                'abstract': str(row.get('abstract', ''))[:200] + '...' if 'abstract' in row else None
                            })

                    # Sort by combined similarity (highest first) and limit to top 5
                    similar_titles.sort(key=lambda x: x['combined_similarity'], reverse=True)
                    return similar_titles[:5]

                except Exception as e:
                    print(f"Error in check_title_similarity function: {e}")
                    traceback.print_exc()
                    return []

            similar_titles_by_title = check_title_similarity(proposal_title, extracted_text)
            print(f"Found {len(similar_titles_by_title)} similar titles")

        except Exception as e:
            print(f"Error checking title similarity: {e}")
            traceback.print_exc()
            similar_titles_by_title = []

        # --- Combine Results ---
        analysis_results = {
            'status': 'success',
            'speech_similarity': round(speech_similarity, 2) if speech_similarity is not None else None,
            'thesis_similarity': round(thesis_similarity, 2),
            'missed_keypoints': missed_keypoints,
            'added_keypoints': added_keypoints,
            'suggested_titles': suggested_titles,
            'keyword_analysis': keyword_info,
            'similar_titles_by_abstract': similar_titles_by_abstract,
            'similar_titles_by_title': similar_titles_by_title
        }

        # --- Save to DB ---
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE files
                SET analysis_json = %s
                WHERE id = %s AND user_email = %s
            """, (json.dumps(analysis_results), file_id, session['user_email']))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error saving results to database: {e}")
            traceback.print_exc()

        print("=== ANALYSIS COMPLETED SUCCESSFULLY ===")
        return jsonify(analysis_results)

    except Exception as e:
        print(f"Error in analyze_content: {e}")
        traceback.print_exc()
        return jsonify(fallback_results)


# This function has been moved to the top of the file

# Original check_thesis_similarity route has been replaced by the enhanced version below
# The implementation is now in the enhanced_context_similarity module

@app.route('/save_speech_transcript', methods=['POST'])
@login_required
def save_speech_transcript():
    try:
        data = request.json
        transcript_text = data.get('text', '')
        file_id = data.get('file_id')

        if not file_id:
            return jsonify({'status': 'error', 'message': 'File ID is required'})

        conn = get_db()
        cursor = conn.cursor()

        # Update the speech_transcript column for the specific file
        cursor.execute("""
            UPDATE files
            SET speech_transcript = %s
            WHERE id = %s AND user_email = %s
        """, (transcript_text, file_id, session['user_email']))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'status': 'success', 'message': 'Transcript saved successfully'})

    except Exception as e:
        print(f"Error saving transcript: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_speech_transcript/<int:file_id>')
@login_required
def get_speech_transcript(file_id):
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT speech_transcript
            FROM files
            WHERE id = %s AND user_email = %s
        """, (file_id, session['user_email']))

        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result and result['speech_transcript']:
            return jsonify({
                'status': 'success',
                'transcript': result['speech_transcript']
            })
        return jsonify({
            'status': 'success',
            'transcript': ''
        })

    except Exception as e:
        print(f"Error retrieving transcript: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Helper function to check string similarity
def is_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold


# Helper function to correct grammar
def correct_grammar(text):
    try:
        input_text = f"grammar: {text}"
        input_ids = grammar_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=128)
        outputs = grammar_model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
        return grammar_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Grammar correction failed: {e}")
        return text  # fallback to original if grammar model fails

# Optional filter: words expected in ComSci research titles
cs_keywords = {"system", "algorithm", "application", "framework", "detection", "machine learning",
               "artificial intelligence", "deep learning", "natural language", "NLP", "automation",
               "technology", "web", "mobile", "classification", "data", "neural", "network", "model"}

# ComSci Keywords for filtering
TECH_KEYWORDS = {
    "AI", "Machine Learning", "Cybersecurity", "Blockchain", "Cloud Computing", "IoT",
    "Software Engineering", "Big Data", "Computer Vision", "Natural Language Processing",
    "Data Science", "Algorithm Optimization", "Deep Learning", "IT Security", "Human-Computer Interaction"
}

# Note: is_similar and correct_grammar functions are already defined above

# Function to extract keywords from a given text
def extract_keywords(text: str) -> list:
    print(f"Extracting keywords from text of length {len(text)}")

    try:
        # Use the enhanced keyword extraction for better results
        return extract_enhanced_keywords(text, top_n=5)
    except Exception as e:
        print(f"Error in extract_keywords: {e}")
        # Return default keywords in case of any error
        default_keywords = ["system", "data", "analysis", "research", "technology"]
        print(f"Returning default keywords: {default_keywords}")
        return default_keywords

# Function to generate titles based on input text and keywords
def generate_titles(extracted_text: str, keywords: list) -> list:
    """
    Function to generate titles with strict filtering for quality.
    """
    # Extract abstract-like content - focus on technical content
    sentences = extracted_text.split(". ")

    # Get a cleaner abstract by focusing on sentences with technical terms
    tech_terms = set(TECH_KEYWORDS) | {"algorithm", "system", "framework", "model", "detection",
                                      "classification", "analysis", "processing", "neural", "data"}
    tech_terms_lower = {term.lower() for term in tech_terms}

    # Find sentences with technical content
    tech_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue

        # Check if sentence contains technical terms
        if any(term.lower() in sentence.lower() for term in tech_terms_lower):
            tech_sentences.append(sentence)

    # If we don't have enough technical sentences, use the first few sentences
    if len(tech_sentences) < 3:
        tech_sentences = [s for s in sentences[:5] if len(s.strip()) > 10]

    # Create a concise abstract
    abstract = ". ".join(tech_sentences[:3])

    # Ensure we have a reasonable abstract
    if len(abstract) < 50:
        abstract = ". ".join([s for s in sentences[:5] if len(s.strip()) > 10])

    # Limit abstract length
    if len(abstract) > 500:
        abstract = abstract[:500]

    # Extract the original title if present in the text
    original_title = ""
    title_patterns = [
        r'(?:title|project title|research title):\s*([^\n\.]+)',
        r'(?:^|\n)([^\n\.]{20,120})\s*\n'
    ]

    for pattern in title_patterns:
        matches = re.findall(pattern, extracted_text, re.IGNORECASE)
        if matches:
            potential_title = matches[0].strip()
            if len(potential_title) > 15:
                original_title = potential_title
                break

    # Create a prompt focused on generating CS/IT titles with specific methodologies
    prompt = (
        f"Generate 10 unique academic paper titles in Computer Science and Information Technology. "
        f"Paper content: {abstract}\n"
        f"Keywords: {', '.join(keywords)}\n\n"
        f"Rules:\n"
        f"1. Each title must be 40-120 characters long\n"
        f"2. Use proper title case format\n"
        f"3. ALWAYS include specific methodologies like NLP, Machine Learning, Neural Networks, CNN, RNN, LSTM, Transformers, etc.\n"
        f"4. Each title must clearly state BOTH the problem domain AND the technical method used\n"
        f"5. DO NOT include phrases like 'This paper is about' or 'The paper is about'\n"
        f"6. DO NOT repeat words within a title\n"
        f"7. Each title must be completely different from the others\n"
        f"8. Titles must be grammatically correct and professional\n"
        f"9. DO NOT include the word 'Keywords:' or any copyright information in the titles\n"
        f"10. DO NOT include phrases like 'This title highlights' or 'This title emphasizes'\n"
        f"11. Generate ONLY the title text with no additional explanations or metadata\n"
    )

    # Add instruction to avoid the original title if we found one
    if original_title:
        prompt += f"12. DO NOT generate this existing title: '{original_title}'\n"
        print(f"Original title detected: {original_title}")

    prompt += "\nGenerate only the titles with no additional text."

    print(f"Using prompt: {prompt}")

    # Generate titles directly using the model
    # Generate titles
    try:
        if title_generator:
            outputs = title_generator(
                prompt,
                max_length=120,
                min_length=30,
                num_return_sequences=3,  # Generate exactly 10 titles
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.2  # Discourage repetition
            )
            all_titles = [out['generated_text'].strip() for out in outputs]
        else:
            # Fallback if title_generator is not available
            input_ids = title_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = title_model.generate(
                input_ids,
                max_length=120,
                min_length=30,
                num_return_sequences=3,  # Generate exactly 10 titles
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.2
            )
            all_titles = [title_tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
    except Exception as e:
        print(f"Error generating titles: {e}")
        # Fallback titles if generation fails - with specific methodologies
        return [
            "Natural Language Processing and BERT Models for Fake Information Detection Systems",
            "Machine Learning and Random Forest Approaches to Computational Fact-Checking in Digital Media",
            "CNN-RNN Hybrid Neural Network Architecture for Content Authenticity Assessment",
            "Transformer-Based NLP Framework for Digital Content Verification and Analysis",
            "Deep Learning and LSTM Networks for Automated Misinformation Detection in Social Media"
        ]

    print(f"Raw generated titles: {all_titles}")

    # Simple filtering for high-quality titles
    filtered_titles = []

    # Basic problematic patterns to filter out
    bad_patterns = [
        r'this paper is about', r'the paper is about',
        r'title:', r'topic:', r'project:',
        r'^\d+[\.\)\:]', r'this title',
        r'keywords:', r'copyright', r'all rights reserved',
        r'this title (highlights|emphasizes|presents|showcases)',
        r'creative commons', r'wikimedia', r'wiley', r'periodicals',
        r'may not be published', r'broadcast', r'redistributed'
    ]

    for title in all_titles:
        # Skip empty or short titles
        if not title or len(title) < 20:
            continue

        # Basic cleanup
        title = title.strip()
        title = re.sub(r'\s+', ' ', title)  # Remove extra whitespace

        # Remove "Keywords:" and everything after it
        if re.search(r'keywords:', title, re.IGNORECASE):
            title = re.sub(r'keywords:.*$', '', title, flags=re.IGNORECASE).strip()

        # Remove copyright notices and similar text
        title = re.sub(r'copyright.*$', '', title, flags=re.IGNORECASE).strip()
        title = re.sub(r'all rights reserved.*$', '', title, flags=re.IGNORECASE).strip()

        # Skip if it contains problematic patterns
        if any(re.search(pattern, title, re.IGNORECASE) for pattern in bad_patterns):
            continue

        # Skip if it matches the original title
        if original_title and is_similar(title, original_title, 0.7):
            continue

        # Apply grammar correction if available
        try:
            if grammar_model and grammar_tokenizer:
                corrected_title = correct_grammar(title)
                if corrected_title and len(corrected_title) > 20:
                    title = corrected_title
        except Exception as e:
            print(f"Grammar correction failed: {e}")

        # Add to filtered titles if not a duplicate
        if title and title not in filtered_titles:
            filtered_titles.append(title)

    # If we have more than 5 titles, keep only the first 5
    filtered_titles = filtered_titles[:5]

    # If we have fewer than 5 titles, add fallback titles with specific methodologies
    if len(filtered_titles) < 5:
        # Fallback titles with specific methodologies
        fallbacks = [
            f"Natural Language Processing Techniques for {keywords[0] if keywords else 'Fake Information'} Detection and Classification",
            f"Deep Learning and BERT Models for Automated {keywords[0] if keywords else 'Content'} Verification",
            f"Transformer-Based Approach to {keywords[0] if keywords else 'Misinformation'} Analysis in Social Media",
            f"CNN-LSTM Hybrid Architecture for Real-Time {keywords[0] if keywords else 'Information'} Authenticity Assessment",
            f"Machine Learning and NLP Framework for {keywords[0] if keywords else 'Fake News'} Detection in Online Media"
        ]

        # Add fallbacks until we have 5 titles
        for fallback in fallbacks:
            if len(filtered_titles) >= 5:
                break
            if fallback not in filtered_titles:
                filtered_titles.append(fallback)

    print(f"Final titles: {filtered_titles}")
    return filtered_titles

# Route to handle POST requests for generating titles
@app.route('/generate_title', methods=['POST'])
def generate_title_route():
    print("=== GENERATING TITLES ===")

    data = request.get_json()
    extracted_text = data.get('extracted_text', '')
    speech_text = data.get('speech_text', '')  # Add this

    # Combine both sources
    combined_text = f"{extracted_text} {speech_text}".strip()

    if not combined_text:
        return jsonify({'status': 'error', 'message': 'No text provided'}), 400

    try:
        # Extract keywords from combined input
        raw_keywords = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=10
        )
        keywords = [kw for kw, _ in raw_keywords]
        print(f"Keywords extracted: {keywords}")

        # Use a shorter and clearer abstract
        sentences = combined_text.split(". ")
        abstract = ". ".join(sentences[:3])

        # Better prompt for title generation
        prompt = (
            f"Suggest 3 concise, academic research titles suitable for a thesis in Computer Science or Information Technology. "
            f"Base them on the following content: {abstract}. Ensure each title uses or aligns with keywords: {', '.join(keywords)}"
        )

        # Generate titles
        outputs = title_generator(
            prompt,
            max_length=64,
            num_return_sequences=3,
            do_sample=True,
            temperature=0.8
        )

        # Clean up model outputs
        titles = list({out['generated_text'].strip().strip('"') for out in outputs})

        print(f"Generated titles: {titles}")

        return jsonify({
            'status': 'success',
            'titles': titles[:3],  # Return max 3 unique titles
            'keywords': keywords
        })

    except Exception as e:
        print(f"Error during title generation: {e}")
        return jsonify({
            'status': 'error',
            'message': 'An error occurred during title generation.'
        }), 500

# ARCHIVE
@app.route('/archive')
@login_required
def archive():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)  # Make sure dictionary=True is set

    cursor.execute("""
        SELECT id, file_name, file_path, upload_date
        FROM files
        WHERE user_email = %s AND archived = TRUE
    """, (session['user_email'],))

    archived_files = cursor.fetchall()
    cursor.close()
    conn.close()

    # Debug print to check what data is being passed
    print("Archived files:", archived_files)

    return render_template("archive.html", files=archived_files)

# ARCHIVE (management)
@app.route('/move_to_archive/<int:file_id>', methods=['POST'])
@login_required
def move_to_archive(file_id):
    conn = get_db()
    cursor = conn.cursor()

    try:
        # Set archived to TRUE for the specified file
        cursor.execute("""
            UPDATE files
            SET archived = TRUE
            WHERE id = %s AND user_email = %s
        """, (file_id, session['user_email']))

        conn.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)})
    finally:
        cursor.close()
        conn.close()

@app.route('/restore_from_archive/<int:file_id>', methods=['POST'])
@login_required
def restore_from_archive(file_id):
    conn = get_db()
    cursor = conn.cursor()

    try:
        # Update the file to remove archived status
        cursor.execute("""
            UPDATE files
            SET archived = FALSE
            WHERE id = %s AND user_email = %s
        """, (file_id, session['user_email']))

        conn.commit()

        # Return success response
        return jsonify({
            "status": "success",
            "message": "File restored successfully"
        })
    except Exception as e:
        conn.rollback()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/delete_file/<int:file_id>', methods=['DELETE'])
@login_required
def delete_file(file_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # First get the file path
    cursor.execute("SELECT file_path FROM files WHERE id = %s AND user_email = %s",
                  (file_id, session['user_email']))
    file_record = cursor.fetchone()

    if file_record:
        # Delete the physical file
        try:
            os.remove(file_record['file_path'])
        except OSError as e:
            print(f"Error deleting file: {e}")

        # Delete from database
        cursor.execute("DELETE FROM files WHERE id = %s AND user_email = %s",
                      (file_id, session['user_email']))
        conn.commit()

    cursor.close()
    conn.close()

    return jsonify({"status": "success"})

@app.route('/delete_archived_file/<int:file_id>', methods=['DELETE'])
@login_required
def delete_archived_file(file_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    try:
        # First get the file path
        cursor.execute("""
            SELECT file_path
            FROM files
            WHERE id = %s AND user_email = %s AND archived = TRUE
        """, (file_id, session['user_email']))

        file_record = cursor.fetchone()

        if file_record:
            # Delete the physical file
            try:
                if os.path.exists(file_record['file_path']):
                    os.remove(file_record['file_path'])
            except OSError as e:
                print(f"Error deleting physical file: {e}")

            # Delete from database
            cursor.execute("""
                DELETE FROM files
                WHERE id = %s AND user_email = %s AND archived = TRUE
            """, (file_id, session['user_email']))

            conn.commit()
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error", "message": "File not found"}), 404

    except Exception as e:
        conn.rollback()
        print(f"Error deleting archived file: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

# ACCOUNT
@app.route('/account')
@login_required
def account():
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        # Fetch user details from database
        cursor.execute("""
            SELECT email, password
            FROM registration
            WHERE email = %s
        """, (session['user_email'],))

        user_data = cursor.fetchone()
        cursor.close()
        conn.close()

        if user_data:
            return render_template("account.html",
                                 email=user_data['email'],
                                 password=user_data['password'])

        return redirect(url_for('login'))

    except Exception as e:
        print(f"Error fetching account details: {str(e)}")
        return redirect(url_for('login'))

@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    conn = None
    cursor = None
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        user_email = session['user_email']

        print(f"Attempting to delete account for user: {user_email}")  # Debug log

        # Start transaction
        conn.start_transaction()

        # First get all files associated with the user
        cursor.execute("SELECT file_path FROM files WHERE user_email = %s", (user_email,))
        files = cursor.fetchall()
        print(f"Found {len(files)} files to delete")  # Debug log

        # Delete physical files from the server
        for file in files:
            try:
                file_path = file['file_path']
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")  # Debug log
            except OSError as e:
                print(f"Warning: Error deleting file {file_path}: {e}")
                continue

        # Delete records from existing tables only
        try:
            # Delete files records
            cursor.execute("DELETE FROM files WHERE user_email = %s", (user_email,))
            print(f"Deleted {cursor.rowcount} records from files table")  # Debug log

            # Delete user record
            cursor.execute("DELETE FROM registration WHERE email = %s", (user_email,))
            print(f"Deleted {cursor.rowcount} records from registration table")  # Debug log

            # Commit the transaction
            conn.commit()
            print("Transaction committed successfully")  # Debug log

            # Clear session
            session.clear()

            return jsonify({
                "status": "success",
                "message": "Account deleted successfully"
            })

        except mysql.connector.Error as e:
            print(f"Database error during deletion: {e}")  # Debug log
            if conn:
                conn.rollback()
            raise e

    except mysql.connector.Error as e:
        print(f"MySQL Error: {e}")  # Debug log
        if conn:
            conn.rollback()
        return jsonify({
            "status": "error",
            "message": f"Database error: {str(e)}"
        }), 500

    except Exception as e:
        print(f"Unexpected error: {e}")  # Debug log
        if conn:
            conn.rollback()
        return jsonify({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("Database connection closed")  # Debug log

@app.route('/check_thesis_similarity', methods=['POST'])
@login_required
def check_thesis_similarity_route():
    try:
        # Use the enhanced context similarity handler
        data = request.json
        result = check_thesis_similarity_route_handler(
            data=data,
            custom_similarity_model=custom_similarity_model,
            context_sim_model=context_sim_model,
            sentence_model=sentence_model,
            extract_technical_description=extract_technical_description,
            load_thesis_dataset_with_embeddings=load_thesis_dataset_with_embeddings,
            BASE_PATH=BASE_PATH
        )

        # If the result is a tuple, it contains an error response
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]

        # Otherwise, return the success response
        return jsonify(result)

    except Exception as e:
        print(f"Error in check_thesis_similarity_route: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    app.run(debug=True, port=3000)

@app.route('/check_title_similarity', methods=['POST'])
@login_required
def check_title_similarity_route():
    try:
        data = request.json
        title = data.get('title', '').strip()

        if not title:
            return jsonify({
                'status': 'error',
                'message': 'Title is required'
            }), 400

        # Define the check_title_similarity function
        def check_title_similarity(title, proposal_text=None):
            try:
                # Load thesis dataset
                df, _ = load_thesis_dataset_with_embeddings(custom_similarity_model)
                if df.empty:
                    print("Main dataset is empty, trying preprocessed CSV")
                    # Try to load preprocessed CSV as fallback
                    preprocessed_path = os.path.join(BASE_PATH, 'preprocessed.csv')
                    if os.path.exists(preprocessed_path):
                        df = pd.read_csv(preprocessed_path).fillna('')
                        print(f"Loaded preprocessed dataset with {len(df)} rows")
                    else:
                        print(f"Preprocessed dataset not found at {preprocessed_path}")
                        return []

                # Check if 'Title' column exists (case-insensitive check)
                title_column = None
                for col in df.columns:
                    if col.lower() == 'title':
                        title_column = col
                        break

                if not title_column:
                    print("No 'title' column found in dataset. Available columns:", df.columns.tolist())
                    # Try to use a different column that might contain titles
                    potential_title_columns = ['Title', 'title', 'TITLE', 'project_title', 'research_title', 'name']
                    for potential_col in potential_title_columns:
                        if potential_col in df.columns:
                            title_column = potential_col
                            print(f"Using '{title_column}' as title column")
                            break

                        if not title_column:
                            return []

                    print(f"Using '{title_column}' column for title similarity")

                    # Encode the input title
                    title_embedding = custom_similarity_model.encode(title, convert_to_tensor=True)

                    # Extract technical description from proposal text if available
                    technical_description = None
                    tech_desc_embedding = None
                    if proposal_text:
                        # Extract technical description from concept paper
                        try:
                            # Look for sections specific to concept papers
                            concept_paper_patterns = [
                                r'(?i)technical\s+description.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)methodology.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)proposed\s+system.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)system\s+architecture.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)implementation\s+plan.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)technical\s+specifications.*?(?=\n\s*\n\s*[A-Z]|\Z)'
                            ]

                            for pattern in concept_paper_patterns:
                                matches = re.search(pattern, proposal_text, re.DOTALL)
                                if matches:
                                    technical_description = matches.group(0).strip()
                                    print(f"Found technical section using pattern: {pattern}")
                                    break

                            # If no specific section found, use the middle portion of the document
                            # (concept papers often have technical details in the middle)
                            if not technical_description:
                                print("No specific technical section found, using middle portion of document")
                                lines = proposal_text.split('\n')
                                if len(lines) > 10:
                                    start_idx = len(lines) // 3
                                    end_idx = (len(lines) * 2) // 3
                                    technical_description = '\n'.join(lines[start_idx:end_idx])
                                else:
                                    technical_description = proposal_text  # Use full text if it's short

                            print(f"Extracted technical description ({len(technical_description)} chars)")

                            # Encode the technical description if we have a context model
                            if context_sim_model and technical_description:
                                tech_desc_embedding = context_sim_model.encode(technical_description, convert_to_tensor=True)
                                print("Technical description encoded successfully")
                        except Exception as e:
                            print(f"Error extracting technical description: {e}")
                            technical_description = None
                        # Encode the technical description if available
                        if technical_description:
                            tech_desc_embedding = context_sim_model.encode(technical_description, convert_to_tensor=True)
                            print("Technical description encoded successfully")

                    # Calculate similarity with existing titles
                    similar_titles = []
                    for _, row in df.iterrows():
                        # Skip if the title column value is empty
                        title_column = col
                        break

                if not title_column:
                    print("No 'title' column found in dataset. Available columns:", df.columns.tolist())
                    # Try to use a different column that might contain titles
                    potential_title_columns = ['Title', 'title', 'TITLE', 'project_title', 'research_title', 'name']
                    for potential_col in potential_title_columns:
                        if potential_col in df.columns:
                            title_column = potential_col
                            print(f"Using '{title_column}' as title column")
                            break

                        if not title_column:
                            return []

                    print(f"Using '{title_column}' column for title similarity")

                    # Encode the input title
                    title_embedding = custom_similarity_model.encode(title, convert_to_tensor=True)

                    # Extract technical description from proposal text if available
                    technical_description = None
                    tech_desc_embedding = None
                    if proposal_text:
                        # Extract technical description from concept paper
                        try:
                            # Look for sections specific to concept papers
                            concept_paper_patterns = [
                                r'(?i)technical\s+description.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)methodology.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)proposed\s+system.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)system\s+architecture.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)implementation\s+plan.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                                r'(?i)technical\s+specifications.*?(?=\n\s*\n\s*[A-Z]|\Z)'
                            ]

                            for pattern in concept_paper_patterns:
                                matches = re.search(pattern, proposal_text, re.DOTALL)
                                if matches:
                                    technical_description = matches.group(0).strip()
                                    print(f"Found technical section using pattern: {pattern}")
                                    break

                            # If no specific section found, use the middle portion of the document
                            # (concept papers often have technical details in the middle)
                            if not technical_description:
                                print("No specific technical section found, using middle portion of document")
                                lines = proposal_text.split('\n')
                                if len(lines) > 10:
                                    start_idx = len(lines) // 3
                                    end_idx = (len(lines) * 2) // 3
                                    technical_description = '\n'.join(lines[start_idx:end_idx])
                                else:
                                    technical_description = proposal_text  # Use full text if it's short

                            print(f"Extracted technical description ({len(technical_description)} chars)")

                            # Encode the technical description if we have a context model
                            if context_sim_model and technical_description:
                                tech_desc_embedding = context_sim_model.encode(technical_description, convert_to_tensor=True)
                                print("Technical description encoded successfully")
                        except Exception as e:
                            print(f"Error extracting technical description: {e}")
                            technical_description = None
                        # Encode the technical description if available
                        if technical_description:
                            tech_desc_embedding = context_sim_model.encode(technical_description, convert_to_tensor=True)
                            print("Technical description encoded successfully")

                    # Calculate similarity with existing titles
                    similar_titles = []
                    for _, row in df.iterrows():
                        # Skip if the title column value is empty
                        title_column = col
                        break

                if not title_column:
                    print("No 'title' column found in dataset. Available columns:", df.columns.tolist())
                    # Try to use a different column that might contain titles
                    potential_title_columns = ['Title', 'title', 'TITLE', 'project_title', 'research_title', 'name']
                    for potential_col in potential_title_columns:
                        if potential_col in df.columns:
                            title_column = potential_col
                            print(f"Using '{title_column}' as title column")
                            break

                        if not title_column:
                            return []

                print(f"Using '{title_column}' column for title similarity")

                # Encode the input title
                title_embedding = custom_similarity_model.encode(title, convert_to_tensor=True)

                # Extract technical description from proposal text if available
                technical_description = None
                tech_desc_embedding = None
                if proposal_text:
                    # Extract technical description from concept paper
                    try:
                        # Look for sections specific to concept papers
                        concept_paper_patterns = [
                            r'(?i)technical\s+description.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                            r'(?i)methodology.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                            r'(?i)proposed\s+system.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                            r'(?i)system\s+architecture.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                            r'(?i)implementation\s+plan.*?(?=\n\s*\n\s*[A-Z]|\Z)',
                            r'(?i)technical\s+specifications.*?(?=\n\s*\n\s*[A-Z]|\Z)'
                        ]

                        for pattern in concept_paper_patterns:
                            matches = re.search(pattern, proposal_text, re.DOTALL)
                            if matches:
                                technical_description = matches.group(0).strip()
                                print(f"Found technical section using pattern: {pattern}")
                                break

                        # If no specific section found, use the middle portion of the document
                        # (concept papers often have technical details in the middle)
                        if not technical_description:
                            print("No specific technical section found, using middle portion of document")
                            lines = proposal_text.split('\n')
                            if len(lines) > 10:
                                start_idx = len(lines) // 3
                                end_idx = (len(lines) * 2) // 3
                                technical_description = '\n'.join(lines[start_idx:end_idx])
                            else:
                                technical_description = proposal_text  # Use full text if it's short

                        print(f"Extracted technical description ({len(technical_description)} chars)")

                        # Encode the technical description if we have a context model
                        if context_sim_model and technical_description:
                            tech_desc_embedding = context_sim_model.encode(technical_description, convert_to_tensor=True)
                            print("Technical description encoded successfully")
                    except Exception as e:
                        print(f"Error extracting technical description: {e}")
                        technical_description = None
                    # Encode the technical description if available
                    if technical_description:
                        tech_desc_embedding = context_sim_model.encode(technical_description, convert_to_tensor=True)
                        print("Technical description encoded successfully")

                # Calculate similarity with existing titles
                similar_titles = []
                for _, row in df.iterrows():
                    # Skip if the title column value is empty
                    if not row[title_column] or pd.isna(row[title_column]):
                        continue

                    # Title similarity
                    title_similarity = float(util.pytorch_cos_sim(
                        title_embedding.reshape(1, -1),
                        custom_similarity_model.encode(str(row[title_column]), convert_to_tensor=True).reshape(1, -1)
                    )[0][0] * 100)

                    # Context similarity if we have technical description and abstract
                    context_similarity = 0
                    if technical_description and 'abstract' in row and row['abstract'] and context_sim_model:
                        abstract = str(row['abstract'])
                        # Skip very short abstracts
                        if len(abstract.strip()) >= 10:
                            try:
                                abstract_embedding = context_sim_model.encode(abstract, convert_to_tensor=True)
                                context_similarity = float(util.pytorch_cos_sim(
                                    tech_desc_embedding.reshape(1, -1),
                                    abstract_embedding.reshape(1, -1)
                                )[0][0] * 100)
                                print(f"Context similarity for '{row[title_column][:30]}...': {context_similarity:.1f}%")
                            except Exception as e:
                                print(f"Error encoding abstract: {e}")
                                context_similarity = 0

                    # Combined similarity score (weighted average)
                    # Give more weight to context similarity if it's available
                    combined_similarity = title_similarity * 0.4 + context_similarity * 0.6 if context_similarity > 0 else title_similarity

                    # Add to results if similarity is above threshold - LOWERED FOR TESTING
                    if combined_similarity >= 20.0 or title_similarity >= 30.0 or context_similarity >= 20.0:
                        similar_titles.append({
                            'title': str(row[title_column]),
                            'title_similarity': round(title_similarity, 1),
                            'context_similarity': round(context_similarity, 1) if context_similarity > 0 else None,
                            'combined_similarity': round(combined_similarity, 1),
                            'abstract': str(row.get('abstract', ''))[:200] + '...' if 'abstract' in row else None
                        })

                # Sort by combined similarity (highest first) and limit to top 5
                similar_titles.sort(key=lambda x: x['combined_similarity'], reverse=True)
                return similar_titles[:5]

            except Exception as e:
                print(f"Error in check_title_similarity function: {e}")
                traceback.print_exc()
                return []

        # Call the function to get similar titles
        similar_titles = check_title_similarity(title)

        # Return the results
        return jsonify({
            'status': 'success',
            'similar_titles': similar_titles
        })

    except Exception as e:
        print(f"Error in /check_title_similarity route: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500
