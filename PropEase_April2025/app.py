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
from t5_model import generate_title
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
from utils import translate_to_english
import json
from datetime import datetime
from typing import List
import spacy
from nltk.tokenize import sent_tokenize
import nltk
import traceback
import base64

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

title_tokenizer = T5Tokenizer.from_pretrained("t5-base")  # or "t5-small"
title_model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Configure SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Initialize models with error handling
try:
    # Initialize sentence transformer model
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
except Exception as e:
    print(f"Error loading sentence transformer model: {e}")
    # Fallback to a simpler model
    try:
        sentence_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
    except Exception as e:
        print(f"Error loading fallback sentence transformer model: {e}")
        sentence_model = None

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
except Exception as e:
    print(f"Error loading grammar correction model: {e}")
    try:
        # Fallback to t5-small
        grammar_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
        grammar_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    except Exception as e:
        print(f"Error loading fallback grammar model: {e}")
        grammar_tokenizer = None
        grammar_model = None

# Add this with other model initializations
try:
    title_tokenizer = T5Tokenizer.from_pretrained("t5-base")
except Exception as e:
    print(f"Error loading title tokenizer: {e}")
    title_tokenizer = None

kw_model = KeyBERT() 
USE_EMOJIS_IN_LOGS = False  # Set to True if you want emojis in your debug logs

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

# BASE PATH for MODELS
BASE_PATH = os.path.dirname(__file__)

ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database connection using Flask `g`
def get_db():
    if 'db' not in g:
        g.db = get_db_connection()
    return g.db

@app.teardown_appcontext
def close_db(error=None):
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
        relative_path = file_record['file_path']
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

    
def extract_keypoints(text, top_n=10):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n
    )
    return [kw.lower() for kw, _ in keywords]

def summarize_discrepancies(paper_text, speech_text):
    paper_keywords = extract_keypoints(paper_text)
    speech_keywords = extract_keypoints(speech_text)

    missed_in_speech = [kw for kw in paper_keywords if kw not in speech_keywords]
    added_in_speech = [kw for kw in speech_keywords if kw not in paper_keywords]

    return missed_in_speech[:5], added_in_speech[:5]

@app.route('/analyze_content/<int:file_id>', methods=['POST'])
@login_required
def analyze_content(file_id):
    try:
        data = request.json
        extracted_text = data.get('extracted_text', '')
        speech_text = data.get('speech_text', '')

        if not extracted_text or not speech_text:
            return jsonify({
                'status': 'error',
                'message': 'Missing text content'
            }), 400

        # Calculate speech similarity using the existing model
        proposal_embedding = model.encode(extracted_text)
        speech_embedding = model.encode(speech_text)
        speech_similarity = float(util.pytorch_cos_sim(
            proposal_embedding.reshape(1, -1),
            speech_embedding.reshape(1, -1)
        )[0][0] * 100)

        # Get key points and discrepancies
        missed_keypoints, added_keypoints = summarize_discrepancies(extracted_text, speech_text)

        # Generate title recommendations
        keywords = kw_model.extract_keywords(extracted_text, top_n=3)
        suggested_titles = [
            f"Analysis of {kw[0].title()} in Research Context" for kw in keywords
        ]

        analysis_results = {
            'status': 'success',
            'speech_similarity': round(speech_similarity, 2),
            'missed_keypoints': missed_keypoints,
            'added_keypoints': added_keypoints,
            'suggested_titles': suggested_titles
        }

        # Save results to database
        conn = get_db_connection()
        cursor = conn.cursor()
            
        try:
            cursor.execute("""
                 UPDATE files 
                SET analysis_json = %s 
                WHERE id = %s AND user_email = %s
            """, (json.dumps(analysis_results), file_id, session['user_email']))
                
            conn.commit()
        finally:
            cursor.close()
            conn.close()

        return jsonify(analysis_results)

    except Exception as e:
        print(f"Error in analyze_content: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Load the pre-trained model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def load_thesis_dataset():
    try:
        # Get the absolute path to the dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'PropEase_ Dataset.xlsx')
        
        print(f"Attempting to load dataset from: {file_path}")  # Debug log
        
        if not os.path.exists(file_path):
            print(f"Dataset not found at: {file_path}")
            # Try alternative path in case file is in parent directory
            parent_dir = os.path.dirname(current_dir)
            file_path = os.path.join(parent_dir, 'PropEase_ Dataset.xlsx')
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found in either {current_dir} or {parent_dir}")
            
        print(f"Loading dataset from: {file_path}")  # Debug log
        df = pd.read_excel(file_path)
        df = df.fillna('')  # Replace NaN values with empty strings

        # Verify required columns exist
        required_columns = ['Title', 'Author', 'Date', 'Program', 'Introduction', 
                          'Literature Review', 'Method', 'Result', 'Discussion', 'Conclusion']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in dataset: {missing_columns}")
            return pd.DataFrame()

        print(f"Successfully loaded dataset with {len(df)} rows")  # Debug log

        # Combine useful sections into a new 'content' column
        df['content'] = (
            df['Introduction'].astype(str) + " " +
            df['Literature Review'].astype(str) + " " +
            df['Method'].astype(str) + " " +
            df['Result'].astype(str) + " " +
            df['Discussion'].astype(str) + " " +
            df['Conclusion'].astype(str)
        )

        print("Computing embeddings for thesis database...")
        # Add progress indicator for long datasets
        total_rows = len(df)
        for idx, row in df.iterrows():
            if idx % 10 == 0:  # Print progress every 10 rows
                print(f"Processing embeddings: {idx}/{total_rows} rows")
            df.at[idx, 'title_embedding'] = model.encode(str(row['Title']))
            df.at[idx, 'content_embedding'] = model.encode(str(row['content']))
        print("Embeddings computation completed")

        return df
        
    except FileNotFoundError as e:
        print(f"❌ Error: Dataset file not found: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading thesis dataset: {str(e)}")
        traceback.print_exc()  # Print full error traceback
        return pd.DataFrame()

def check_thesis_similarity(new_title, new_content, threshold=0.6):
    try:
        df = load_thesis_dataset()
        if df.empty:
            return []

        # Encode new title and content
        new_title_embedding = model.encode(new_title)
        new_content_embedding = model.encode(new_content)

        similar_theses = []
        
        for idx, row in df.iterrows():
            # Calculate similarities
            title_similarity = cosine_similarity(
                [new_title_embedding], 
                [row['title_embedding']]
            )[0][0]
            
            content_similarity = cosine_similarity(
                [new_content_embedding], 
                [row['content_embedding']]
            )[0][0]

            # Calculate combined similarity score
            combined_similarity = (title_similarity * 0.4 + content_similarity * 0.6)

            if combined_similarity > threshold:
                thesis_info = {
                    'existing_title': row['Title'],
                    'author': row['Author'],
                    'date': row['Date'],
                    'program': row['Program'],
                    'title_similarity': round(title_similarity * 100, 2),
                    'content_similarity': round(content_similarity * 100, 2),
                    'combined_similarity': round(combined_similarity * 100, 2)
                }

                similar_theses.append(thesis_info)

        # Sort by combined similarity score
        similar_theses.sort(key=lambda x: x['combined_similarity'], reverse=True)
        return similar_theses[:5]  # Return top 5 similar theses

    except Exception as e:
        print(f"Error checking thesis similarity: {e}")
        return []

@app.route('/check_thesis_similarity', methods=['POST'])
def check_similarity():
    try:
        data = request.json
        new_title = data.get('title', '')
        new_content = data.get('content', '')

        if not new_title or not new_content:
            return jsonify({
                'status': 'error',
                'message': 'Both title and content are required'
            }), 400

        similar_theses = check_thesis_similarity(new_title, new_content)

        return jsonify({
            'status': 'success',
            'similar_theses': similar_theses
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

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
    
def is_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

def correct_grammar(text: str) -> str:
    if grammar_model is None or grammar_tokenizer is None:
        return text  # Return original text if models aren't available
        
    try:
        input_text = f"grammar: {text}"
        input_ids = grammar_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = grammar_model.generate(
                input_ids, 
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
        
        corrected_text = grammar_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return corrected_text if corrected_text else text
    except Exception as e:
        print(f"Grammar correction error: {e}")
        return text  # Return original text if correction fails

# Optional filter: words expected in ComSci research titles
cs_keywords = {
    "system", "algorithm", "application", "framework", "detection", 
    "machine learning", "artificial intelligence", "deep learning", 
    "natural language", "NLP", "automation", "technology", "web", 
    "mobile", "classification", "data", "neural", "network", "model"
}

def generate_titles(text: str, keywords: List[str], num_titles: int = 3) -> List[str]:
    """
    Generate titles based on text content and keywords using the T5 model
    """
    try:
        # Prepare input text
        keyword_str = ', '.join(keywords)
        prompt = (
            f"Generate {num_titles} academic research project titles related to computer science. "
            f"Context: {text.strip()}. Use keywords: {keyword_str}."
        )

        # Use the existing title generation model and tokenizer
        inputs = title_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        outputs = title_model.generate(
            inputs,
            max_length=32,
            min_length=8,
            num_beams=4,
            num_return_sequences=num_titles,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            early_stopping=True
        )
        
        # Decode titles
        titles = [title_tokenizer.decode(output, skip_special_tokens=True).strip() 
                 for output in outputs]
        
        # Filter and clean titles
        filtered_titles = []
        for title in titles:
            if len(title.split()) >= 4:  # Ensure minimum length
                if all(not is_similar(title, existing) for existing in filtered_titles):
                    filtered_titles.append(title)
        
        # Apply CS filter
        final_titles = [
            title for title in filtered_titles 
            if any(cs_word.lower() in title.lower() for cs_word in cs_keywords)
        ]
        
        # Return requested number of titles
        return final_titles[:num_titles]
        
    except Exception as e:
        print(f"Error generating titles: {str(e)}")
        return ["Error generating titles"] * num_titles

@app.route('/generate_title', methods=['POST'])
def generate_title_route():
    try:
        data = request.get_json()
        extracted_text = data.get('extracted_text', '')
        speech_text = data.get('speech_text', '')
        
        # Extract keywords from the text
        # This is a simple example - you might want to use a more sophisticated keyword extraction method
        keywords = extract_keywords(extracted_text + ' ' + speech_text)
        
        # Generate titles using your existing function
        titles = generate_titles(extracted_text, keywords)
        
        return jsonify({
            'status': 'success',
            'titles': titles
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def extract_keywords(text):
    # Simple keyword extraction - replace with your preferred method
    common_cs_terms = set(['system', 'algorithm', 'data', 'network', 'software', 'application'])
    words = set(text.lower().split())
    keywords = list(words.intersection(common_cs_terms))
    return keywords[:5]  # Return up to 5 keywords

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

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    app.run(debug=True, port=3000)