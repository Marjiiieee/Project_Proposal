from flask import Flask, request, jsonify, send_file, session, g, render_template, redirect, url_for, flash
import os
import pdfplumber
import mysql.connector
import uuid  # For generating unique filenames
from werkzeug.utils import secure_filename
from db import get_db_connection  # Import the database connection function
from flask import send_from_directory
from functools import wraps
import PyPDF2
import pickle
from flask_session import Session
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import camelot
import re


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")  # Use environment variable for security
app.config['SESSION_TYPE'] = 'filesystem'  # Store session data on the server
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file upload limit
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
app.config['SESSION_PERMANENT'] = True
extracted_texts = {}  # Cache for extracted text

vectorizer = TfidfVectorizer()

# Save vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'docx', 'txt'}

# Ensure the upload folder exists
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

# Authentication check decorator
# def login_required(func):
#     def wrapper(*args, **kwargs):
#         if 'user_id' not in session:
#             return redirect(url_for('login'))
#         return func(*args, **kwargs)
#     wrapper.__name__ = func.__name__
#     return wrapper

# Login required decorator
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

# Home Route - Main Dashboard
@app.route('/home')
@login_required
def home():
    return render_template("home.html")

# 🔹 LOGIN ROUTE
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
                    "message": "Email not found. Please check your email or sign up."
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

# 🔹 LOGOUT ROUTE
@app.route('/logout')
def logout():
    session.clear()  # Clear session data
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))


# 🔹 REGISTRATION ROUTE (No Hashing)
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template("signin.html")

    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required"}), 400

    # Validate email domain
    valid_domains = ['gmail.com', 'yahoo.com', 'edu.ph', 'outlook.com', 'hotmail.com']
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
                            text = page.extract_text()
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

                # If still no text, try to extract tables using camelot
                if not extracted_text.strip():
                    try:
                        tables = camelot.read_pdf(file_path, pages='all')
                        for table in tables:
                            extracted_text += table.df.to_string() + "\n"
                    except Exception as e:
                        print(f"Camelot extraction error: {e}")

                # Verify if we got any text
                if not extracted_text.strip():
                    print("Warning: No text could be extracted from the PDF")
                    extracted_text = "No text could be extracted from this PDF."

                # Debug print
                print(f"Extracted text length: {len(extracted_text)}")
                print(f"First 200 characters: {extracted_text[:200]}")

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

# Archive Page
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

# Add these new routes for archive management
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

# Account/Profile Page
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

# Load the SVM model
# Define the correct path
svm_model_path = r"C:\xampp\htdocs\PropEase-main\Project-main\svm_model.pkl"

if not os.path.exists(svm_model_path):
    raise FileNotFoundError(f"Model file not found: {svm_model_path}")

with open(svm_model_path, "rb") as model_file:
    svm_model = pickle.load(model_file)

# Load the TF-IDF Vectorizer
vectorizer_path = r"C:\xampp\htdocs\PropEase-main\Project-main\tfidf_vectorizer.pkl"

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

# Load the vectorizer
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Check if it has a vocabulary
if hasattr(vectorizer, "vocabulary_") and vectorizer.vocabulary_:
    print("Vectorizer is fitted. Ready to use.")
else:
    print("Vectorizer is NOT fitted. You need to train it.")


import time

# Extract Text from PDF
def extract_text_from_pdf(file_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT file_path FROM files WHERE id = %s", (file_id,))
    file_record = cursor.fetchone()

    cursor.close()
    conn.close()

    if not file_record:
        return "File not found in the database."

    file_path = file_record["file_path"]

    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        print(f"Extracted text (first 200 chars): {text[:200]}")
    except Exception as e:
        print(f"PDF Extraction error: {e}")
        return "Error extracting text."
    
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    except Exception as e:
        print(f"Error extracting text: {str(e)}")

    return text if text else "No text extracted."

def extract_tables_from_pdf(file_path):
    tables_data = []
    if os.path.exists(file_path):
        try:
            tables = camelot.read_pdf(file_path, pages='all')
            for table in tables:
                tables_data.append(table.df.to_json(orient="records"))
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
    else:
        print("File not found for table extraction.")
    return tables_data


@app.route('/debug_session')
def debug_session():
    return str(session)  # Print current session data



@app.route('/toggle_menu', methods=['POST'])
def toggle_menu():
    session['menu_open'] = not session.get('menu_open', False)  # Toggle state
    return '', 204  # Return success with no content

from flask import send_from_directory, jsonify, session, request

@app.route('/proposal_view/<int:file_id>')
@login_required
def proposal_view(file_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    
    # Fetch file details based on the logged-in user
    cursor.execute("SELECT file_path, extracted_text FROM files WHERE id = %s AND user_email = %s", 
                   (file_id, session['user_email']))
    
    file_record = cursor.fetchone()
    
    cursor.close()
    conn.close()

    if file_record:
        extracted_text = file_record.get('extracted_text', '')  # Ensure it’s not None
        print("Extracted Text:", extracted_text)  # Debugging

        return render_template('proposal_view.html', extracted_text=extracted_text)
    else:  
        flash("File not found or access denied", "error")
        return redirect(url_for('proposal_upload'))


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


# Initialize Flask-Session correctly before running the app
Session(app)

if __name__ == "__main__":
    app.run(debug=True, port=3000)  # Ensure only this instance exists
