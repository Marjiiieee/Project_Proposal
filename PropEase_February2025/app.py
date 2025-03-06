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

# Home Route - Main Dashboard
@app.route('/home')
@login_required
def home():
    return render_template("home.html")

# 🔹 LOGIN ROUTE
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template("main.html")

    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        return render_template("main.html", error="Email and password are required")

    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, email, password FROM registration WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

    except mysql.connector.Error as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    if user and user["password"] == password:
        session.permanent = True  # Ensure session persists
        session['user_id'] = user['id']
        session['user_email'] = user['email']  # Store email for consistent access

        flash("Login successful!", "success")
        return redirect(url_for('home'))

    flash("Invalid email or password", "error")
    return render_template("main.html", error="Invalid email or password")

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

    try:
        conn = get_db()
        cursor = conn.cursor()

        # Insert user data into the database (Plain Text ⚠️)
        cursor.execute("INSERT INTO registration (email, password) VALUES (%s, %s)", (email, password))
        conn.commit()

        cursor.close()
        conn.close()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))

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
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        if file and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            secure_name = secure_filename(unique_filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_name)

            try:
                file.save(file_path)

                # Extract text from PDF
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    extracted_text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                
                # Save file details and extracted text to the database
                cursor.execute("""
                    INSERT INTO files (user_email, file_name, file_path, extracted_text)
                    VALUES (%s, %s, %s, %s)
                """, (session['user_email'], unique_filename, file_path, extracted_text))
                conn.commit()
                return jsonify({"status": "success", "file_id": unique_filename}), 200
            
            except Exception as e:
                return jsonify({"status": "error", "message": f"Error processing file: {str(e)}"}), 500

        return jsonify({"status": "error", "message": "Invalid file type. Only PDFs are allowed."}), 400

    cursor.execute("SELECT id, file_name FROM files WHERE user_email = %s", (session['user_email'],))
    files = [{'id': row['id'], 'file_name': row['file_name']} for row in cursor.fetchall()]

    cursor.close()
    conn.close()
    return render_template('proposal_upload.html', files=files)

# Archive Page
@app.route('/archive')
@login_required
def archive():
    return render_template("archive.html")

# Account/Profile Page
@app.route('/account')
@login_required
def account():
    return render_template("account.html")

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

#Proposal View
import time

@app.route('/upload_proposal', methods=['POST'])
def upload_proposal():
    if 'proposal_file' not in request.files:
        flash("No file part", "error")
        return redirect(request.url)

    proposal_file = request.files['proposal_file']

    if proposal_file.filename == '':
        flash("No selected file", "error")
        return redirect(request.url)

    if proposal_file:
        filename = secure_filename(proposal_file.filename)
        unique_filename = f"{int(time.time())}_{filename}"  # Append timestamp
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        proposal_file.save(file_path)

        cursor = mysql.connection.cursor()
        cursor.execute(
            "INSERT INTO proposals (user_email, file_name, file_path) VALUES (%s, %s, %s)",
            (session['user_email'], unique_filename, file_path)
        )
        mysql.connection.commit()
        cursor.close()

        flash("File uploaded successfully", "success")
        return redirect(url_for('proposal_upload'))

    flash("File upload failed", "error")
    return redirect(request.url)

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

# UPLOAD
@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"})

    if "email" not in session:
        return jsonify({"success": False, "error": "User not logged in"})

    # Generate unique filename and store full file path
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(file_path)  # Save file

        # Save filename and file path to database
        connection = get_db()
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO files (user_email, file_name, file_path) VALUES (%s, %s, %s)",
                (session["user_email"], filename, file_path)  # Consistent session key
            )
            connection.commit()

        return jsonify({"success": True, "message": "File uploaded successfully"})

    except mysql.connector.Error as e:
        return jsonify({"success": False, "error": f"Database error: {str(e)}"})
    
@app.route('/toggle_menu', methods=['POST'])
def toggle_menu():
    session['menu_open'] = not session.get('menu_open', False)  # Toggle state
    return '', 204  # Return success with no content

from flask import send_from_directory, jsonify, session, request

@app.route('/view_proposal/<int:file_id>')
@login_required
def view_proposal(file_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT file_path, extracted_text FROM files WHERE id = %s AND user_email = %s", (file_id, session['user_email']))
    file_record = cursor.fetchone()
    cursor.close()
    conn.close()

    if file_record:
        return render_template('proposal_view.html', text=file_record['extracted_text'])
    else:
        flash("File not found or access denied", "error")
        return redirect(url_for('proposal_upload'))

@app.route('/uploaded_file', methods=['GET'])
@app.route('/uploaded_file/<filename>', methods=['GET'])
def uploaded_file(filename=None):
    user_email = session.get('user_email')

    if not user_email:
        return jsonify({"error": "User not logged in"}), 401  

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    if filename:  
        # Serve a specific file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        else:
            return jsonify({"error": "File not found"}), 404
    else:
        # List all files uploaded by the user
        cursor.execute("SELECT id, file_name, file_path FROM files WHERE user_email = %s", (user_email,))
        files = cursor.fetchall()

        cursor.close()
        db.close()

        return jsonify(files)


# Initialize Flask-Session correctly before running the app
Session(app)

if __name__ == "__main__":
    app.run(debug=True, port=3000)  # Ensure only this instance exists
