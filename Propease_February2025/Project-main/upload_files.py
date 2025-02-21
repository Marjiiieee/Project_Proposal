import os
import mimetypes
from flask import Flask, request, session, jsonify
from werkzeug.utils import secure_filename
import mysql.connector
from datetime import datetime

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a secure secret key

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="your_db_host",
        user="your_db_user",
        password="your_db_password",
        database="your_db_name"
    )

# Upload directory
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions and MIME types
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'wps'}
ALLOWED_MIME_TYPES = {
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-works'
}

# Helper function: Check allowed file type
def allowed_file(filename, mime_type):
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return extension in ALLOWED_EXTENSIONS and mime_type in ALLOWED_MIME_TYPES

# Route to handle file uploads
@app.route('/upload', methods=['POST'])
def upload_files():
    # Validate user session
    if 'email' not in session:
        return jsonify({"status": "error", "message": "User not logged in."}), 401

    user_email = session['email']
    files = request.files.getlist('files')
    if not files:
        return jsonify({"status": "error", "message": "No files uploaded."}), 400

    responses = []

    for file in files:
        original_filename = file.filename
        if not original_filename:
            responses.append({"status": "error", "message": "Invalid file."})
            continue

        # Get MIME type
        mime_type = mimetypes.guess_type(original_filename)[0]

        # Validate file type
        if not allowed_file(original_filename, mime_type):
            responses.append({
                "status": "error",
                "message": f"{original_filename} is not a supported file type. Only PDF, DOC, DOCX, and WPS are allowed."
            })
            continue

        # Sanitize file name
        sanitized_filename = secure_filename(original_filename)

        # Check for duplicate file names and rename
        absolute_path = os.path.join(app.config['UPLOAD_FOLDER'], sanitized_filename)
        counter = 1
        while os.path.exists(absolute_path):
            name, ext = os.path.splitext(sanitized_filename)
            sanitized_filename = f"{name}_{counter}{ext}"
            absolute_path = os.path.join(app.config['UPLOAD_FOLDER'], sanitized_filename)
            counter += 1

        # Save the file
        try:
            file.save(absolute_path)
        except Exception as e:
            responses.append({
                "status": "error",
                "message": f"Failed to upload {original_filename}. Error: {str(e)}"
            })
            continue

        # Save file details to the database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            relative_path = os.path.join('uploads', sanitized_filename)
            upload_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            query = "INSERT INTO files (user_email, file_name, file_path, upload_date) VALUES (%s, %s, %s, %s)"
            cursor.execute(query, (user_email, sanitized_filename, relative_path, upload_date))
            conn.commit()
            cursor.close()
            conn.close()

            responses.append({"status": "success", "file": sanitized_filename})
        except mysql.connector.Error as err:
            responses.append({
                "status": "error",
                "message": f"Database error while saving {original_filename}: {str(err)}"
            })

    return jsonify(responses)

if __name__ == '__main__':
    app.run(debug=True)
