from flask import Flask, request, jsonify
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

# Database connection settings
DB_CONFIG = {
    'host': 'localhost',  # Replace with your database host
    'user': 'root',       # Replace with your database username
    'password': '',       # Replace with your database password
    'database': 'propease'  # Replace with your database name
}

def create_connection():
    """Create a database connection."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"Error: {e}")
    return None

@app.route('/signup', methods=['POST'])
def signup():
    # Validate input data
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required."}), 400

    try:
        conn = create_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed."}), 500

        cursor = conn.cursor(dictionary=True)

        # Check if email is already registered
        cursor.execute("SELECT * FROM registration WHERE email = %s", (email,))
        result = cursor.fetchone()

        if result:
            return jsonify({"status": "error", "message": "Email already exists."}), 409

        # Insert new user into the database
        cursor.execute("INSERT INTO registration (email, password) VALUES (%s, %s)", (email, password))
        conn.commit()

        return jsonify({"status": "success", "message": "Sign-up successful!", "redirect": "main.html"}), 201

    except Error as e:
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500
    finally:
        if conn:
            conn.close()

    return jsonify({"status": "error", "message": "Sign-up failed."}), 500

if __name__ == '__main__':
    app.run(debug=True)
