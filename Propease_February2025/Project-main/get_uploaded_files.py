from flask import Flask, jsonify, session, request
from flask_mysqldb import MySQL
import MySQLdb.cursors

# Initialize Flask app
app = Flask(__name__)

# Secret key for sessions
app.secret_key = 'your_secret_key'

# MySQL database configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'propease'

# Initialize MySQL
mysql = MySQL(app)

# Route to get uploaded files for the logged-in user
@app.route('/get-uploaded-files', methods=['GET'])
def get_uploaded_files():
    # Check if the user is logged in
    if 'email' not in session:
        return jsonify({
            "status": "error",
            "message": "User not logged in. Please log in to view uploaded files."
        }), 401

    user_email = session['email']  # Get logged-in user's email from the session

    try:
        # Connect to the database and execute query
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        query = """
            SELECT id, file_name, file_path, upload_date 
            FROM files 
            WHERE user_email = %s 
            ORDER BY upload_date DESC
        """
        cursor.execute(query, (user_email,))
        files = cursor.fetchall()  # Fetch all results

        if files:
            return jsonify({
                "status": "success",
                "files": files
            })
        else:
            return jsonify({
                "status": "success",
                "files": [],
                "message": "No uploaded files found for this user."
            })

    except Exception as e:
        # Handle exceptions and return error response
        return jsonify({
            "status": "error",
            "message": f"An error occurred while fetching files: {str(e)}"
        }), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
