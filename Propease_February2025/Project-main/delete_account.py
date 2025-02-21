from flask import Flask, session, jsonify, request
from flask_mysqldb import MySQL

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

# Database configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'propease'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# Initialize MySQL object
mysql = MySQL(app)

@app.route('/delete-account', methods=['POST'])
def delete_account():
    # Check if the user is logged in
    if 'email' in session:
        email = session['email']

        try:
            # Connect to the database and prepare the DELETE query
            cursor = mysql.connection.cursor()
            delete_query = "DELETE FROM registration WHERE email = %s"
            cursor.execute(delete_query, (email,))
            mysql.connection.commit()

            # Check if any row was affected
            if cursor.rowcount > 0:
                session.clear()  # Clear session data after account deletion
                return jsonify({"status": "success", "message": "Account deleted successfully."})
            else:
                return jsonify({"status": "error", "message": "Failed to delete the account."})
        except Exception as e:
            return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"})
        finally:
            cursor.close()
    else:
        # User not logged in
        return jsonify({"status": "error", "message": "User is not logged in."})

if __name__ == "__main__":
    app.run(debug=True)
