from flask import Flask, jsonify, session
from flask_mysqldb import MySQL
import MySQLdb.cursors

# Initialize Flask app
app = Flask(__name__)

# Secret key for session
app.secret_key = 'your_secret_key'

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'propease'

# Initialize MySQL
mysql = MySQL(app)

# Route to get user details
@app.route('/get-user-details', methods=['GET'])
def get_user_details():
    # Check if the user is logged in
    if 'email' not in session:
        return jsonify({
            'status': 'error',
            'message': 'User is not logged in.'
        }), 401

    # Get the logged-in user's email from the session
    email = session['email']

    try:
        # Create a cursor and execute the query
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        query = "SELECT email, password FROM registration WHERE email = %s"
        cursor.execute(query, (email,))
        result = cursor.fetchone()

        # Check if the user exists
        if result:
            # Mask the password with asterisks
            masked_password = '*' * len(result['password'])
            return jsonify({
                'status': 'success',
                'email': result['email'],
                'password': masked_password
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'User not found.'
            }), 404

    except Exception as e:
        # Handle exceptions
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
