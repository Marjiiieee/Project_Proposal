from flask import Flask
from flask_mysqldb import MySQL

# Initialize Flask app
app = Flask(__name__)

# Configure database connection details
app.config['MYSQL_HOST'] = 'localhost'         # Hostname
app.config['MYSQL_USER'] = 'root'             # Username
app.config['MYSQL_PASSWORD'] = ''             # Password
app.config['MYSQL_DB'] = 'propease'           # Database name
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'  # Fetch data as dictionary for convenience

# Initialize MySQL object
mysql = MySQL(app)

# Test database connection
@app.route('/test-db-connection', methods=['GET'])
def test_db_connection():
    try:
        # Open a database cursor
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT DATABASE();")
        db_name = cursor.fetchone()

        return {"status": "success", "message": f"Connected to database: {db_name['DATABASE()']}"}
    except Exception as e:
        return {"status": "error", "message": f"Database connection failed: {str(e)}"}

if __name__ == "__main__":
    app.run(debug=True)
