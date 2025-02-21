from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from flask_mysqldb import MySQL
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash  # For password hashing

app = Flask(__name__)

# Configuration for MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''  # Set your MySQL root password here
app.config['MYSQL_DB'] = 'propease'

# Configure session (store sessions server-side)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

mysql = MySQL(app)

# Route for rendering the login page
@app.route('/')
def index():
    return render_template('main.html')

# Route for login
@app.route('/login', methods=['POST'])
def login():
    try:
        # Get form data
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return render_template('main.html', error="Please enter both email and password.")

        # Connect to the database
        cursor = mysql.connection.cursor()
        query = "SELECT password FROM registration WHERE email = %s"
        cursor.execute(query, (email,))
        result = cursor.fetchone()

        if result:
            stored_password = result[0]  # Extract hashed password from result

            # Compare entered password with stored hashed password
            if check_password_hash(stored_password, password):
                # Set the session
                session['email'] = email
                return redirect(url_for('dashboard'))  # Redirect to the dashboard/home
            else:
                return render_template('main.html', error="Invalid password. Please try again.")
        else:
            return render_template('main.html', error="Email not registered. Please sign up first.")

    except Exception as e:
        # Log the error for debugging (optional)
        print(f"Error during login: {e}")
        return render_template('main.html', error="An unexpected error occurred. Please try again.")
    finally:
        cursor.close()

# Route for dashboard/home after successful login
@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        # Render home.html or any other template as the user’s dashboard
        return render_template('home.html', email=session['email'])  # Pass the email to the template
    else:
        return redirect(url_for('index'))

# Route for logout
@app.route('/logout')
def logout():
    # Clear the session and redirect to the login page
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.secret_key = 'your_secret_key'  # Replace with a strong secret key
    app.run(debug=True)
