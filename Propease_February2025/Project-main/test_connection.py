import MySQLdb

# Database configuration
host = 'localhost'
user = 'root'
password = ''
database = 'propease'

try:
    # Attempt to connect to the database
    connection = MySQLdb.connect(host=host, user=user, passwd=password, db=database)
    print("Connection successful!")
    connection.close()
except Exception as e:
    print(f"Failed to connect to the database: {str(e)}")
