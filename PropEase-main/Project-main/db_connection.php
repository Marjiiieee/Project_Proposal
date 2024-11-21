<?php
// Database connection details
$servername = "localhost"; // XAMPP runs on localhost
$username = "root";        // Default username for XAMPP
$password = "";            // Default password is empty
$dbname = "test";          // Your database name

// Create connection
try {
    $conn = new mysqli($servername, $username, $password, $dbname);

    // Check connection
    if ($conn->connect_error) {
        throw new Exception("Connection failed: " . $conn->connect_error);
    }

} catch (Exception $e) {
    die("Error: " . $e->getMessage());
}
?>
