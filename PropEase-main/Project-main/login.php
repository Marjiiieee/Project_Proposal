<?php
session_start();
include 'db_connection.php';

header('Content-Type: application/json'); // Set content type to JSON

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $email = $_POST['email'];
    $password = $_POST['password'];

    // Prepare statement to select the user from the database
    $stmt = $conn->prepare("SELECT password FROM registration WHERE email = ?");
    $stmt->bind_param("s", $email);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($result->num_rows > 0) {
        $row = $result->fetch_assoc();
        $stored_password = $row['password'];

        // Directly compare the passwords
        if ($password === $stored_password) {
            $_SESSION['email'] = $email; // Set session variable
            echo json_encode(["status" => "success", "message" => "Login successful"]);
        } else {
            echo json_encode(["status" => "error", "message" => "Invalid password. Please try again."]);
        }
    } else {
        echo json_encode(["status" => "error", "message" => "Email not registered. Please sign up first."]);
    }

    $stmt->close();
} else {
    echo json_encode(["status" => "error", "message" => "Invalid request method."]);
}

$conn->close();
?>
