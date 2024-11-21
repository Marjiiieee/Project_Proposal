<?php
header('Content-Type: application/json'); // Set content type to JSON
include 'db_connection.php'; // Include your database connection file

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Validate input
    $email = filter_var($_POST['email'], FILTER_SANITIZE_EMAIL);
    $password = $_POST['password'];

    // Check if the email and password are provided
    if (empty($email) || empty($password)) {
        echo json_encode(["status" => "error", "message" => "Email and password are required."]);
        exit();
    }

    // Check if the email is already registered
    $stmt = $conn->prepare("SELECT * FROM registration WHERE email = ?");
    if (!$stmt) {
        echo json_encode(["status" => "error", "message" => "Failed to prepare query."]);
        exit();
    }

    $stmt->bind_param("s", $email);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($result->num_rows > 0) {
        echo json_encode(["status" => "error", "message" => "Email already exists."]);
    } else {
        // Insert the new user into the database
        $stmt = $conn->prepare("INSERT INTO registration (email, password) VALUES (?, ?)");
        if (!$stmt) {
            echo json_encode(["status" => "error", "message" => "Failed to prepare query."]);
            exit();
        }

        $stmt->bind_param("ss", $email, $password);

        if ($stmt->execute()) {
            // Redirect to `main.html`
            echo json_encode(["status" => "success", "message" => "Sign-up successful!", "redirect" => "main.html"]);
        } else {
            echo json_encode(["status" => "error", "message" => "Sign-up failed."]);
        }
        $stmt->close();
    }

    $conn->close();
} else {
    echo json_encode(["status" => "error", "message" => "Invalid request method."]);
}
?>
