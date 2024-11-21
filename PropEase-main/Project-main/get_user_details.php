<?php
session_start();
header('Content-Type: application/json');

if (isset($_SESSION['email'])) {
    $email = $_SESSION['email'];

    include 'db_connection.php';

    $stmt = $conn->prepare("SELECT email, password FROM registration WHERE email = ?");
    if (!$stmt) {
        echo json_encode(['status' => 'error', 'message' => 'Database query preparation failed.']);
        exit();
    }
    $stmt->bind_param("s", $email);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($result->num_rows > 0) {
        $row = $result->fetch_assoc();
        echo json_encode([
            'status' => 'success',
            'email' => $row['email'],
            'password' => $row['password']
        ]);
    } else {
        echo json_encode(['status' => 'error', 'message' => 'User not found in the database.']);
    }

    $stmt->close();
    $conn->close();
} else {
    echo json_encode(['status' => 'error', 'message' => 'No active session found. Please log in.']);
}
?>
