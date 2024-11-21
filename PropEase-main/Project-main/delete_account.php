<?php
session_start();
header('Content-Type: application/json');

// Check if the user is logged in
if (isset($_SESSION['email'])) {
    $email = $_SESSION['email'];

    // Include database connection
    include 'db_connection.php';

    // Prepare the DELETE statement
    $stmt = $conn->prepare("DELETE FROM registration WHERE email = ?");
    if (!$stmt) {
        echo json_encode(['status' => 'error', 'message' => 'Failed to prepare the query.']);
        exit();
    }

    $stmt->bind_param("s", $email);
    $stmt->execute();

    // Check if any row was affected
    if ($stmt->affected_rows > 0) {
        // Account deleted successfully
        session_destroy(); // Destroy the session after account deletion
        echo json_encode(['status' => 'success', 'message' => 'Account deleted successfully.']);
    } else {
        echo json_encode(['status' => 'error', 'message' => 'Failed to delete the account.']);
    }

    $stmt->close();
    $conn->close();
} else {
    // No user logged in
    echo json_encode(['status' => 'error', 'message' => 'User is not logged in.']);
}
?>
