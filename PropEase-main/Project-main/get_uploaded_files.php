<?php
session_start();
include 'db_connection.php';

// Set header to return JSON
header('Content-Type: application/json');

// Check if the user is logged in
if (!isset($_SESSION['email'])) {
    echo json_encode([
        "status" => "error",
        "message" => "User not logged in."
    ]);
    exit();
}

$user_email = $_SESSION['email'];

try {
    // Fetch files associated with the logged-in user
    $stmt = $conn->prepare("SELECT file_name, file_path FROM files WHERE user_email = ?");
    if (!$stmt) {
        throw new Exception("Failed to prepare the SQL statement: " . $conn->error);
    }

    $stmt->bind_param("s", $user_email);
    $stmt->execute();
    $result = $stmt->get_result();

    $files = [];
    while ($row = $result->fetch_assoc()) {
        $files[] = [
            "file_name" => $row['file_name'],
            "file_path" => $row['file_path'],
        ];
    }

    echo json_encode([
        "status" => "success",
        "files" => $files
    ]);

    $stmt->close();
    $conn->close();
} catch (Exception $e) {
    // Catch errors and return an appropriate response
    echo json_encode([
        "status" => "error",
        "message" => "An error occurred while fetching files: " . $e->getMessage()
    ]);
}
?>
