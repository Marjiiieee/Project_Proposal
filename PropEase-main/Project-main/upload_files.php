<?php
session_start();
include 'db_connection.php';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $user_email = $_SESSION['email'] ?? 'unknown_user'; // User email from the session
    $upload_dir = 'uploads/';
    $responses = [];

    // Ensure the upload directory exists
    if (!is_dir($upload_dir)) {
        mkdir($upload_dir, 0777, true); // Create upload directory if it doesn't exist
    }

    foreach ($_FILES['files']['tmp_name'] as $key => $tmp_name) {
        $file_name = $_FILES['files']['name'][$key];
        $file_tmp = $_FILES['files']['tmp_name'][$key];
        $file_path = 'Project-main/' . $upload_dir . basename($file_name);

        // Validate file type
        $allowed_extensions = ['pdf', 'doc', 'docx', 'wps'];
        $file_extension = strtolower(pathinfo($file_name, PATHINFO_EXTENSION));

        if (!in_array($file_extension, $allowed_extensions)) {
            $responses[] = ["status" => "error", "message" => "$file_name is not a supported file type."];
            continue;
        }

        // Move file to the upload directory
        if (move_uploaded_file($file_tmp, $upload_dir . basename($file_name))) {
            // Save file info to the database
            $stmt = $conn->prepare("INSERT INTO files (user_email, file_name, file_path, upload_date) VALUES (?, ?, ?, NOW())");
            $stmt->bind_param("sss", $user_email, $file_name, $file_path);

            if ($stmt->execute()) {
                $responses[] = ["status" => "success", "file" => $file_name];
            } else {
                $responses[] = ["status" => "error", "message" => "Failed to save file info to database: " . $stmt->error];
            }
            $stmt->close();
        } else {
            $responses[] = ["status" => "error", "message" => "Failed to upload $file_name."];
        }
    }

    echo json_encode($responses);
}

$conn->close();
?>
