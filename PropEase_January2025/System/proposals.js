const fileList = document.getElementById('file-list');
const popup = document.getElementById('popup');
const fileContent = document.getElementById('file-content');
const fileAnalytics = document.getElementById('file-analytics');

// Function to display uploaded files as cards
function displayUploadedFiles() {
    const storedFiles = JSON.parse(localStorage.getItem('uploadedFiles')) || [];
    fileList.innerHTML = ''; // Clear current files

    storedFiles.forEach((file, index) => {
        const fileCard = document.createElement('div');
        fileCard.className = 'file-card';
        fileCard.innerHTML = `
            <h3>${file.name}</h3>
            <p>Size: ${(file.size / 1024).toFixed(2)} KB</p>
            <p>Uploaded on: ${new Date(file.uploadedAt).toLocaleDateString()}</p>
        `;
        fileCard.addEventListener('click', () => openFilePopup(file));
        fileList.appendChild(fileCard);
    });
}

// Function to handle file upload
function uploadFiles() {
    const fileInput = document.getElementById('file-upload');
    const files = Array.from(fileInput.files);
    
    // Get previously stored files or empty array if no files
    const storedFiles = JSON.parse(localStorage.getItem('uploadedFiles')) || [];

    // Add new files to storage
    files.forEach(file => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const fileObj = {
                name: file.name,
                size: file.size,
                content: e.target.result,
                uploadedAt: new Date()
            };
            storedFiles.push(fileObj);
            localStorage.setItem('uploadedFiles', JSON.stringify(storedFiles));
            displayUploadedFiles();
        };
        reader.readAsText(file); // Read file content
    });
}

// Function to open the file content popup
function openFilePopup(file) {
    fileContent.innerHTML = `<pre>${file.content}</pre>`;
    fileAnalytics.innerHTML = `
        <p>File Name: ${file.name}</p>
        <p>File Size: ${(file.size / 1024).toFixed(2)} KB</p>
        <p>Upload Date: ${new Date(file.uploadedAt).toLocaleDateString()}</p>
    `;
    popup.style.display = 'block';
}

// Function to close the file popup
function closePopup() {
    popup.style.display = 'none';
}

// Display files on page load
document.addEventListener('DOMContentLoaded', displayUploadedFiles);