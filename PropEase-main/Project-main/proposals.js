// Upload Files
async function uploadFiles(event) {
    event.preventDefault(); // Prevent default form submission

    const fileInput = document.getElementById('file-upload');
    const formData = new FormData();

    // Append all selected files to the form data
    Array.from(fileInput.files).forEach((file) => {
        formData.append('files[]', file);
    });

    try {
        const response = await fetch('upload_files.php', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        result.forEach((fileResult) => {
            if (fileResult.status === 'success') {
                console.log(`Uploaded: ${fileResult.file}`);
            } else {
                console.error(`Error: ${fileResult.message}`);
            }
        });

        // Reload the file list after uploading
        fetchUploadedFiles();
    } catch (error) {
        console.error('Error uploading files:', error);
        alert('An error occurred during the upload.');
    }
}

// Fetch and Display Uploaded Files
async function fetchUploadedFiles() {
    try {
        const response = await fetch('get_uploaded_files.php');
        const result = await response.json();

        if (result.status === 'success') {
            const fileList = document.getElementById('file-list');
            fileList.innerHTML = ''; // Clear the existing list

            result.files.forEach((file) => {
                const fileContainer = document.createElement('div');
                fileContainer.className = 'file-container';

                // File link to open in the viewer
                const fileLink = document.createElement('a');
                fileLink.href = `proposal_view.html?file_path=${encodeURIComponent(file.file_path)}`;
                fileLink.className = 'file-bubble';

                // File icon
                const fileIcon = document.createElement('i');
                fileIcon.className = 'fas fa-file-alt file-icon';
                fileLink.appendChild(fileIcon);

                // File caption
                const fileCaption = document.createElement('div');
                fileCaption.className = 'file-caption';
                fileCaption.textContent = file.file_name;

                fileContainer.appendChild(fileLink);
                fileContainer.appendChild(fileCaption);
                fileList.appendChild(fileContainer);
            });
        } else {
            console.error(result.message);
        }
    } catch (error) {
        console.error('Error fetching files:', error);
    }
}

// Load the file in the viewer
window.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const filePath = urlParams.get('file_path');
    const fileDisplay = document.getElementById('file-preview');
    const unsupportedFile = document.getElementById('unsupported-file');
    const downloadLink = document.getElementById('download-link');

    if (filePath) {
        const fileExtension = filePath.split('.').pop().toLowerCase();

        if (fileExtension === 'pdf') {
            // Display PDF in the iframe
            fileDisplay.style.display = 'block';
            unsupportedFile.style.display = 'none';
            fileDisplay.src = filePath;
        } else if (['doc', 'docx', 'wps'].includes(fileExtension)) {
            // Use Google Docs Viewer for document formats
            fileDisplay.style.display = 'block';
            unsupportedFile.style.display = 'none';
            fileDisplay.src = `https://docs.google.com/viewer?url=${encodeURIComponent(filePath)}&embedded=true`;
        } else {
            // Show unsupported file message
            fileDisplay.style.display = 'none';
            unsupportedFile.style.display = 'block';
            downloadLink.href = filePath;
        }
    }
});

// Toggle the recommendation modal
function toggleModal() {
    const modal = document.getElementById('recommendationModal');
    modal.style.display = modal.style.display === 'flex' ? 'none' : 'flex';
}

// Close the recommendation modal
function closeModal() {
    const modal = document.getElementById('recommendationModal');
    modal.style.display = 'none';
}

// Toggle Menu
function toggleMenu() {
    const menu = document.getElementById('menu');
    menu.style.left = menu.style.left === '0px' ? '-300px' : '0px';
}

// Close menu when clicking outside
document.addEventListener('click', (event) => {
    const menu = document.getElementById('menu');
    const icon = document.querySelector('.menu-icon');
    if (!menu.contains(event.target) && !icon.contains(event.target)) {
        menu.style.left = '-300px'; // Hide menu
    }
});
