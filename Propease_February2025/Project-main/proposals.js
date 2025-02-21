// TOGGLE MENU
function toggleMenu() {
    const body = document.body;
    body.classList.toggle('menu-open');

    const menu = document.getElementById('menu');
    if (body.classList.contains('menu-open')) {
        menu.style.left = '0'; // Show the menu
    } else {
        menu.style.left = '-300px'; // Hide the menu
    }
}

// Close menu only if clicking outside
document.addEventListener('click', function(event) {
    const menu = document.getElementById('menu');
    const icon = document.querySelector('.menu-icon');

    // Check if the menu is open and if the click is outside both the menu and the icon
    if (!menu.contains(event.target) && !icon.contains(event.target) && document.body.classList.contains('menu-open')) {
        menu.style.left = '-300px'; // Hide the menu
        document.body.classList.remove('menu-open');
    }
});

// PROPOSAL PAGE -- UPLOADED FILES
function displayUploadedFiles() {
    const fileInput = document.getElementById('file-upload');
    const fileList = document.getElementById('file-list');

    // Loop through the newly selected files and append them to the list
    Array.from(fileInput.files).forEach((file) => {
        const fileContainer = document.createElement('div');

        // Link wrapper for file bubble
        const fileLink = document.createElement('a');
        // Use 'file.name' to pass the file name as a URL parameter
        fileLink.href = `proposal_view.html?file=${encodeURIComponent(file.name)}`; 
        fileLink.className = 'file-bubble'; 

        // File icon
        const fileIcon = document.createElement('i');
        fileIcon.className = 'fas fa-file-alt file-icon'; 
        fileLink.appendChild(fileIcon);

        // File caption
        const fileCaption = document.createElement('div');
        fileCaption.className = 'file-caption';
        fileCaption.textContent = file.name; 

        fileContainer.appendChild(fileLink);
        fileContainer.appendChild(fileCaption);
        fileList.appendChild(fileContainer);

        // Save the file in local storage to pass it to the next page
        const reader = new FileReader();
        reader.onload = function(event) {
            localStorage.setItem(file.name, event.target.result);
        };
        reader.readAsText(file); // For text-based files (adjust for PDFs, DOCX as needed)
    });

    // Clear the file input to allow re-uploading the same file if needed
    fileInput.value = '';
}

function openFileView(file) {
    // Redirect to the second interface page with the file name as a parameter
    window.location.href = `proposal_view.html?file=${encodeURIComponent(file.name)}`;
}

// PROPOSAL PAGE -- VIEWING UPLOADED FILES
document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const fileName = urlParams.get('file');

    if (fileName) {
        // Retrieve file content from localStorage (or adjust this for URL parameter usage)
        const fileContent = localStorage.getItem(fileName);

        if (fileContent) {
            document.getElementById('file-view').innerHTML = `<pre>${fileContent}</pre>`; // Display the content
        } else {
            document.getElementById('file-view').innerHTML = '<p>File content not found.</p>';
        }
    }

    // Ensure modal remains hidden initially
    const modal = document.getElementById("recommendationModal");
    modal.style.display = "none";

    // Ensure recommendation modal is only triggered by button click
    const recommendationButton = document.querySelector('.recommendation-button');
    if (recommendationButton) {
        recommendationButton.style.pointerEvents = "auto"; // Ensure it's clickable
        recommendationButton.style.cursor = "pointer"; // Show hand cursor

        recommendationButton.addEventListener("click", function(event) {
            event.preventDefault(); // Prevent any unintended default behavior
            console.log("Recommendation button clicked!"); // Debugging message
            toggleModal();
        });
    } else {
        console.warn("Recommendation button not found in the DOM!");
    }
});

// PROPOSAL PAGE -- RECOMMENDATION POP UP
function toggleModal() {
    const modal = document.getElementById("recommendationModal");
    modal.style.display = modal.style.display === "flex" ? "none" : "flex";
}

function closeModal() {
    const modal = document.getElementById("recommendationModal");
    modal.style.display = "none";
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById("recommendationModal");
    if (event.target === modal) {
        modal.style.display = "none";
    }
};

// PROPOSAL PAGE -- SPEECH TO TEXT
let recognition;
let isRecognizing = false;

function initializeSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = function() {
            console.log('Speech recognition started...');
            isRecognizing = true; // Set to true when recognition starts
        };

        recognition.onend = function() {
            console.log('Speech recognition ended.');
            isRecognizing = false; // Set to false when recognition ends
        };

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            console.log('Speech recognized:', transcript);
            document.getElementById('yourTextAreaId').value = transcript; // Replace 'yourTextAreaId' with the target element ID
        };

        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            stopSpeechRecognition(); // Stop recognition on error
        };
    } else {
        alert('Your browser does not support speech recognition. Please use Chrome or another compatible browser.');
    }
}

function toggleSpeechToText() {
    if (!recognition) {
        initializeSpeechRecognition();
    }

    if (isRecognizing) {
        stopSpeechRecognition();
    } else {
        startSpeechRecognition();
    }
}

function startSpeechRecognition() {
    if (recognition && !isRecognizing) {
        recognition.start();
    }
}

function stopSpeechRecognition() {
    if (recognition && isRecognizing) {
        recognition.stop();
    }
}