
// ARCHIVE PAGE
document.addEventListener('DOMContentLoaded', function () {
    const archiveList = document.getElementById('archive-list');
    const urlParams = new URLSearchParams(window.location.search);
    const fileId = urlParams.get('fileId');

    if (fileId) {
        const fileItem = document.createElement('div');
        fileItem.classList.add('file-item');
        fileItem.innerHTML = `<img src="file-icon.png" alt="file"><span>Archived File ${fileId}</span>`;
        archiveList.appendChild(fileItem);
    }
});

// TOGGLE MENU
function toggleMenu() {
    const menu = document.getElementById('menu');
    menu.style.left = menu.style.left === '0px' ? '-300px' : '0px';
}

// Close menu when clicking outside of it
document.addEventListener('click', function(event) {
    const menu = document.getElementById('menu');
    const icon = document.querySelector('.menu-icon');
    if (!menu.contains(event.target) && !icon.contains(event.target)) {
        menu.style.left = '-300px'; // Hide the menu
    }
});
