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
