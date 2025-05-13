// This is a direct fix for the popup issue
document.addEventListener('DOMContentLoaded', function() {
    console.log("Loading direct popup fix");
    
    // Get the popup element
    const popup = document.getElementById('popup');
    
    // Get the results button
    const resultsButton = document.getElementById('resultsButton');
    
    // Get the close button
    const closeButton = document.getElementById('closeButton');
    
    // Function to show the popup
    function showPopup() {
        console.log("showPopup called");
        popup.style.display = 'flex';
        
        // Trigger analysis
        if (typeof analyzePDFContent === 'function') {
            console.log("Calling analyzePDFContent");
            analyzePDFContent();
        }
        
        // Generate titles
        if (typeof generateTitles === 'function') {
            console.log("Calling generateTitles");
            generateTitles();
        }
    }
    
    // Function to hide the popup
    function hidePopup() {
        console.log("hidePopup called");
        popup.style.display = 'none';
    }
    
    // Add click event to results button
    if (resultsButton) {
        console.log("Adding click event to results button");
        resultsButton.onclick = function(e) {
            console.log("Results button clicked");
            e.preventDefault();
            showPopup();
            return false;
        };
    } else {
        console.error("Results button not found");
    }
    
    // Add click event to close button
    if (closeButton) {
        console.log("Adding click event to close button");
        closeButton.onclick = function(e) {
            console.log("Close button clicked");
            e.preventDefault();
            hidePopup();
            return false;
        };
    } else {
        console.error("Close button not found");
    }
    
    // Close popup when clicking outside
    window.onclick = function(event) {
        if (event.target === popup) {
            hidePopup();
        }
    };
    
    // Override the global togglePopup function
    window.togglePopup = showPopup;
    window.closePopup = hidePopup;
    
    console.log("Direct popup fix loaded");
});