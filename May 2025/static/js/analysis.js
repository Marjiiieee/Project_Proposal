// Global analysis functions - optimized for faster generation
window.analyzePDFContent = async function() {
    console.log("⭐⭐⭐ GLOBAL analyzePDFContent FUNCTION CALLED ⭐⭐⭐");

    try {
        // Get the file ID from the URL
        const pathParts = window.location.pathname.split('/');
        const fileId = pathParts[pathParts.length - 1];
        console.log(`File ID: ${fileId}`);

        // Get the extracted text
        const extractedText = document.querySelector('#extracted-text')?.innerText || "";
        console.log(`Extracted text length: ${extractedText.length}`);

        // Get the speech text
        const speechText = window.transcriptText || "";
        console.log(`Speech text length: ${speechText.length}`);

        // Start a timer to measure performance
        const startTime = performance.now();

        // Make the API call
        console.log(`Making API call to /analyze_content/${fileId}`);
        const response = await fetch(`/analyze_content/${fileId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                extracted_text: extractedText,
                speech_text: speechText,
                transcribed_text: speechText  // Add this line - using speechText as transcribed_text
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Parse the response
        const results = await response.json();

        // Calculate and log the time taken
        const endTime = performance.now();
        const timeTaken = endTime - startTime;
        console.log(`Analysis completed in ${timeTaken.toFixed(2)}ms`);

        console.log("Analysis results:", results);

        // Update the display
        updateAnalysisDisplay(results);

        // Show the popup
        const popup = document.getElementById('popup');
        if (popup) {
            popup.style.display = 'flex';
            console.log("Showing popup");
        }

        return results;
    } catch (error) {
        console.error("Error in analyzePDFContent:", error);
        alert("An error occurred during analysis. Please try again.");
        throw error; // Re-throw the error so it can be caught by the caller
    }
};

// Global function to update the analysis display
window.updateAnalysisDisplay = function(results) {
    console.log("⭐⭐⭐ GLOBAL updateAnalysisDisplay FUNCTION CALLED ⭐⭐⭐");
    console.log("Results:", results);
    console.log("Similar titles by abstract:", results.similar_titles_by_abstract);

    // Debug elements
    console.log("similarTitlesSection element:", document.getElementById('similarTitlesSection'));
    console.log("similarTitlesList element:", document.getElementById('similarTitlesList'));

    // Debug speech accuracy specifically
    console.log("Speech accuracy value:", results.speech_accuracy);

    if (!results) {
        console.error("No results to display");
        return;
    }

    // Update speech similarity
    const speechProgressBar = document.getElementById('progressSpeech');
    const speechLabel = document.getElementById('speechSimilarityLabel');
    if (speechProgressBar && speechLabel) {
        if (results.speech_similarity !== undefined && results.speech_similarity !== null) {
            // We have a valid similarity score
            const similarityValue = Math.round(results.speech_similarity);
            speechProgressBar.style.width = `${similarityValue}%`;
            speechProgressBar.style.display = 'block';
            speechProgressBar.style.backgroundColor = '#d8b4f8';
            speechLabel.textContent = `${similarityValue}%`;
            console.log(`Updated speech similarity to ${similarityValue}%`);
        } else {
            // No similarity score available (likely due to insufficient words)
            speechProgressBar.style.width = '0%';
            speechProgressBar.style.display = 'block';
            speechProgressBar.style.backgroundColor = '#d8b4f8';
            speechLabel.innerHTML = `<span style="font-size: 0.8em; color: #666;">Insufficient words (min. 50 required)</span>`;
            console.log('Speech similarity not available - insufficient words');
        }
    }

    // Update thesis similarity
    const thesisProgressBar = document.getElementById('progressThesis');
    const thesisLabel = document.getElementById('thesisSimilarityLabel');
    if (thesisProgressBar && thesisLabel && results.thesis_similarity !== undefined) {
        const thesisSimilarityValue = Math.round(results.thesis_similarity);
        thesisProgressBar.style.width = `${thesisSimilarityValue}%`;
        thesisProgressBar.style.display = 'block';
        thesisProgressBar.style.backgroundColor = '#d8b4f8';
        thesisLabel.textContent = `${thesisSimilarityValue}%`;
        console.log(`Updated thesis similarity to ${thesisSimilarityValue}%`);
    }

    // Update similar titles
    const similarTitlesSection = document.getElementById('similarTitlesSection');
    const similarTitlesList = document.getElementById('similarTitlesList');

    if (similarTitlesSection && similarTitlesList) {
        similarTitlesList.innerHTML = '';

        if (results.similar_titles_by_abstract && results.similar_titles_by_abstract.length > 0) {
            similarTitlesSection.style.display = 'block';

            results.similar_titles_by_abstract.forEach(item => {
                const li = document.createElement('li');
                li.innerHTML = `${item.title} <span style="color: #666; font-size: 0.9em;">(${item.similarity}% similarity)</span>`;
                similarTitlesList.appendChild(li);
            });

            console.log(`Updated similar titles list with ${results.similar_titles_by_abstract.length} items`);
        } else {
            similarTitlesSection.style.display = 'none';
            console.log("No contextually similar thesis titles to display");
        }
    }

    // Update discrepancies
    const discrepancyList = document.getElementById('discrepancyList');
    if (discrepancyList && results.missed_keypoints && results.added_keypoints) {
        discrepancyList.innerHTML = '';

        // Add missed keypoints
        results.missed_keypoints.forEach(point => {
            const li = document.createElement('li');
            li.innerHTML = `🔻 <strong>Missing in speech:</strong> ${point}`;
            discrepancyList.appendChild(li);
        });

        // Add extra keypoints
        results.added_keypoints.forEach(point => {
            const li = document.createElement('li');
            li.innerHTML = `🔺 <strong>Extra in speech:</strong> ${point}`;
            discrepancyList.appendChild(li);
        });

        console.log("Updated discrepancies");
    }

    // Update title recommendations
    const titleRecommendationsSection = document.getElementById('titleRecommendationsSection');
    if (titleRecommendationsSection && results.suggested_titles && results.suggested_titles.length > 0) {
        titleRecommendationsSection.innerHTML = '<h5>Title Recommendations:</h5>';

        // Different reasons for each title
        const reasons = [
            "This title highlights the main focus and technical aspects of your research.",
            "This title emphasizes the innovative approach and potential impact of your work.",
            "This title presents your research in a professional academic context with clear focus.",
            "This title showcases the problem-solving aspects and practical applications of your work.",
            "This title emphasizes the methodological contributions of your research."
        ];

        // Add titles to the section
        results.suggested_titles.forEach((title, index) => {
            const div = document.createElement('div');
            div.className = 'recommendation-item';
            div.innerHTML = `
                <div class="recommendation-text">${title}</div>
                <div class="recommendation-reason">
                    ${reasons[index % reasons.length]}
                </div>
            `;
            titleRecommendationsSection.appendChild(div);
        });

        console.log(`Added ${results.suggested_titles.length} title recommendations`);
    }

    console.log("Analysis display updated successfully");
};

// Global function to generate titles
window.generateTitles = async function() {
    console.log("⭐⭐⭐ GLOBAL generateTitles FUNCTION CALLED ⭐⭐⭐");

    try {
        // Get the extracted text
        const extractedText = document.querySelector('#extracted-text')?.innerText || "";
        console.log(`Extracted text length: ${extractedText.length}`);

        // Get the speech text
        const speechText = window.transcriptText || "";
        console.log(`Speech text length: ${speechText.length}`);

        // Show loading indicator
        const titleRecommendationsSection = document.getElementById('titleRecommendationsSection');
        if (titleRecommendationsSection) {
            titleRecommendationsSection.innerHTML = `
                <h5 style="color: #6a1b9a; font-weight: bold; font-family: 'Comfortaa', sans-serif;">
                    Title Recommendations:
                </h5>
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
                    padding: 20px;
                    border-radius: 16px;
                    box-shadow: 0 4px 20px rgba(128, 90, 213, 0.2);
                    margin-top: 10px;
                    font-family: 'Comfortaa', sans-serif;
                    color: #4a0072;
                ">
                    <div style="margin-right: 10px;">
                        <i class="fas fa-spinner fa-spin" style="font-size: 18px; color: #6a1b9a;"></i>
                    </div>
                    <div style="font-size: 15px; font-family: 'Comfortaa', sans-serif;">
                        Generating Title Recommendations...
                    </div>
                </div>
            `;
        }

        // Make the API call
        console.log("Making API call to /generate_title");
        const response = await fetch('/generate_title', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                extracted_text: extractedText,
                speech_text: speechText
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Parse the response
        const data = await response.json();
        console.log("Title generation results:", data);

        // Update the display
        if (titleRecommendationsSection && data.titles && data.titles.length > 0) {
            titleRecommendationsSection.innerHTML = '<h5>Title Recommendations:</h5>';

            // Different reasons for each title
            const reasons = [
                "This title highlights the main focus and technical aspects of your research.",
                "This title emphasizes the innovative approach and potential impact of your work.",
                "This title presents your research in a professional academic context with clear focus.",
                "This title showcases the problem-solving aspects and practical applications of your work.",
                "This title emphasizes the methodological contributions of your research."
            ];

            // Add titles to the section
            data.titles.forEach((title, index) => {
                const div = document.createElement('div');
                div.className = 'recommendation-item';
                div.innerHTML = `
                    <div class="recommendation-text">${title}</div>
                    <div class="recommendation-reason">
                        ${reasons[index % reasons.length]}
                    </div>
                `;
                titleRecommendationsSection.appendChild(div);
            });

            console.log(`Added ${data.titles.length} title recommendations`);
        } else {
            if (titleRecommendationsSection) {
                titleRecommendationsSection.innerHTML = `
                    <h5>Title Recommendations:</h5>
                    <div style="padding: 10px; color: #666;">
                        No title recommendations could be generated. Please try again.
                    </div>
                `;
            }
        }
        return data;
    } catch (error) {
        console.error("Error in generateTitles:", error);
        const titleRecommendationsSection = document.getElementById('titleRecommendationsSection');
        if (titleRecommendationsSection) {
            titleRecommendationsSection.innerHTML = `
                <h5>Title Recommendations:</h5>
                <div style="padding: 10px; color: #666;">
                    An error occurred while generating title recommendations. Please try again.
                </div>
            `;
        }
    }
};

// Global function to refresh the analysis
window.reAnalyzeAndShow = async function() {
    console.log("⭐⭐⭐ GLOBAL reAnalyzeAndShow FUNCTION CALLED ⭐⭐⭐");

    // Show loading state
    const button = document.querySelector('.refresh-button');
    if (button) {
        button.classList.add('loading');
    }

    const cornerLoading = document.querySelector('.corner-loading');
    if (cornerLoading) {
        cornerLoading.style.display = 'flex';
    }

    try {
        // Call the analysis function
        await window.analyzePDFContent();

        // Generate titles
        await window.generateTitles();

        console.log("Analysis refreshed successfully");
    } catch (error) {
        console.error("Error in reAnalyzeAndShow:", error);
        alert("An error occurred during analysis. Please try again.");
    } finally {
        // Hide loading state
        if (button) {
            button.classList.remove('loading');
        }

        if (cornerLoading) {
            cornerLoading.style.display = 'none';
        }
    }
};

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log("⭐⭐⭐ GLOBAL ANALYSIS.JS LOADED ⭐⭐⭐");

    // Set up global variables
    window.transcriptText = window.transcriptText || "";

    // Set up arrow button handler with enhanced functionality
    const resultsButton = document.getElementById('resultsButton');
    if (resultsButton) {
        // Make the button more noticeable
        resultsButton.style.transition = 'all 0.3s ease';
        resultsButton.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';

        // Add a pulsing effect to draw attention
        const pulseAnimation = `
            @keyframes pulse {
                0% { transform: scale(1); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
                50% { transform: scale(1.05); box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3); }
                100% { transform: scale(1); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
            }
        `;

        const styleElement = document.createElement('style');
        styleElement.textContent = pulseAnimation;
        document.head.appendChild(styleElement);

        resultsButton.style.animation = 'pulse 2s infinite';

        // Enhanced click handler
        resultsButton.onclick = function() {
            console.log("Arrow button clicked - starting analysis");

            // Visual feedback
            resultsButton.style.animation = 'none';
            resultsButton.style.transform = 'scale(0.95)';
            setTimeout(() => {
                resultsButton.style.transform = 'scale(1)';
            }, 200);

            // Show the popup
            const popup = document.getElementById('popup');
            if (popup) {
                popup.style.display = 'flex';
            }

            // Run the analysis
            window.analyzePDFContent();

            return false;
        };

        // Add hover effect
        resultsButton.onmouseover = function() {
            resultsButton.style.transform = 'scale(1.1)';
        };

        resultsButton.onmouseout = function() {
            resultsButton.style.transform = 'scale(1)';
        };
    }

    const closeButton = document.getElementById('closeButton');
    if (closeButton) {
        closeButton.onclick = function() {
            console.log("Close button clicked");
            const popup = document.getElementById('popup');
            if (popup) {
                popup.style.display = 'none';
            }
            return false;
        };
    }

    const refreshButton = document.getElementById('refreshButton');
    if (refreshButton) {
        refreshButton.onclick = function() {
            console.log("Refresh button clicked");
            window.reAnalyzeAndShow();
            return false;
        };
    }

    // We don't automatically analyze on page load anymore
    // Instead, we wait for the user to click the Results button
    console.log("Waiting for user to click Results button to perform analysis");

    // Don't load initial analysis data automatically
    // We'll only load and display results when the user clicks the Results button
    console.log("Initial analysis data loading disabled - waiting for user to click Results button");
});

// Function to check title similarity
window.checkTitleSimilarity = async function(title, content) {
    try {
        const response = await fetch('/check_title_similarity', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: title,
                content: content  // Send the full content for context analysis
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data.similar_titles || [];
    } catch (error) {
        console.error('Error checking title similarity:', error);
        return [];
    }
}

// Function to check thesis context similarity
window.checkThesisSimilarity = async function(content) {
    try {
        // We've removed the context similarity section, so no need to show loading indicators
        console.log("Checking thesis similarity in the background...");

        // The backend will now always return exactly the top 3 theses
        const response = await fetch('/check_thesis_similarity', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                content: content,
                threshold: 0.0  // No threshold needed as we'll always take top 3
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Thesis similarity results:", data);
        return data.similar_theses || [];
    } catch (error) {
        console.error('Error checking thesis context similarity:', error);
        return [];
    }
}

// Function to display context similarity results
window.displayContextSimilarity = function(similarTheses) {
    console.log("Displaying context similarity results");

    // Get the similar titles section and list
    const similarTitlesSection = document.getElementById('similarTitlesSection');
    const similarTitlesList = document.getElementById('similarTitlesList');

    if (!similarTitlesSection || !similarTitlesList) {
        console.error("Could not find similar titles section or list elements");
        return;
    }

    // Clear the current list
    similarTitlesList.innerHTML = '';

    if (similarTheses && similarTheses.length > 0) {
        console.log(`Found ${similarTheses.length} contextually similar theses`);

        // Show the section
        similarTitlesSection.style.display = 'block';

        // Get the top 3 theses (or fewer if less are available)
        const thesesToShow = similarTheses.slice(0, 3);

        // Add each thesis to the list
        thesesToShow.forEach((thesis, index) => {
            console.log(`Displaying thesis ${index+1}: "${thesis.title}" (${thesis.context_similarity}% similar)`);

            const titleItem = document.createElement('div');
            titleItem.className = 'similar-title-item';

            // Create title header with similarity score
            const titleHeader = document.createElement('div');
            titleHeader.className = 'similar-title-header';

            // Title text
            const titleText = document.createElement('div');
            titleText.textContent = thesis.title;
            titleHeader.appendChild(titleText);

            // Similarity score badge with more precise percentage
            const scoreSpan = document.createElement('span');
            scoreSpan.className = 'similar-title-score';
            // Format the similarity score to always show 2 decimal places
            const formattedSimilarity = typeof thesis.context_similarity === 'number'
                ? thesis.context_similarity.toFixed(2)
                : thesis.context_similarity;
            scoreSpan.textContent = `${formattedSimilarity}% similar`;
            titleHeader.appendChild(scoreSpan);

            titleItem.appendChild(titleHeader);

            // Check if we have a similar paragraph to display
            if (thesis.similar_paragraph) {
                const paragraphSection = document.createElement('div');
                paragraphSection.className = 'similar-paragraph';

                // Add header for the paragraph
                const paragraphHeader = document.createElement('div');
                paragraphHeader.className = 'similar-paragraph-header';
                paragraphHeader.textContent = 'Similar content from your technical description:';
                paragraphSection.appendChild(paragraphHeader);

                // Add the paragraph text
                const paragraphText = document.createElement('div');
                paragraphText.textContent = thesis.similar_paragraph;
                paragraphSection.appendChild(paragraphText);

                titleItem.appendChild(paragraphSection);
            }

            // Display a representative sentence from the abstract
            if (thesis.similar_abstract_sentence) {
                const abstractSection = document.createElement('div');
                abstractSection.className = 'similar-abstract';

                // Add header for the abstract sentence
                const abstractHeader = document.createElement('div');
                abstractHeader.className = 'similar-abstract-header';
                abstractHeader.textContent = 'From the thesis abstract:';
                abstractSection.appendChild(abstractHeader);

                // Add the abstract sentence text
                const abstractText = document.createElement('div');
                abstractText.textContent = thesis.similar_abstract_sentence;
                abstractSection.appendChild(abstractText);

                titleItem.appendChild(abstractSection);
            }

            similarTitlesList.appendChild(titleItem);
        });
    } else {
        console.log("No context similarity results found");
        similarTitlesSection.style.display = 'block';

        // Create a message to inform the user
        const noResultsMessage = document.createElement('div');
        noResultsMessage.className = 'similar-title-item';
        noResultsMessage.style.textAlign = 'center';
        noResultsMessage.style.padding = '15px';
        noResultsMessage.innerHTML = `
            <div style="color: #666; margin-bottom: 10px;">
                <i class="fas fa-info-circle" style="color: #51087e; margin-right: 5px;"></i>
                No similar theses found
            </div>
            <div style="font-size: 0.9em; color: #888;">
                We couldn't find any contextually similar theses in our database.
            </div>
        `;

        similarTitlesList.appendChild(noResultsMessage);
    }
}

// Update the updateAnalysisDisplay function to handle similar titles
window.updateAnalysisDisplay = function(results) {
    console.log("⭐⭐⭐ GLOBAL updateAnalysisDisplay FUNCTION CALLED ⭐⭐⭐");
    console.log("Results:", results);

    // Debug speech accuracy specifically
    console.log("Speech accuracy value:", results.speech_accuracy);

    // We've removed the context similarity section as requested by the user
    // No need to check for context similarity when the results button is clicked

    if (!results) {
        console.error("No results to display");
        return;
    }

    // Update speech similarity
    const speechProgressBar = document.getElementById('progressSpeech');
    const speechLabel = document.getElementById('speechSimilarityLabel');
    if (speechProgressBar && speechLabel) {
        if (results.speech_similarity !== undefined && results.speech_similarity !== null) {
            // We have a valid similarity score
            const similarityValue = Math.round(results.speech_similarity);
            speechProgressBar.style.width = `${similarityValue}%`;
            speechProgressBar.style.display = 'block';
            speechProgressBar.style.backgroundColor = '#d8b4f8';
            speechLabel.textContent = `${similarityValue}%`;
            console.log(`Updated speech similarity to ${similarityValue}%`);
        } else {
            // No similarity score available (likely due to insufficient words)
            speechProgressBar.style.width = '0%';
            speechProgressBar.style.display = 'block';
            speechProgressBar.style.backgroundColor = '#d8b4f8';
            speechLabel.innerHTML = `<span style="font-size: 0.8em; color: #666;">Insufficient words (min. 50 required)</span>`;
            console.log('Speech similarity not available - insufficient words');
        }
    }

    // Update thesis similarity
    const thesisProgressBar = document.getElementById('progressThesis');
    const thesisLabel = document.getElementById('thesisSimilarityLabel');
    if (thesisProgressBar && thesisLabel && results.thesis_similarity !== undefined) {
        const thesisSimilarityValue = Math.round(results.thesis_similarity);
        thesisProgressBar.style.width = `${thesisSimilarityValue}%`;
        thesisProgressBar.style.display = 'block';
        thesisProgressBar.style.backgroundColor = '#d8b4f8';
        thesisLabel.textContent = `${thesisSimilarityValue}%`;
        console.log(`Updated thesis similarity to ${thesisSimilarityValue}%`);
    }

    // Update similar titles
    const similarTitlesSection = document.getElementById('similarTitlesSection');
    const similarTitlesList = document.getElementById('similarTitlesList');

    if (similarTitlesSection && similarTitlesList) {
        similarTitlesList.innerHTML = '';

        if (results.similar_titles_by_abstract && results.similar_titles_by_abstract.length > 0) {
            similarTitlesSection.style.display = 'block';

            results.similar_titles_by_abstract.forEach(item => {
                const titleItem = document.createElement('div');
                titleItem.className = 'similar-title-item';

                // Create title header with similarity score
                const titleHeader = document.createElement('div');
                titleHeader.className = 'similar-title-header';

                // Title text
                const titleText = document.createElement('div');
                titleText.textContent = item.title;
                titleHeader.appendChild(titleText);

                // Similarity score badge with more precise percentage
                const scoreSpan = document.createElement('span');
                scoreSpan.className = 'similar-title-score';
                // Format the similarity score to always show 2 decimal places
                const formattedSimilarity = typeof item.similarity === 'number'
                    ? item.similarity.toFixed(2)
                    : item.similarity;
                scoreSpan.textContent = `${formattedSimilarity}% similar`;
                titleHeader.appendChild(scoreSpan);

                titleItem.appendChild(titleHeader);

                // Check if we have a similar paragraph to display
                if (item.similar_paragraph) {
                    const paragraphSection = document.createElement('div');
                    paragraphSection.className = 'similar-paragraph';

                    // Add header for the paragraph
                    const paragraphHeader = document.createElement('div');
                    paragraphHeader.className = 'similar-paragraph-header';
                    paragraphHeader.textContent = 'Similar content from your technical description:';
                    paragraphSection.appendChild(paragraphHeader);

                    // Add the paragraph text
                    const paragraphText = document.createElement('div');
                    paragraphText.textContent = item.similar_paragraph;
                    paragraphSection.appendChild(paragraphText);

                    titleItem.appendChild(paragraphSection);
                }

                // Display a representative sentence from the abstract
                if (item.similar_abstract_sentence) {
                    const abstractSection = document.createElement('div');
                    abstractSection.className = 'similar-abstract';

                    // Add header for the abstract sentence
                    const abstractHeader = document.createElement('div');
                    abstractHeader.className = 'similar-abstract-header';
                    abstractHeader.textContent = 'From the thesis abstract:';
                    abstractSection.appendChild(abstractHeader);

                    // Add the abstract sentence text
                    const abstractText = document.createElement('div');
                    abstractText.textContent = item.similar_abstract_sentence;
                    abstractSection.appendChild(abstractText);

                    titleItem.appendChild(abstractSection);
                }

                similarTitlesList.appendChild(titleItem);
            });

            console.log(`Updated similar titles list with ${results.similar_titles_by_abstract.length} items`);
        } else {
            console.log("No contextually similar thesis titles to display");
            similarTitlesSection.style.display = 'block';
            similarTitlesList.innerHTML = '';

            // Create a message to inform the user
            const noResultsMessage = document.createElement('div');
            noResultsMessage.className = 'similar-title-item';
            noResultsMessage.style.textAlign = 'center';
            noResultsMessage.style.padding = '15px';
            noResultsMessage.innerHTML = `
                <div style="color: #666; margin-bottom: 10px;">
                    <i class="fas fa-info-circle" style="color: #51087e; margin-right: 5px;"></i>
                    No similar theses found
                </div>
                <div style="font-size: 0.9em; color: #888;">
                    We couldn't find any contextually similar theses in our database.
                </div>
            `;

            similarTitlesList.appendChild(noResultsMessage);
        }
    }

    // Update discrepancies with more informative display in a single block
    const discrepancyList = document.getElementById('discrepancyList');
    if (discrepancyList && results.missed_keypoints && results.added_keypoints) {
        discrepancyList.innerHTML = '';

        // Create a single container for all discrepancies
        const discrepancyContainer = document.createElement('div');
        discrepancyContainer.className = 'discrepancy-container';

        // Add a header for the combined list
        const header = document.createElement('div');
        header.className = 'discrepancy-block-header';
        header.innerHTML = `<h6 style="margin: 0 0 10px 0; color: #51087e; font-family: 'Comfortaa', sans-serif;">Key differences between your document and speech:</h6>`;
        discrepancyContainer.appendChild(header);

        // Combine all discrepancies in a single list
        const allDiscrepancies = [];

        // Add missed keypoints
        results.missed_keypoints.forEach(point => {
            if (point.includes(':')) {
                const [keyword, context] = point.split(':', 2);
                allDiscrepancies.push({
                    type: 'missing',
                    icon: '🔻',
                    label: 'Missing in speech:',
                    keyword: keyword.trim(),
                    context: context.trim()
                });
            } else {
                allDiscrepancies.push({
                    type: 'missing',
                    icon: '🔻',
                    label: 'Missing in speech:',
                    keyword: point,
                    context: null
                });
            }
        });

        // Add extra keypoints
        results.added_keypoints.forEach(point => {
            if (point.includes(':')) {
                const [keyword, context] = point.split(':', 2);
                allDiscrepancies.push({
                    type: 'extra',
                    icon: '🔺',
                    label: 'Extra in speech:',
                    keyword: keyword.trim(),
                    context: context.trim()
                });
            } else {
                allDiscrepancies.push({
                    type: 'extra',
                    icon: '🔺',
                    label: 'Extra in speech:',
                    keyword: point,
                    context: null
                });
            }
        });

        // Create the combined list
        const ul = document.createElement('ul');
        ul.className = 'discrepancy-combined-list';
        ul.style.paddingLeft = '0';
        ul.style.listStyleType = 'none';
        ul.style.margin = '0';

        // Add all discrepancies to the list
        allDiscrepancies.forEach(item => {
            const li = document.createElement('li');
            li.className = 'discrepancy-item';

            if (item.context) {
                li.innerHTML = `
                    <div class="discrepancy-header">
                        <span class="discrepancy-icon">${item.icon}</span>
                        <strong class="discrepancy-type">${item.label}</strong>
                        <span class="discrepancy-keyword">${item.keyword}</span>
                    </div>
                    <div class="discrepancy-context">${item.context}</div>
                `;
            } else {
                li.innerHTML = `
                    <div class="discrepancy-header">
                        <span class="discrepancy-icon">${item.icon}</span>
                        <strong class="discrepancy-type">${item.label}</strong>
                        <span class="discrepancy-keyword">${item.keyword}</span>
                    </div>
                `;
            }

            ul.appendChild(li);
        });

        discrepancyContainer.appendChild(ul);
        discrepancyList.appendChild(discrepancyContainer);

        console.log("Updated discrepancies with enhanced single-block display");
    }

    // Update title recommendations
    const titleRecommendationsSection = document.getElementById('titleRecommendationsSection');
    if (titleRecommendationsSection && results.suggested_titles && results.suggested_titles.length > 0) {
        titleRecommendationsSection.innerHTML = '<h5>Title Recommendations:</h5>';

        // Different reasons for each title
        const reasons = [
            "This title highlights the main focus and technical aspects of your research.",
            "This title emphasizes the innovative approach and potential impact of your work.",
            "This title presents your research in a professional academic context with clear focus.",
            "This title showcases the problem-solving aspects and practical applications of your work.",
            "This title emphasizes the methodological contributions of your research."
        ];

        // Add titles to the section
        results.suggested_titles.forEach((title, index) => {
            const div = document.createElement('div');
            div.className = 'recommendation-item';
            div.innerHTML = `
                <div class="recommendation-text">${title}</div>
                <div class="recommendation-reason">
                    ${reasons[index % reasons.length]}
                </div>
            `;
            titleRecommendationsSection.appendChild(div);
        });

        console.log(`Added ${results.suggested_titles.length} title recommendations`);
    }

    console.log("Analysis display updated successfully");
};

// Global function to generate titles
window.generateTitles = async function() {
    console.log("⭐⭐⭐ GLOBAL generateTitles FUNCTION CALLED ⭐⭐⭐");

    try {
        // Get the extracted text
        const extractedText = document.querySelector('#extracted-text')?.innerText || "";
        console.log(`Extracted text length: ${extractedText.length}`);

        // Get the speech text
        const speechText = window.transcriptText || "";
        console.log(`Speech text length: ${speechText.length}`);

        // Show loading indicator
        const titleRecommendationsSection = document.getElementById('titleRecommendationsSection');
        if (titleRecommendationsSection) {
            titleRecommendationsSection.innerHTML = `
                <h5 style="color: #6a1b9a; font-weight: bold; font-family: 'Comfortaa', sans-serif;">
                    Title Recommendations:
                </h5>
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
                    padding: 20px;
                    border-radius: 16px;
                    box-shadow: 0 4px 20px rgba(128, 90, 213, 0.2);
                    margin-top: 10px;
                    font-family: 'Comfortaa', sans-serif;
                    color: #4a0072;
                ">
                    <div style="margin-right: 10px;">
                        <i class="fas fa-spinner fa-spin" style="font-size: 18px; color: #6a1b9a;"></i>
                    </div>
                    <div style="font-size: 15px; font-family: 'Comfortaa', sans-serif;">
                        Generating Title Recommendations...
                    </div>
                </div>
            `;
        }

        // Make the API call
        console.log("Making API call to /generate_title");
        const response = await fetch('/generate_title', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                extracted_text: extractedText,
                speech_text: speechText
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Parse the response
        const data = await response.json();
        console.log("Title generation results:", data);

        // Update the display
        if (titleRecommendationsSection && data.titles && data.titles.length > 0) {
            titleRecommendationsSection.innerHTML = '<h5>Title Recommendations:</h5>';

            // Different reasons for each title
            const reasons = [
                "This title highlights the main focus and technical aspects of your research.",
                "This title emphasizes the innovative approach and potential impact of your work.",
                "This title presents your research in a professional academic context with clear focus.",
                "This title showcases the problem-solving aspects and practical applications of your work.",
                "This title emphasizes the methodological contributions of your research."
            ];

            // Add titles to the section
            data.titles.forEach((title, index) => {
                const div = document.createElement('div');
                div.className = 'recommendation-item';
                div.innerHTML = `
                    <div class="recommendation-text">${title}</div>
                    <div class="recommendation-reason">
                        ${reasons[index % reasons.length]}
                    </div>
                `;
                titleRecommendationsSection.appendChild(div);
            });

            console.log(`Added ${data.titles.length} title recommendations`);
        } else {
            if (titleRecommendationsSection) {
                titleRecommendationsSection.innerHTML = `
                    <h5>Title Recommendations:</h5>
                    <div style="padding: 10px; color: #666;">
                        No title recommendations could be generated. Please try again.
                    </div>
                `;
            }
        }
        return data;
    } catch (error) {
        console.error("Error in generateTitles:", error);
        const titleRecommendationsSection = document.getElementById('titleRecommendationsSection');
        if (titleRecommendationsSection) {
            titleRecommendationsSection.innerHTML = `
                <h5>Title Recommendations:</h5>
                <div style="padding: 10px; color: #666;">
                    An error occurred while generating title recommendations. Please try again.
                </div>
            `;
        }
    }
};

// Global function to refresh the analysis - optimized for faster generation
window.reAnalyzeAndShow = async function() {
    console.log("⭐⭐⭐ GLOBAL reAnalyzeAndShow FUNCTION CALLED ⭐⭐⭐");

    // Show loading state
    const button = document.querySelector('.refresh-button');
    if (button) {
        button.classList.add('loading');
    }

    const cornerLoading = document.querySelector('.corner-loading');
    if (cornerLoading) {
        cornerLoading.style.display = 'flex';
    }

    // Show loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'refresh-loading-indicator';
    loadingDiv.style.position = 'fixed';
    loadingDiv.style.top = '50%';
    loadingDiv.style.left = '50%';
    loadingDiv.style.transform = 'translate(-50%, -50%)';
    loadingDiv.style.background = 'linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)';
    loadingDiv.style.color = '#4a0072';
    loadingDiv.style.padding = '25px 30px';
    loadingDiv.style.borderRadius = '20px';
    loadingDiv.style.boxShadow = '0 8px 30px rgba(128, 90, 213, 0.3)';
    loadingDiv.style.zIndex = '9999';
    loadingDiv.style.fontFamily = '"Comfortaa", sans-serif';
    loadingDiv.innerHTML = `
        <div style="text-align: center;">
            <div style="margin-bottom: 12px;">
                <i class="fas fa-spinner fa-spin" style="font-size: 26px; color: #6a1b9a;"></i>
            </div>
            <div>
                <strong style="font-size: 16px;">Analyzing your document...</strong><br>
                <small style="color: #4a0072;">Please wait while we process your content.</small>
            </div>
        </div>
    `;
    document.body.appendChild(loadingDiv);

    try {
        // Create a promise for each operation
        let analysisPromise = window.analyzePDFContent();
        let titlesPromise = null;
        let similarityPromise = null;

        // Wait for analysis to complete
        await analysisPromise;
        console.log("Analysis complete, starting title generation");

        // Generate titles after analysis is complete
        if (typeof window.generateTitles === 'function') {
            console.log("Calling generateTitles");
            titlesPromise = window.generateTitles();
            await titlesPromise;
        }

        console.log("Title generation complete, checking thesis similarity");

        // Check context similarity
        const extractedText = document.querySelector('#extracted-text')?.innerText || "";
        if (extractedText && typeof checkThesisSimilarity === 'function') {
            console.log("Checking thesis context similarity");
            similarityPromise = checkThesisSimilarity(extractedText);
            const similarTheses = await similarityPromise;

            if (similarTheses && typeof displayContextSimilarity === 'function') {
                console.log("Displaying context similarity results");
                displayContextSimilarity(similarTheses);
            }
        }

        console.log("Analysis refreshed successfully");
    } catch (error) {
        console.error("Error in reAnalyzeAndShow:", error);
        alert("An error occurred during analysis. Please try again.");
    } finally {
        // Hide loading state
        if (button) {
            button.classList.remove('loading');
        }

        if (cornerLoading) {
            cornerLoading.style.display = 'none';
        }

        // Remove loading indicator
        const loadingIndicator = document.getElementById('refresh-loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }

        // Also remove any other loading indicators that might be present
        const analysisLoadingIndicator = document.getElementById('analysis-loading-indicator');
        if (analysisLoadingIndicator) {
            analysisLoadingIndicator.remove();
        }

        const globalLoadingIndicator = document.getElementById('global-loading-indicator');
        if (globalLoadingIndicator) {
            globalLoadingIndicator.remove();
        }
    }
};

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log("⭐⭐⭐ GLOBAL ANALYSIS.JS LOADED ⭐⭐⭐");

    // Set up global variables
    window.transcriptText = window.transcriptText || "";

    // Set up arrow button handler with enhanced functionality
    const resultsButton = document.getElementById('resultsButton');
    if (resultsButton) {
        // Make the button more noticeable
        resultsButton.style.transition = 'all 0.3s ease';
        resultsButton.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';

        // Add a pulsing effect to draw attention
        const pulseAnimation = `
            @keyframes pulse {
                0% { transform: scale(1); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
                50% { transform: scale(1.05); box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3); }
                100% { transform: scale(1); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
            }
        `;

        const styleElement = document.createElement('style');
        styleElement.textContent = pulseAnimation;
        document.head.appendChild(styleElement);

        resultsButton.style.animation = 'pulse 2s infinite';

        // Enhanced click handler
        resultsButton.onclick = function() {
            console.log("Arrow button clicked - starting analysis");

            // Visual feedback
            resultsButton.style.animation = 'none';
            resultsButton.style.transform = 'scale(0.95)';
            setTimeout(() => {
                resultsButton.style.transform = 'scale(1)';
            }, 200);

            // Show the popup
            const popup = document.getElementById('popup');
            if (popup) {
                popup.style.display = 'flex';
            }

            // Run the analysis
            window.analyzePDFContent();

            return false;
        };

        // Add hover effect
        resultsButton.onmouseover = function() {
            resultsButton.style.transform = 'scale(1.1)';
        };

        resultsButton.onmouseout = function() {
            resultsButton.style.transform = 'scale(1)';
        };
    }

    const closeButton = document.getElementById('closeButton');
    if (closeButton) {
        closeButton.onclick = function() {
            console.log("Close button clicked");
            const popup = document.getElementById('popup');
            if (popup) {
                popup.style.display = 'none';
            }
            return false;
        };
    }

    const refreshButton = document.getElementById('refreshButton');
    if (refreshButton) {
        refreshButton.onclick = function() {
            console.log("Refresh button clicked");
            window.reAnalyzeAndShow();
            return false;
        };
    }

    // Don't load initial analysis data automatically
    // We'll only load and display results when the user clicks the Results button
    console.log("Initial analysis data loading disabled - waiting for user to click Results button");
});












