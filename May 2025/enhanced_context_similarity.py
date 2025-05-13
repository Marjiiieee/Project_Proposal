"""
Enhanced Context Similarity Module for PropEase

This module provides improved context similarity checking between concept papers
and dataset abstracts. It extracts both technical descriptions and introductions
from concept papers and compares them with dataset abstracts using advanced
semantic similarity models.
"""

import re
import os
import pandas as pd
import time
from sentence_transformers import util

def extract_introduction(content):
    """
    Extract the introduction section from a concept paper.

    Args:
        content (str): The full text content of the concept paper

    Returns:
        str: The extracted introduction section or None if not found
    """
    intro_patterns = [
        r'(?i)introduction.*?(?=\n\s*\n\s*[A-Z]|\Z)',
        r'(?i)background.*?(?=\n\s*\n\s*[A-Z]|\Z)',
        r'(?i)overview.*?(?=\n\s*\n\s*[A-Z]|\Z)',
        r'(?i)problem\s+statement.*?(?=\n\s*\n\s*[A-Z]|\Z)',
        r'(?i)project\s+description.*?(?=\n\s*\n\s*[A-Z]|\Z)',
    ]

    for pattern in intro_patterns:
        matches = re.search(pattern, content, re.DOTALL)
        if matches:
            introduction = matches.group(0).strip()
            print(f"Found introduction section ({len(introduction)} chars)")
            return introduction

    # If no introduction found, try to extract the first few paragraphs
    paragraphs = content.split('\n\n')
    if len(paragraphs) >= 2:
        intro = '\n\n'.join(paragraphs[:2])
        print(f"Using first two paragraphs as introduction ({len(intro)} chars)")
        return intro

    return None

def calculate_context_similarity(tech_desc_embedding, abstract_embedding):
    """
    Calculate the context similarity between a technical description and an abstract.

    This function computes a more accurate similarity score between the technical
    description of a concept paper and the abstract of a thesis paper.

    The similarity is calculated using cosine similarity between the embeddings,
    which measures the cosine of the angle between the two vectors, providing
    a more accurate measure of semantic similarity.

    Args:
        tech_desc_embedding: The embedding of the technical description
        abstract_embedding: The embedding of the abstract

    Returns:
        float: The similarity score as a percentage (0-100) with higher precision
    """
    # Calculate raw cosine similarity with full precision
    raw_similarity = float(util.pytorch_cos_sim(
        tech_desc_embedding.reshape(1, -1),
        abstract_embedding.reshape(1, -1)
    )[0][0] * 100)

    # Return the raw similarity without scaling to ensure accurate scoring
    # This provides a more accurate representation of the actual similarity
    # and allows the 20% threshold to be more meaningful
    return raw_similarity

def extract_key_sentences(text, max_sentences=3):
    """
    Extract the most representative sentences from a text.

    This function attempts to find the most informative sentences that best
    represent the content, rather than just taking the first few sentences.
    It looks for sentences with technical terms and higher information density.

    Args:
        text (str): The text to extract sentences from
        max_sentences (int): Maximum number of sentences to extract

    Returns:
        str: The extracted sentences
    """
    if not text:
        return ""

    # More robust sentence splitting
    # First try to use regex to split on sentence boundaries
    import re
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_endings, text)

    # If that didn't work well (e.g., we got only one sentence), fall back to simple splitting
    if len(sentences) <= 1:
        sentences = text.split('.')
        # Add periods back to sentences except the last one if it's empty
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

    # If we still have very few sentences, just return what we have
    if len(sentences) <= max_sentences:
        key_sentences = ' '.join(sentences)
        if key_sentences and not key_sentences.endswith('.'):
            key_sentences += '.'
        return key_sentences

    # Technical terms that might indicate important sentences
    tech_terms = [
        "system", "algorithm", "method", "approach", "framework", "model",
        "implementation", "architecture", "design", "solution", "technology",
        "application", "development", "process", "technique", "analysis",
        "evaluation", "performance", "results", "data", "interface", "module",
        "component", "function", "feature", "capability", "requirement"
    ]

    # Score sentences based on length and presence of technical terms
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        # Skip very short sentences
        if len(sentence.split()) < 5:
            continue

        # Calculate a score based on:
        # 1. Position in text (earlier sentences often more important)
        # 2. Presence of technical terms
        # 3. Sentence length (not too short, not too long)
        position_score = 1.0 if i < len(sentences) // 3 else 0.5

        # Count technical terms
        term_count = sum(1 for term in tech_terms if term.lower() in sentence.lower())
        term_score = min(term_count / 2, 1.0)  # Cap at 1.0

        # Length score - prefer sentences between 10 and 25 words
        words = len(sentence.split())
        length_score = 0.0
        if 10 <= words <= 25:
            length_score = 1.0
        elif words < 10:
            length_score = words / 10
        else:  # words > 25
            length_score = max(0.0, 1.0 - (words - 25) / 25)

        # Combined score
        total_score = position_score * 0.3 + term_score * 0.5 + length_score * 0.2

        scored_sentences.append((sentence, total_score))

    # Sort by score (highest first)
    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    # Take the top max_sentences
    selected_sentences = [s[0] for s in scored_sentences[:max_sentences]]

    # Sort back by original position to maintain flow
    original_order = []
    for sentence in selected_sentences:
        try:
            idx = sentences.index(sentence)
            original_order.append((idx, sentence))
        except ValueError:
            # If not found (shouldn't happen), append to end
            original_order.append((len(sentences), sentence))

    original_order.sort(key=lambda x: x[0])
    ordered_sentences = [s[1] for s in original_order]

    # Join the sentences
    key_sentences = ' '.join(ordered_sentences)

    # Make sure the result ends with a period
    if key_sentences and not key_sentences.endswith('.'):
        key_sentences += '.'

    return key_sentences

def extract_representative_sentence(abstract, tech_desc_embedding=None, similarity_model=None):
    """
    Extract the most representative sentence from the abstract.

    If tech_desc_embedding and similarity_model are provided, this function
    will find the sentence in the abstract that is most similar to the
    technical description. Otherwise, it will select the most informative
    sentence based on technical content.

    Args:
        abstract (str): The abstract text
        tech_desc_embedding: Optional embedding of the technical description
        similarity_model: Optional model to use for similarity calculation

    Returns:
        str: The most representative sentence from the abstract
    """
    if not abstract:
        return ""

    # Split abstract into sentences
    import re
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_endings, abstract)

    # If that didn't work well, fall back to simple splitting
    if len(sentences) <= 1:
        sentences = abstract.split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]

    # Skip very short sentences
    meaningful_sentences = [s for s in sentences if len(s.split()) >= 5]

    # If no meaningful sentences, use whatever we have
    if not meaningful_sentences and sentences:
        meaningful_sentences = sentences

    # If we still have no sentences, return a portion of the abstract
    if not meaningful_sentences:
        if len(abstract) > 100:
            return abstract[:100] + "..."
        return abstract

    # If we have tech_desc_embedding and similarity_model, find the most similar sentence
    if tech_desc_embedding is not None and similarity_model is not None and len(meaningful_sentences) > 1:
        try:
            # Encode all sentences
            sentence_embeddings = similarity_model.encode(meaningful_sentences, convert_to_tensor=True)

            # Calculate similarity with technical description
            similarities = []
            for i, sentence_embedding in enumerate(sentence_embeddings):
                similarity = float(util.pytorch_cos_sim(
                    tech_desc_embedding.reshape(1, -1),
                    sentence_embedding.reshape(1, -1)
                )[0][0])
                similarities.append((i, similarity))

            # Get the most similar sentence
            most_similar_idx = max(similarities, key=lambda x: x[1])[0]
            return meaningful_sentences[most_similar_idx]
        except Exception as e:
            print(f"Error finding most similar sentence: {e}")
            # Fall back to scoring method

    # If we can't use similarity or there was an error, score sentences based on technical content
    tech_terms = [
        "system", "algorithm", "method", "approach", "framework", "model",
        "implementation", "architecture", "design", "solution", "technology",
        "application", "development", "process", "technique", "analysis"
    ]

    # Score sentences based on technical terms and position
    scored_sentences = []
    for i, sentence in enumerate(meaningful_sentences):
        # Position score - earlier sentences often more important in abstracts
        position_score = 1.0 if i == 0 else (0.9 if i == 1 else 0.8)

        # Technical term score
        term_count = sum(1 for term in tech_terms if term.lower() in sentence.lower())
        term_score = min(term_count / 2, 1.0)  # Cap at 1.0

        # Length score - prefer sentences between 10 and 30 words
        words = len(sentence.split())
        length_score = 0.0
        if 10 <= words <= 30:
            length_score = 1.0
        elif words < 10:
            length_score = words / 10
        else:  # words > 30
            length_score = max(0.0, 1.0 - (words - 30) / 30)

        # Combined score
        total_score = position_score * 0.4 + term_score * 0.4 + length_score * 0.2
        scored_sentences.append((sentence, total_score))

    # Get the highest scoring sentence
    if scored_sentences:
        return max(scored_sentences, key=lambda x: x[1])[0]

    # Fallback to first sentence
    return meaningful_sentences[0]

def check_thesis_similarity_route_handler(data, custom_similarity_model, context_sim_model,
                                         sentence_model, extract_technical_description,
                                         load_thesis_dataset_with_embeddings, BASE_PATH):
    """
    Handler function for the check_thesis_similarity route.
    Optimized for faster performance and reduced processing time.

    Args:
        data (dict): The request data
        custom_similarity_model: The custom similarity model
        context_sim_model: The context similarity model
        sentence_model: The sentence transformer model
        extract_technical_description: Function to extract technical description
        load_thesis_dataset_with_embeddings: Function to load thesis dataset
        BASE_PATH: The base path for the application

    Returns:
        dict: The response data
    """
    print("=== STARTING ENHANCED THESIS SIMILARITY CHECK (OPTIMIZED) ===")

    # Get the content (concept paper content)
    content = data.get('content', '').strip()

    # Early validation to avoid unnecessary processing
    if not content:
        print("Error: No content provided")
        return {
            'status': 'error',
            'message': 'Content is required for context similarity'
        }, 400

    # Extract technical description - this is the most important part for similarity
    technical_description = extract_technical_description(content)
    print(f"Extracted technical description ({len(technical_description)} chars)")

    # Use context similarity model if available, otherwise use sentence_model
    similarity_model = context_sim_model if context_sim_model is not None else sentence_model
    print(f"Using model: {type(similarity_model).__name__}")

    # Encode the technical description
    print("Encoding technical description...")
    tech_desc_embedding = similarity_model.encode(technical_description, convert_to_tensor=True)

    # Load thesis dataset
    print("Loading thesis dataset...")
    df, cached_embeddings = load_thesis_dataset_with_embeddings(custom_similarity_model)

    # Handle case where dataset is not available
    if df is None or df.empty:
        print("Dataset is empty or not available")
        return {
            'status': 'success',
            'similar_theses': [],
            'similarity_stats': {
                'min': 0,
                'max': 0,
                'avg': 0,
                'count': 0
            }
        }

    # Get title column
    title_column = None
    for col in df.columns:
        if col.lower() == 'title':
            title_column = col
            break

    if not title_column:
        title_column = 'title'  # Default

    # Performance optimization: Use cached embeddings if available
    if cached_embeddings is not None:
        print("Using cached embeddings for faster processing")

        # Calculate similarities between technical description and all thesis embeddings
        print("Calculating similarities with cached embeddings...")
        similarities = util.cos_sim(tech_desc_embedding.reshape(1, -1), cached_embeddings)[0]

        # Convert to percentages without scaling to ensure accurate scoring
        all_similarities = [float(score * 100) for score in similarities]

        # Create list of theses with similarity scores
        all_theses_with_similarity = []

        # Get the indices of the top N theses (we only need top 3)
        max_results = 3
        top_indices = sorted(range(len(all_similarities)), key=lambda i: all_similarities[i], reverse=True)[:max_results*3]  # Get more than needed for filtering

        # Process only the top theses
        for idx in top_indices:
            if idx >= len(df):
                continue

            thesis_title = str(df.iloc[idx][title_column])
            context_similarity = all_similarities[idx]

            # We'll collect all similarities and sort later, not skipping any here
            # This ensures we always get the top 3 most similar theses regardless of threshold

            # Get abstract
            abstract = str(df.iloc[idx].get('abstract', ''))
            if not abstract or pd.isna(abstract):
                abstract = "Abstract not available for this thesis."

            # Extract key sentences and representative sentence
            key_abstract = extract_key_sentences(abstract)
            # Use the tech_desc_embedding to find the most similar sentence in the abstract
            representative_abstract_sentence = extract_representative_sentence(abstract, tech_desc_embedding, similarity_model)

            # Find the most similar paragraph from the technical description
            # Only do this for the top candidates to save processing time
            most_similar_paragraph = ""
            highest_paragraph_similarity = 0

            # Split technical description into paragraphs
            tech_desc_paragraphs = technical_description.split('\n\n')
            tech_desc_paragraphs = [p.strip() for p in tech_desc_paragraphs if len(p.strip()) >= 30]  # Only keep substantial paragraphs

            # If we have very few paragraphs, try splitting by single newlines
            if len(tech_desc_paragraphs) <= 2:
                tech_desc_paragraphs = technical_description.split('\n')
                tech_desc_paragraphs = [p.strip() for p in tech_desc_paragraphs if len(p.strip()) >= 30]

            # Process if we have paragraphs, regardless of similarity threshold
            # This ensures we always find the most similar paragraph for the top candidates
            if tech_desc_paragraphs:
                try:
                    # Encode the abstract
                    abstract_embedding = similarity_model.encode(abstract, convert_to_tensor=True)
                    abstract_embedding_tensor = abstract_embedding.reshape(1, -1)

                    # Encode all paragraphs at once for better performance
                    paragraph_embeddings = similarity_model.encode(tech_desc_paragraphs, convert_to_tensor=True)

                    # Calculate similarity for each paragraph
                    paragraph_similarities = []
                    for j, paragraph_embedding in enumerate(paragraph_embeddings):
                        # Calculate paragraph similarity with higher precision
                        para_similarity = float(util.pytorch_cos_sim(
                            paragraph_embedding.reshape(1, -1),
                            abstract_embedding_tensor
                        )[0][0] * 100)

                        # Store with full precision for sorting
                        paragraph_similarities.append((j, para_similarity, tech_desc_paragraphs[j]))

                    # Find the paragraph with the highest similarity
                    if paragraph_similarities:
                        # Sort by similarity (highest first)
                        paragraph_similarities.sort(key=lambda x: x[1], reverse=True)
                        highest_paragraph_similarity = paragraph_similarities[0][1]
                        most_similar_paragraph = paragraph_similarities[0][2]
                except Exception as e:
                    print(f"Error finding similar paragraph: {e}")

            # Add to results
            all_theses_with_similarity.append({
                'title': thesis_title,
                'context_similarity': round(context_similarity, 2),  # Increased precision to 2 decimal places
                'abstract': abstract[:500] + '...' if len(abstract) > 500 else abstract,
                'key_abstract': key_abstract if key_abstract else (abstract[:200] + '...' if len(abstract) > 200 else abstract),
                'similar_paragraph': most_similar_paragraph,
                'paragraph_similarity': round(highest_paragraph_similarity, 2),  # Increased precision to 2 decimal places
                'similar_abstract_sentence': representative_abstract_sentence,
                'abstract_sentence_similarity': round(context_similarity, 2)  # Increased precision to 2 decimal places
            })
    else:
        # Fallback to processing without cached embeddings
        print("No cached embeddings available, processing abstracts individually...")

        # Prepare valid abstracts
        valid_abstracts = []
        valid_titles = []

        for _, row in df.iterrows():
            if 'abstract' not in row or not row['abstract'] or pd.isna(row['abstract']):
                continue

            abstract = str(row['abstract'])
            thesis_title = str(row[title_column])

            # Skip very short abstracts
            if len(abstract.strip()) < 10:
                continue

            valid_abstracts.append(abstract)
            valid_titles.append(thesis_title)

        # Limit the number of abstracts to process for better performance
        max_abstracts_to_process = 500
        if len(valid_abstracts) > max_abstracts_to_process:
            print(f"Sampling {max_abstracts_to_process} abstracts from {len(valid_abstracts)} for faster processing")
            import random
            random.seed(42)
            indices = random.sample(range(len(valid_abstracts)), max_abstracts_to_process)
            valid_abstracts = [valid_abstracts[i] for i in indices]
            valid_titles = [valid_titles[i] for i in indices]

        # Batch encode abstracts
        print(f"Batch encoding {len(valid_abstracts)} abstracts...")
        batch_size = 200
        all_abstract_embeddings = []
        all_similarities = []
        all_theses_with_similarity = []

        for i in range(0, len(valid_abstracts), batch_size):
            batch_abstracts = valid_abstracts[i:i+batch_size]
            batch_titles = valid_titles[i:i+batch_size]

            # Encode batch
            batch_embeddings = similarity_model.encode(batch_abstracts, convert_to_tensor=True)

            # Calculate similarities for this batch
            for j, (abstract, thesis_title, abstract_embedding) in enumerate(zip(batch_abstracts, batch_titles, batch_embeddings)):
                # Calculate context similarity
                context_similarity = calculate_context_similarity(tech_desc_embedding, abstract_embedding)
                all_similarities.append(context_similarity)

                # We'll collect all similarities and sort later, not skipping any here
                # This ensures we always get the top 3 most similar theses regardless of threshold

                # Extract key sentences
                key_abstract = extract_key_sentences(abstract)
                # Use the tech_desc_embedding to find the most similar sentence in the abstract
                representative_abstract_sentence = extract_representative_sentence(abstract, tech_desc_embedding, similarity_model)

                # Add to results without paragraph similarity (for performance)
                all_theses_with_similarity.append({
                    'title': thesis_title,
                    'context_similarity': round(context_similarity, 2),  # Increased precision to 2 decimal places
                    'abstract': abstract[:500] + '...' if len(abstract) > 500 else abstract,
                    'key_abstract': key_abstract if key_abstract else (abstract[:200] + '...' if len(abstract) > 200 else abstract),
                    'similar_paragraph': "",  # Skip paragraph similarity for performance
                    'paragraph_similarity': 0,
                    'similar_abstract_sentence': representative_abstract_sentence,
                    'abstract_sentence_similarity': round(context_similarity, 2)  # Increased precision to 2 decimal places
                })

    # Sort all theses by context similarity (highest first)
    all_theses_with_similarity.sort(key=lambda x: x['context_similarity'], reverse=True)

    # Always take exactly the top 3 theses (or fewer if less are available)
    max_results = 3
    similar_theses = all_theses_with_similarity[:min(max_results, len(all_theses_with_similarity))]

    # Log the results with more precise percentages
    if len(similar_theses) > 0:
        print(f"Found {len(similar_theses)} similar theses to display")
        for i, thesis in enumerate(similar_theses):
            print(f"{i+1}. {thesis['title']} - Similarity: {thesis['context_similarity']:.2f}%")
    else:
        print("No similar theses found")

    # Calculate similarity statistics if we have similarities
    if all_similarities:
        min_sim = min(all_similarities)
        max_sim = max(all_similarities)
        avg_sim = sum(all_similarities)/len(all_similarities)
        print(f"Similarity statistics: min={min_sim:.2f}%, max={max_sim:.2f}%, avg={avg_sim:.2f}%")
    else:
        min_sim = max_sim = avg_sim = 0
        print("No similarities calculated")

    return {
        'status': 'success',
        'similar_theses': similar_theses,
        'similarity_stats': {
            'min': round(min_sim, 2) if all_similarities else 0,  # Increased precision to 2 decimal places
            'max': round(max_sim, 2) if all_similarities else 0,  # Increased precision to 2 decimal places
            'avg': round(avg_sim, 2) if all_similarities else 0,  # Increased precision to 2 decimal places
            'count': len(all_similarities)
        }
    }
