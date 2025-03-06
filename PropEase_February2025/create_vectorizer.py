import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the preprocessed dataset
dataset_path = "C:/xampp/htdocs/PropEase-main/Project-main/preprocessed_dataset.csv"
df = pd.read_csv(dataset_path)

# Use the correct text column
text_column = "combined_text"  # Change this if needed
if text_column not in df.columns:
    raise ValueError(f"Column '{text_column}' not found in the dataset. Check column names: {df.columns}")

# Convert the column values to a list of strings
texts = df[text_column].astype(str).tolist()

# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(texts)

# Save the fitted vectorizer to a file
vectorizer_path = r"C:\xampp\htdocs\PropEase-main\Project-main\tfidf_vectorizer.pkl"
with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

print("Fitted vectorizer saved successfully.")

from sklearn.svm import SVC
import pickle

# Load dataset
df = pd.read_csv("C:/xampp/htdocs/PropEase-main/Project-main/preprocessed_dataset.csv")

# Transform using the new TF-IDF vectorizer
texts = df["combined_text"].astype(str).tolist()
X = vectorizer.transform(texts)

# Define labels (replace 'labels_column' with actual label column)
y = df["combined_text"]  # Update this with your actual labels

# Train a new SVM model
model = SVC()
model.fit(X, y)

# Save the new model
model_path = "C:/xampp/htdocs/PropEase-main/Project-main/svm_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("New SVM model trained and saved successfully.")
