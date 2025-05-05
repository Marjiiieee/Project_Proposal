import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer, util
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from bert_score import score
from keybert import KeyBERT
import string


# === SIMILARITY MODEL EVALUATION ===

# Load evaluation data
df = pd.read_csv("C:/xampp/htdocs/PropEase-main/Project-main/similarity_eval_data.csv")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Extract true labels
y_true = [int(row['label']) for _, row in df.iterrows()]

# Initialize best scores
best_threshold = 0.0
best_f1 = 0.0
best_results = {}

print("🔍 Evaluating thresholds from 0.60 to 0.90...\n")

# Try different thresholds to find best F1
for t in [x / 100 for x in range(60, 91)]:
    y_pred = []
    
    for _, row in df.iterrows():
        emb1 = model.encode(row['text1'], convert_to_tensor=True)
        emb2 = model.encode(row['text2'], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        y_pred.append(1 if similarity >= t else 0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Threshold: {t:.2f} → Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t
        best_results = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        }

# Final best threshold and metrics
print("\n✅ Similarity Model:")
print(f"Threshold: {best_threshold:.2f}")
for metric, value in best_results.items():
    print(f"{metric}: {value:.4f}")


# === GRAMMAR CORRECTION MODEL EVALUATION ===
def evaluate_grammar_model():
    df = pd.read_csv("C:/xampp/htdocs/PropEase-main/Project-main/grammar_eval_data.csv")
    tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")

    y_true, y_pred = [], []

    for _, row in df.iterrows():
        input_text = "grammar: " + row['original']
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(input_ids)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        y_pred.append(prediction)
        y_true.append(row['corrected'])

    # Optional: exact match accuracy
    exact_match_acc = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)

    print("\n📝 Grammar Correction Model:")
    print("Exact Match Accuracy:", round(exact_match_acc, 4))
    # You may also use token-level or word-level matching here if needed



# === TITLE GENERATION EVALUATION ===
def evaluate_title_generation_model():
    df = pd.read_csv("C:/xampp/htdocs/PropEase-main/Project-main/title_eval_data.csv", encoding='latin1')
    generator = pipeline("text2text-generation", model="EngLip/flan-t5-sentence-generator")

    refs, preds = [], []

    for _, row in df.iterrows():
        prompt = f"Generate a title for this abstract: {row['abstract']}"
        result = generator(prompt, max_length=50)[0]['generated_text'].strip()
        refs.append(row['title'])
        preds.append(result)

    P, R, F1 = score(preds, refs, lang="en")
    print("\n🏷️ Title Generation Model:")
    print("BERTScore - Precision:", round(P.mean().item(), 4))
    print("BERTScore - Recall:", round(R.mean().item(), 4))
    print("BERTScore - F1:", round(F1.mean().item(), 4))



# === KEYWORD EXTRACTION EVALUATION ===
def clean_keywords(keyword_list):
    return set(
        kw.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        for kw in keyword_list if kw.strip()
    )

def evaluate_keyword_extraction_model(k=5):
    df = pd.read_csv("C:/xampp/htdocs/PropEase-main/Project-main/keyword_eval_data.csv", encoding='ISO-8859-1')
    kw_model = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")

    precisions, recalls, f1s = []

    for _, row in df.iterrows():
        gold = clean_keywords(row['keywords'].split(","))
        predicted = kw_model.extract_keywords(row['text'], top_n=k)
        pred_set = clean_keywords([kw for kw, _ in predicted])

        true_positives = len(gold.intersection(pred_set))
        precision = true_positives / k
        recall = true_positives / len(gold) if gold else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    print("\n🔑 Keyword Extraction Evaluation:")
    print("Average Precision@5:", round(sum(precisions) / len(precisions), 4))
    print("Average Recall@5:   ", round(sum(recalls) / len(recalls), 4))
    print("Average F1@5:       ", round(sum(f1s) / len(f1s), 4))


if __name__ == "__main__":
    evaluate_grammar_model()
    evaluate_title_generation_model()
    evaluate_keyword_extraction_model()
