import os
import warnings

# Suppress NVML and NumPy warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage
warnings.filterwarnings("ignore", category=UserWarning, message=".*falling back to type probe function.*")

import json
import re
import spacy
import shutil
from collections import Counter
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Initialize KeyBERT model
kw_model = KeyBERT()

# Step 1: Clean the text (remove page numbers, footnotes, etc.)
def clean_text(text):
    # Remove page numbers (e.g., "[1]", "(Page 12)")
    text = re.sub(r"\[\d+\]|\(Page \d+\)", "", text)
    
    # Remove footnotes (e.g., "^1", "Footnote 3:")
    text = re.sub(r"\^\d+|Footnote \d+:", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# Step 2: Preprocess the text into chunks using spaCy's sentence boundary detection
def preprocess_text(text):
    doc = nlp(text)
    chunks = [sent.text for sent in doc.sents]  # Split into sentences using spaCy
    return chunks

# Step 3: Extract keywords using a combination of KeyBERT and TF-IDF
def extract_keywords(text, max_keywords=5):
    # Use KeyBERT for high-quality embeddings
    keybert_keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english")
    
    # Use TF-IDF for ranking
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    sorted_indices = tfidf_scores.argsort()[::-1]
    tfidf_keywords = [feature_names[i] for i in sorted_indices[:max_keywords]]
    
    # Combine and deduplicate keywords
    combined_keywords = list(set([kw[0] for kw in keybert_keywords] + tfidf_keywords))
    return combined_keywords[:max_keywords]

# Step 4: Extract named entities using spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Step 5: Add metadata and structure the data
def create_chunks_with_metadata(chunks, source):
    structured_data = []
    for i, chunk in enumerate(chunks):
        keywords = extract_keywords(chunk)  # Extract keywords
        entities = extract_entities(chunk)  # Extract named entities
        structured_data.append({
            "text": chunk,
            "metadata": {
                "source": source,
                "chunk_id": i + 1,
                "keywords": keywords,  # Add extracted keywords
                "entities": entities  # Add extracted named entities
            }
        })
    return structured_data

# Step 6: Load existing knowledge base (if it exists)
def load_knowledge_base(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []

# Step 7: Save the updated knowledge base
def save_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(data)} chunks to {filename}")

# Step 8: Move processed files to the documents folder
def move_processed_files(source_dir, target_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith(".txt"):
            shutil.move(os.path.join(source_dir, filename), os.path.join(target_dir, filename))
            print(f"Moved {filename} to {target_dir}")

# Step 9: Process new documents and update the knowledge base
def update_knowledge_base(new_docs_dir, docs_dir, knowledge_base_file="knowledge_base.json"):
    # Load the existing knowledge base
    knowledge_base = load_knowledge_base(knowledge_base_file)
    
    # Track the highest chunk_id to avoid duplicates
    max_chunk_id = max([chunk["metadata"]["chunk_id"] for chunk in knowledge_base], default=0)
    
    # Process each document in the new documents directory
    for filename in os.listdir(new_docs_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(new_docs_dir, filename), "r") as f:
                document = f.read()
            # Clean the document
            document = clean_text(document)
            # Preprocess into chunks using spaCy
            chunks = preprocess_text(document)
            # Add metadata and structure the data
            structured_data = create_chunks_with_metadata(chunks, filename)
            # Update chunk_ids to avoid conflicts
            for chunk in structured_data:
                max_chunk_id += 1
                chunk["metadata"]["chunk_id"] = max_chunk_id
            # Add new chunks to the knowledge base
            knowledge_base.extend(structured_data)
    
    # Save the updated knowledge base
    save_to_json(knowledge_base, knowledge_base_file)
    
    # Move processed files to the documents folder
    move_processed_files(new_docs_dir, docs_dir)

# Main workflow
if __name__ == "__main__":
    # Directory containing new text files
    new_documents_directory = "newDocumentsFolder"
    
    # Directory to archive processed files
    documents_directory = "documentsFolder" 
    
    # Update the knowledge base with new documents
    update_knowledge_base(new_documents_directory, documents_directory)