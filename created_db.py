import os
import json
import numpy as np
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize FAISS index and metadata storage
embedding_dim = 384  # Dimension of "all-MiniLM-L6-v2" embeddings
index = faiss.IndexFlatL2(embedding_dim)
doc_metadata = []  # To store metadata for each vector

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def preprocess_and_store(json_folder):
    """Load JSON files, preprocess content, and store in FAISS."""
    for idx, filename in enumerate(os.listdir(json_folder)):
        if filename.endswith(".json"):
            with open(os.path.join(json_folder, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                drug_name = filename.replace(".json", "")
                for section, content in data.items():
                    if isinstance(content, str):  # Only process text content
                        embedding = embeddings.embed_query(content)
                        index.add(np.array([embedding]))  # Add to FAISS index
                        doc_metadata.append({"drug_name": drug_name, "section": section, "content": content})
                        print(f"Added {drug_name} - {section}")

# Folder containing your JSON files
json_folder = "give_a_directory"
preprocess_and_store(json_folder)

# Save FAISS index and metadata
faiss.write_index(index, "pharma_vector_index.faiss")
with open("pharma_metadata.json", "w", encoding="utf-8") as f:
    json.dump(doc_metadata, f)

print("Dataset successfully indexed and stored in FAISS!")

