import json
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama  # For using Llama3.2
import faiss

# Load FAISS index and metadata
index = faiss.read_index("pharma_vector_index.faiss")
with open("pharma_metadata.json", "r", encoding="utf-8") as f:
    doc_metadata = json.load(f)

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Llama (via Ollama or API)
llm = Ollama(model="llama3.2", temperature=0.1)

def retrieve_relevant_documents(query, top_k=3):
    """Retrieve relevant documents from the FAISS vector database."""
    query_embedding = embeddings.embed_query(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    # Match indices with metadata
    results = [
        {"content": doc_metadata[i], "distance": d}
        for i, d in zip(indices[0], distances[0])
    ]
    return results

def generate_response(query, retrieved_documents):
    """Generate a response using Llama3.2."""
    # Combine the query and retrieved documents into a single prompt
    context = "\n".join(
        [
            f"Document {i + 1}:\nSection: {doc['content']['section']}\nContent: {doc['content']['content']}"
            for i, doc in enumerate(retrieved_documents)
        ]
    )
    prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Generate response using the LLM
    response = llm(prompt)
    return response

def main():
    """Main loop for interacting with the RAG pipeline."""
    print("RAG Pipeline Ready!")
    print("Enter your query (type 'exit' to quit):")

    while True:
        query = input("\nYour Query: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        # Retrieve relevant documents
        retrieved_documents = retrieve_relevant_documents(query)
        if not retrieved_documents:
            print("No relevant documents found.")
            continue

        # Generate response
        response = generate_response(query, retrieved_documents)
        print(f"\nGenerated Response:\n{response}\n")

if __name__ == "__main__":
    main()
