import os
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Configuration
DATA_DIR = "data"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunks_metadata.pkl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def get_pdf_text(filepath):
    text = ""
    reader = PdfReader(filepath)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def chunk_text(text, filename):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append({"text": chunk, "source": filename})
        start += (CHUNK_SIZE - CHUNK_OVERLAP)
    return chunks

def main():
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_chunks = []
    
    print("Reading and chunking PDFs...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR} directory. Please add PDFs and run again.")
        return

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            text = get_pdf_text(filepath)
            file_chunks = chunk_text(text, filename)
            all_chunks.extend(file_chunks)
            print(f"Processed {filename}: {len(file_chunks)} chunks.")

    if not all_chunks:
        print("No chunks created. Make sure you have PDFs in the data folder.")
        return

    print("Generating embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(all_chunks, f)
        
    print("Ingestion complete! Index and metadata saved.")

if __name__ == "__main__":
    main()
