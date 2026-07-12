import streamlit as st
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize HuggingFace Client using a fast, capable instruct model
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN)

# App Configuration
st.set_page_config(page_title="IITB Academic Assistant", page_icon="🎓")
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunks_metadata.pkl"

@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        chunks = pickle.load(f)
    return model, index, chunks

st.title("🎓 IITB Academic Assistant")
st.markdown("Ask me anything about IIT Bombay course registration, grading policies, academic calendars, or exam rules!")

try:
    model, index, chunks_metadata = load_resources()
except Exception as e:
    # 🚨 Modified error handler to reveal the true error
    st.error(f"Failed to load resources. The exact error is: {e}")
    st.stop()

query = st.text_input("Enter your question:")

if st.button("Ask") and query:
    with st.spinner("Searching institute documents..."):
        # 1. Embed query and search FAISS
        query_vector = model.encode([query]).astype('float32')
        k = 3 # Retrieve top 3 chunks
        distances, indices = index.search(query_vector, k)
        
        retrieved_contexts = []
        sources = []
        for idx in indices[0]:
            chunk = chunks_metadata[idx]
            retrieved_contexts.append(chunk["text"])
            sources.append(chunk["source"])
            
        context_string = "\n\n".join(retrieved_contexts)
        
        # 2. Construct strict prompt
        prompt = f"""<|system|>
You are a helpful Academic Assistant for IIT Bombay students. 
Answer the user's question based strictly on the provided context. 
If the answer is not in the context, you must output exactly: "I don't know based on the provided documents."
Do not guess or use outside knowledge.
Context:
{context_string}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""
        # 3. Call LLM via HuggingFace
        try:
            response = client.text_generation(prompt, max_new_tokens=256, temperature=0.1)
            
            # 4. Display results
            st.subheader("Answer:")
            st.write(response)
            
            st.subheader("Sources used:")
            unique_sources = list(set(sources))
            for source in unique_sources:
                st.write(f"- {source}")
                
            with st.expander("View Retrieved Context Chunks"):
                for i, ctx in enumerate(retrieved_contexts):
                    st.write(f"**Chunk {i+1} from {sources[i]}:**")
                    st.write(ctx)
                    st.markdown("---")
                    
        except Exception as e:
            st.error(f"Error calling LLM: {e}")
