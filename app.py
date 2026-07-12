import streamlit as st
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize client using the public Inference API
client = InferenceClient(token=HF_TOKEN)

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
st.markdown("Ask me anything about IIT Bombay academic documents!")

try:
    model, index, chunks_metadata = load_resources()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

query = st.text_input("Enter your question:")

if st.button("Ask") and query:
    with st.spinner("Searching institute documents..."):
        # 1. Search FAISS
        query_vector = model.encode([query]).astype('float32')
        k = 3
        _, indices = index.search(query_vector, k)
        
        retrieved_contexts = [chunks_metadata[idx]["text"] for idx in indices[0]]
        context_string = "\n\n".join(retrieved_contexts)
        
        # 2. Call LLM using the most stable supported model
        try:
            response = client.chat_completion(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful Academic Assistant. Answer based strictly on the context. If the answer is not in the context, output: 'I don't know based on the provided documents.' After each answer, you must provide the source for your information as well. In the last line, you must display the source like so: 'source: <link>'."
                    },
                    {
                        "role": "user", 
                        "content": f"Context:\n{context_string}\n\nQuestion: {query}"
                    }
                ],
                max_tokens=256,
                temperature=0.1
            )
            
            st.subheader("Answer:")
            st.write(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"Error calling LLM: {e}")
            st.info("If this persists, the free-tier API may be experiencing high traffic. Please try again later.")
