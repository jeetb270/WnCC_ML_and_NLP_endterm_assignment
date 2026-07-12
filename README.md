# WnCC_ML_and_NLP_endterm_assignment
## IITB Insti-Assist: Academic Assistant

A Retrieval-Augmented Generation (RAG) powered AI assistant designed to answer questions about IIT Bombay's academic policies, grading, and course registration without hallucinating.

Chosen Scope: Academic assistant
Reason: From a personal perspective, an academic assistant would've been the most helpful for me considering how often I found myself searching through fragmented PDFs and institute emails just to clarify basic examination rules.

## Setup Instructions

1. Install dependencies:
   '''powershell
   pip install -r requirements.txt
   '''

3. Set up your HuggingFace Token:
   Create a `.env` file in the root directory and add:
   HF_TOKEN=your_token_here
   (A token has already been added for the sake of this assignment, but in case it expires, you may make your new token (permission: read) and paste it in place of the initial token

4. Run the Ingestion Pipeline:
   '''python ingest.py'''
   (This will chunk the data PDFs and create a local FAISS vector database).

5. Run the Application:
   '''streamlit run app.py'''

## Chunking Strategy
For this project, I implemented a **Fixed-Size Chunking strategy** with an overlap. The documents are processed into text segments of approximately 500 characters, with a 50-character overlap between consecutive chunks.

### Why this strategy was chosen
*   **Context Preservation**: The 50-character overlap ensures that semantic context is not lost at the boundaries of the chunks. This prevents critical information (like a deadline or a policy exception) from being split awkwardly between two segments.
*   **Search Accuracy**: By maintaining consistent, granular chunk sizes, the vector search (FAISS) can return highly specific document passages. This ensures that the context provided to the LLM is relevant to the user's specific query rather than bloated with irrelevant institute-wide data.
*   **Efficiency**: Given the constraints of the free-tier API, keeping the context window compact while maximizing information density is essential for reliable, low-latency performance.


