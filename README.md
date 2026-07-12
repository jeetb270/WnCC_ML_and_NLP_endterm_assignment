# WnCC_ML_and_NLP_endterm_assignment
## IITB Insti-Assist: Academic Assistant

A Retrieval-Augmented Generation (RAG) powered AI assistant designed to answer questions about IIT Bombay's academic policies, grading, and course registration without hallucinating.

## Setup Instructions

1. Install dependencies:
   pip install -r requirements.txt

2. Set up your HuggingFace Token:
   Create a `.env` file in the root directory and add:
   HF_TOKEN=your_token_here

3. Add Data:
   Place at least 5 relevant IITB academic PDFs (like the rulebook or academic calendar) into the `data/` folder.

4. Run the Ingestion Pipeline:
   python ingest.py
   (This will chunk your PDFs and create a local FAISS vector database).

5. Run the Application:
   streamlit run app.py

## Project Details

Scope: Academic Assistant
Data Sources: 5 official IITB PDFs placed in the data folder.
Chunking Strategy: Fixed-size sliding window (500 characters with 50 character overlap) to ensure concepts aren't abruptly cut off between chunks while remaining small enough to fit within LLM context windows efficiently.
Limitations: Complex tables in PDFs might not parse perfectly using basic PyPDF2. Given more time, using OCR or advanced table extraction (like unstructured.io) would improve retrieval on exam schedule tables.
