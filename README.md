Mini-RAG
A lightweight, local RAG (Retrieval-Augmented Generation) system for rapid prototyping and testing.

Purpose
Mini-RAG is designed for developers who need to quickly set up a RAG system locally to:

Test different token size parameters

Experiment with semantic search parameters

Compare different LLM providers

Build a simple foundation for more complex RAG implementations such as a Chatbot with authentication, chatrooms, chat history persistence, more sophisticated query template etc.

All functionality is packed into just two files for maximum simplicity and ease of use.

Features
PDF Support: Currently supports PDF document ingestion
Local Processing: Runs entirely on your machine
Performance Monitoring: Built-in timers for pipeline performance analysis (Context retrieval time,LLM response time in seconds)
Flexible LLM Integration: Easy to swap between different API providers
Minimal Dependencies: Lightweight setup with essential libraries only


Quick Start

1. Installation
pip install -r requirements.txt

2. Setup
Add your PDF documents to the rag_files/ folder
Add your LLM API key in rag.py
Add your file path in pdf_to_chunks.py 

3. Generate Embeddings
python pdf_to_chunks.py
This creates your embeddings and text chunks for retrieval.

4. Run the Chatbot
python rag.py

Ask any questions about your documents!


File Structure
mini-rag/
├── rag.py              # Main RAG pipeline and chatbot
├── pdf_to_chunks.py    # Document processing and embedding generation
├── rag_files/          # Your PDF documents go here
├── requirements.txt    # Dependencies
└── README.md


Performance Monitoring
The system includes built-in timing for each pipeline stage:

Context retrieval time
Query formatting time
LLM response time
