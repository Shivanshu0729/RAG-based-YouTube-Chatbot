RAG-Based YouTube Video QA System

A Retrieval-Augmented Generation (RAG) based application that enables users to interact with YouTube videos using natural language. The system extracts video transcripts, converts them into semantic embeddings, and generates context-aware answers using a Large Language Model.

Overview

This project allows users to:

Input any YouTube video URL
Ask questions about the video content
Receive accurate, context-aware answers
Receive responses in either English or Hindi

The system leverages modern NLP techniques including vector search, semantic retrieval, and LLM-based generation.

Features
Retrieval-Augmented Generation (RAG) pipeline
Multilingual support (English and Hindi)
Semantic search using FAISS vector database
Real-time question answering
Streamlit-based interactive UI
Efficient document chunking and embedding
Architecture
YouTube Video
      ↓
Transcript Extraction
      ↓
Text Chunking
      ↓
Embeddings (HuggingFace)
      ↓
FAISS Vector Store
      ↓
Retriever
      ↓
LLM (Groq)
      ↓
Generated Answer
Tech Stack
Python
LangChain
FAISS (Vector Database)
HuggingFace Embeddings (all-MiniLM-L6-v2)
Groq LLM (LLaMA 3.1)
Streamlit
YouTube Transcript API
Project Structure
project/
│
├── app.py
├── notebook.ipynb
├── .env
├── .gitignore
├── requirements.txt
└── README.md
Installation

Clone the repository:

git clone https://github.com/Shivanshu0729/RAG-based-YouTube-Chatbot
cd RAG-based-YouTube-Chatbot

Install dependencies:

pip install -r requirements.txt
Environment Setup

Create a .env file in the root directory:

GROQ_API_KEY=your_api_key_here

Ensure .env is included in .gitignore.

Running the Application
streamlit run app.py

Then open:

http://localhost:8501
Usage
Enter a YouTube video URL
Select output language (English or Hindi)
Ask a question related to the video
Receive an AI-generated answer
Key Components
1. Transcript Loader

Extracts captions using YouTubeTranscriptAPI via LangChain.

2. Text Processing

Splits transcript into chunks for embedding.

3. Embedding Model

Uses HuggingFace sentence-transformers for semantic encoding.

4. Vector Database

Stores embeddings using FAISS.

5. Retriever

Fetches relevant chunks based on user query.

6. LLM Integration

Uses Groq-hosted LLaMA model for answer generation.

Performance Considerations
Caching reduces recomputation
Optimized chunk size improves retrieval
Fast inference using Groq LLM
Limitations
Requires videos with available transcripts
Performance depends on transcript quality
Limited to supported caption languages
Future Improvements
Multi-query retrieval
Reranking using cross-encoders
Timestamp-based answers
Chat history support
Multi-video querying
Cloud deployment
Author

Shivanshu Gangwar
