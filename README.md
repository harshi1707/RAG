# AI PDF Assistant — RAG Chatbot using Groq, LangChain & Streamlit

An intelligent Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions in natural language. The chatbot retrieves relevant content from uploaded documents using vector embeddings and generates accurate answers using Groq LLMs.

---

# Features

* Upload multiple PDF files
* AI-powered question answering
* Semantic document search using FAISS
* Fast inference with Groq LLMs
* HuggingFace embeddings support
* Interactive Streamlit frontend
* Modern professional UI
* Adjustable chunk size and overlap
* Retrieved document chunk viewer
* Real-time response generation

---

# Tech Stack

| Technology            | Purpose         |
| --------------------- | --------------- |
| Streamlit             | Frontend UI     |
| LangChain             | RAG pipeline    |
| Groq API              | LLM inference   |
| FAISS                 | Vector database |
| HuggingFace           | Embeddings      |
| Sentence Transformers | Semantic search |
| Python                | Backend logic   |

---

# Project Structure

```bash id="y1r6mf"
AI-PDF-Assistant/
│
├── Chatbot.py
├── requirements.txt
├── .env
├── README.md
└── assets/
```

---

# Installation

## Clone Repository

```bash id="t8v4zy"
git clone https://github.com/your-username/AI-PDF-Assistant.git

cd AI-PDF-Assistant
```

---

## Create Virtual Environment

### Windows

```bash id="e6m3bk"
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash id="p5k9tx"
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash id="r2n7cw"
pip install -r requirements.txt
```

Or manually install:

```bash id="z4q8sd"
pip install streamlit
pip install langchain==0.2.16
pip install langchain-core==0.2.38
pip install langchain-community==0.2.16
pip install langchain-groq==0.1.9
pip install langchain-text-splitters==0.2.4
pip install faiss-cpu
pip install sentence-transformers
pip install huggingface-hub
pip install pypdf
pip install python-dotenv
```

---

# Environment Variables

Create a `.env` file in the project root directory.

```env id="f7d1qh"
GROQ_API_KEY=your_groq_api_key
```
Get your Groq API key from:

https://console.groq.com/keys
---

# Run the Application

```bash id="m8w2js"
streamlit run Chatbot.py
```

---

# How It Works

## Step 1 — Upload PDFs

Users upload one or more PDF files through the Streamlit interface.

## Step 2 — Text Extraction

PDF content is extracted using `PyPDFLoader`.

## Step 3 — Text Chunking

Documents are split into smaller chunks using `RecursiveCharacterTextSplitter`.

## Step 4 — Embedding Generation

Semantic embeddings are generated using HuggingFace sentence transformers.

## Step 5 — Vector Storage

Embeddings are stored in a FAISS vector database.

## Step 6 — Retrieval-Augmented Generation

Relevant document chunks are retrieved and passed to the Groq LLM to generate answers.

---

# Application Features

* Drag-and-drop PDF upload
* Interactive dashboard
* Fast AI responses
* Retrieved chunk viewer
* Professional dark UI
* AI research assistant

---

# Use Cases

* Research paper Q&A
* Resume analysis
* Academic assistant
* Legal document search
* Business knowledge base
* PDF chatbot applications

---

# Future Enhancements

* Multi-format document support
* Persistent vector databases
* Authentication system
* Conversation memory
* Cloud deployment
* Voice assistant integration

---

# Author

Developed using:

* LangChain
* Groq
* FAISS
* Streamlit
* HuggingFace

---

# Support

If you found this project useful, consider giving the repository a star on GitHub.
