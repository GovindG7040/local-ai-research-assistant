# 🤖 Local AI Research Assistant

A Hybrid Conversational Retrieval-Augmented Generation (RAG) application built using **FastAPI**, **LangChain**, **ChromaDB**, **BM25**, **CrossEncoder Reranking**, **Ollama**, and **React**.

The assistant retrieves relevant knowledge from a local knowledge base and generates grounded responses with source attribution.

---

## ✨ Features

- Hybrid Search (Dense + BM25 Retrieval)
- History-Aware Retrieval
- CrossEncoder Document Reranking
- Conversational Memory
- Source Attribution
- Local LLM Inference using Ollama
- FastAPI REST Backend
- React-based Frontend
- Modular and Scalable Architecture

---

## 🛠️ Tech Stack

### Backend

- Python
- FastAPI
- LangChain (LCEL)
- ChromaDB
- HuggingFace Embeddings
- Sentence Transformers
- Ollama
- BM25 Retriever

### Frontend

- React
- Vite
- Axios
- CSS

---

## 📂 Project Structure

```text
local-ai-research-assistant/
│
├── backend/
│   ├── app/
│   ├── main.py
│
├── frontend/
│   ├── src/
│   └── public/
│
├── knowledge_base/
│
├── pyproject.toml
└── README.md
```

---

## 🚀 Getting Started

### Backend

```bash
cd backend

uv sync

uv run main.py
```

### Frontend

```bash
cd frontend

npm install

npm run dev
```

---

## 🔄 RAG Pipeline

```text
User Query
      │
      ▼
History-Aware Query Rewrite
      │
      ▼
Hybrid Retrieval
(Dense + BM25)
      │
      ▼
CrossEncoder Reranker
      │
      ▼
Prompt Construction
      │
      ▼
Ollama LLM
      │
      ▼
Answer + Sources
```

---

## 📌 Current Capabilities

- Retrieval-Augmented Generation (RAG)
- Conversational Search
- Hybrid Retrieval
- Document Reranking
- Local LLM Inference
- Source Attribution

---

## 🔮 Future Improvements

- Enhanced User Interface
- Streaming Responses
- Multi-session Chat
- PDF Upload Support
- Advanced Evaluation Metrics
- Authentication & User Management

---

## 📄 License

This project is intended for learning, research, and educational purposes.
