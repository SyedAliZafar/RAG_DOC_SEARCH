# 📄 RAG Document Search

A Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **LangChain**, and **OpenAI**, supporting document ingestion, embedding, and intelligent Q&A over your own files.

## 🏗️ Architecture Overview

This project uses **both FastAPI and Streamlit**, serving different roles:

- **FastAPI** is the backend API framework that handles:
  - Document ingestion and indexing
  - Vectorstore management with FAISS
  - Exposing API endpoints (including Swagger UI at `/docs`)
  - Running the core logic for Retrieval-Augmented Generation (RAG)

- **Streamlit** provides a user-friendly frontend interface for interacting with the system:
  - Upload documents directly through the web UI
  - Ask questions and receive answers based on your documents
  - Display chat history and responses interactively

### Running the apps

- **Backend:** FastAPI runs the backend server, but **you don’t need to start it manually** if you’re using the Streamlit frontend, because the Streamlit app directly calls the shared application logic in Python without relying on the FastAPI server.

- **Frontend:** You only need to run the Streamlit app to use the full application UI and functionality:

```bash
streamlit run streamlit_app.py

---

## 🖼️ App Interface

![RAG Search UI](images/userinterface.png)

---

## ✨ Features

- 📄 Upload `.txt` documents
- 🧠 Embed and index content using `LangChain` + `OpenAIEmbeddings`
- 📦 Vector search powered by `FAISS`
- 🤖 Ask questions and get intelligent, context-aware answers from your data
- ⚙️ CI/CD with GitHub Actions
- 🧪 Built-in linting, formatting, and testing

---

## 🧰 Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [OpenAI API](https://platform.openai.com/)
- GitHub Actions for CI
- Pytest, Black, Isort, MyPy for testing and formatting

---

## 🛠️ Development Setup

### 📥 Clone the Repository

```bash
git clone https://github.com/SyedAliZafar/RAG_DOC_SEARCH.git
cd RAG_DOC_SEARCH



📁 Project Structure

RAG_DOC_SEARCH/
├── app/
│   ├── main.py
│   ├── ingestion.py
│   ├── rag_chain.py
│   └── embedding.py (this is not required by the application)
├── tests/
│   ├── test_health.py
│   ├── test_ingestion.py
│   └── test_rag_chain.py
├── images/
│   └── userinterface.PNG
├── requirements.txt
├── requirements-dev.txt
├── .github/workflows/ci.yml
└── README.md



📦 Install Dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt


🔑 Set Your OpenAI API Key
Either in your shell:

export OPENAI_API_KEY=your_key_here

Or by creating a .env file and loading it:

OPENAI_API_KEY=your_key_here

🧪 Run Locally
uvicorn app.main:app --reload

Open your browser and visit:
👉 http://localhost:8000/docs — for FastAPI's Swagger UI


🐍 Requirements
Python 3.9+

OpenAI API Key


🧪 Run Tests
pytest tests/


🧹 Formatting & Linting
Run all checks:

black --check app tests
isort --check-only app tests
flake8 app tests
mypy app


🔄 CI/CD
GitHub Actions runs on every push to main:

✅ Black formatting check

✅ Pytest unit tests

✅ Uses OPENAI_API_KEY via GitHub secrets [https://github.com/SyedAliZafar/RAG_DOC_SEARCH/settings/secrets/actions]

✅ Future Improvements

Add Streamlit-based frontend

Add PDF document support

Add mocking to tests for CI compatibility

Docker support for deployment