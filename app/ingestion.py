import os
import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

INDEX_PATH = "vector_db"

_vectorstore = None  # internal memory

def save_vectorstore(vs):
    vs.save_local(INDEX_PATH)

def load_vectorstore():
    if os.path.exists(INDEX_PATH):
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True  # Add this flag here
        )
    return None



def set_vectorstore(vs):
    global _vectorstore
    _vectorstore = vs

def get_vectorstore():
    return _vectorstore

def select_loader(file_path: Path):
    if file_path.suffix == ".pdf":
        return PyPDFLoader(str(file_path))
    elif file_path.suffix == ".txt":
        return TextLoader(str(file_path), encoding="utf-8")
    elif file_path.suffix == ".md":
        return UnstructuredMarkdownLoader(str(file_path))
    elif file_path.suffix == ".docx":
        return Docx2txtLoader(str(file_path))
    else:
        return UnstructuredFileLoader(str(file_path))  # fallback
    

def load_and_index(file_path: Path):
    loader = select_loader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vs = get_vectorstore() or load_vectorstore()
    if vs is None:
        vs = FAISS.from_documents(chunks, embeddings)
    else:
        vs.add_documents(chunks)

    set_vectorstore(vs)
    save_vectorstore(vs)

