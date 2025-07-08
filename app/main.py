# app/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
import shutil
from pydantic import BaseModel

from app.ingestion import load_and_index
from app.rag_chain import get_rag_chain
import os

from dotenv import load_dotenv
load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is not set!"


app = FastAPI()
UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    load_and_index(file_path)

    return {"filename": file.filename, "status": "processed and indexed"}

class AskRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: AskRequest):
    chain = get_rag_chain()
    result = chain.run(request.query)
    return {"answer": result}
