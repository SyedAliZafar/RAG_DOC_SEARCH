# app/embedding.py

import os
from dotenv import load_dotenv
import openai

load_dotenv()  # 🔥 Load variables from .env

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    response = openai.Embedding.create(input=text, model=model)
    return response["data"][0]["embedding"]
