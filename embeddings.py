import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [e.embedding for e in response.data]