import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# New Pinecone client instance
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def upsert_embeddings(embeddings, metadatas=None):
    # embeddings: list of (id, vector) tuples
    # metadatas: list of dicts (optional)
    if metadatas:
        vectors = [(str(i), vec, metadatas[i]) for i, (id, vec) in enumerate(embeddings)]
    else:
        vectors = [(str(i), vec) for i, (id, vec) in enumerate(embeddings)]
    index.upsert(vectors) 