from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import numpy as np

# In-memory store for document chunks and embeddings
SESSION_STORE = {}
doc_id_counter = 0

def process_and_store_document_local(text, user_id=None, chunk_size=500, chunk_overlap=50):
    global doc_id_counter
    doc_id_counter += 1
    doc_id = str(doc_id_counter)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    embedder = OpenAIEmbeddings()
    chunk_ids = [f"{user_id or 'anon'}_{doc_id}_{i}" for i in range(len(chunks))]
    embeddings = [embedder.embed_query(chunk) for chunk in chunks]
    # Store in session
    for i, chunk_id in enumerate(chunk_ids):
        SESSION_STORE[chunk_id] = {
            "embedding": embeddings[i],
            "text": chunks[i],
            "user_id": user_id,
            "doc_id": doc_id,
            "chunk_index": i
        }
    return len(chunks), chunk_ids

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def fetch_relevant_chunks(query: str, top_k: int = 3):
    """
    Given a query string, embed it and return the top_k most similar chunks from SESSION_STORE.
    Returns a list of dicts with 'text' and 'score'.
    """
    embedder = OpenAIEmbeddings()
    query_embedding = embedder.embed_query(query)
    scored_chunks = []
    for chunk in SESSION_STORE.values():
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append({"text": chunk["text"], "score": score})
    # Sort by score descending
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]

