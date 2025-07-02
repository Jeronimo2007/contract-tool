from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

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

