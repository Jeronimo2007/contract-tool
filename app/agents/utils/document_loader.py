from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from app.agents.retriever.pinecone_client import upsert_embeddings

# Example function to load and process a document

def process_and_store_document(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    
    embedder = OpenAIEmbeddings()
    embeddings = [(str(i), embedder.embed_query(chunk)) for i, chunk in enumerate(chunks)]
    metadatas = [{"text": chunk} for chunk in chunks]
    upsert_embeddings(embeddings, metadatas)
    return len(chunks)

