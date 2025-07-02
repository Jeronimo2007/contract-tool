from langchain.tools import StructuredTool
from app.agents.utils.document_loader import process_and_store_document_local, SESSION_STORE
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model_name="google/gemini-2.0-flash-001",
    openai_api_key=os.getenv("api_key"),
    openai_api_base=os.getenv("base_url"),
)


class SummarizeDocumentTool(BaseModel):
    ids: Optional[List[str]] = None
    state: Optional[dict] = None

class IngestDocumentTool(BaseModel):
    text: str
    user_id: Optional[str] = None


def ingest_document_tool(text: str, user_id: Optional[str] = None) -> dict:
    """
    Tool for ingesting a document: splits, embeds, and stores in local session storage.
    Args:
        args (IngestDocumentTool): The document text to process.
    Returns:
        dict: Success message with number of chunks stored and chunk IDs.
    """
    num_chunks, chunk_ids = process_and_store_document_local(text, user_id=user_id)
    return {
        "messages": [{"role": "assistant", "content": f"Document ingested successfully with {num_chunks} chunks stored."}],
        "chunk_ids": chunk_ids
    }

def fetch_document_chunks_by_ids_local(ids):
    return [SESSION_STORE[cid] for cid in ids if cid in SESSION_STORE]

def sumarize_document_tool(ids: Optional[List[str]] = None, state: Optional[dict] = None) -> dict:
    """
    Tool for summarizing a document by fetching its chunks from local session storage by IDs.
    Args:
        ids (Optional[List[str]]): List of chunk IDs to fetch and summarize.
        state (Optional[dict]): State containing chunk IDs.
    Returns:
        str: Concatenated text of the document, ready for LLM summarization.
    """
    if ids is None and state is not None:
        ids = state.get("chunk_ids", [])
    chunks = fetch_document_chunks_by_ids_local(ids)
    document_text = " ".join(chunk.get('text', '') for chunk in chunks)

    summary = llm.invoke(f"Summarize the following document: {document_text}")
    return {"messages": [{"role": "assistant", "content": summary}]}


ingest_tool = StructuredTool(
    name="Ingest_Document",
    func=ingest_document_tool,
    description="Ingests, stores, or saves a document by splitting, embedding, and saving it in local session storage.",
    args_schema=IngestDocumentTool
)

summarize_tool = StructuredTool(
    name="Summarize_Document",
    func=sumarize_document_tool,
    description="Summarizes a document by fetching its chunks from local session storage by IDs.",
    args_schema=SummarizeDocumentTool
)