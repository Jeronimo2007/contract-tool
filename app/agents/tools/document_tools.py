from langchain.tools import StructuredTool
from app.agents.utils.document_loader import process_and_store_document_local, SESSION_STORE, fetch_relevant_chunks
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
    ids: List[str]
    state: Optional[dict] = None

class IngestDocumentTool(BaseModel):
    text: str
    user_id: Optional[str] = None

class QA_Tool(BaseModel):
    question: str
    ids: List[str]


def ingest_document_tool(text: str, user_id: Optional[str] = None) -> dict:
    """
    Tool for ingesting a document: splits, embeds, and stores in local session storage.
    Args:
        text (str): The document text to process.
        user_id (Optional[str]): User ID for the document.
    Returns:
        dict: Response with ingestion details and chunk_ids for state update.
    """

    if not text:
        return {
            "messages": [{"role": "assistant", "content": "No text provided for ingestion."}],
            "chunk_ids": []
        }

    num_chunks, chunk_ids = process_and_store_document_local(text, user_id=user_id)
    return {
        "messages": [{"role": "assistant", "content": f"Document ingested successfully with {num_chunks} chunks stored."}],
        "chunk_ids": chunk_ids
    }

def fetch_document_chunks_by_ids_local(ids):
    return [SESSION_STORE[cid] for cid in ids if cid in SESSION_STORE]

def summarize_document_tool(ids: List[str], state: Optional[dict] = None) -> dict:
    """
    Tool for summarizing a document by fetching its chunks from local session storage by IDs.
    Args:
        ids (List[str]): List of chunk IDs to fetch and summarize.
        state (Optional[dict]): State containing chunk IDs (not used when ids are provided).
    Returns:
        dict: Response with summary and processing details.
    """
    if not ids:
        return {
            "messages": [{"role": "assistant", "content": "No document chunks found to summarize. Please provide chunk IDs."}]
        }
    
    chunks = fetch_document_chunks_by_ids_local(ids)
    
    if not chunks:
        return {
            "messages": [{"role": "assistant", "content": f"No document chunks found with the provided IDs: {ids}. Please check if the document has been properly ingested."}]
        }
    
    document_text = " ".join(chunk.get('text', '') for chunk in chunks)

    summary_response = llm.invoke(f"Summarize the following document: {document_text}")
    summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
    return {
        "messages": [{"role": "assistant", "content": summary}]
    }

def question_answer_tool(question: str, ids: List[str]) -> dict:
    """
    Tool for answering a question based on a given list of chunk ids. Fetches the context from those chunk ids.
    Args:
        question (str): The question to answer.
        ids (List[str]): List of chunk IDs to fetch context from.
    Returns:
        dict: Response with answer and context details.
    """
    if not ids:
        return {
            "messages": [{"role": "assistant", "content": "No document chunks provided. Please provide chunk IDs to answer questions about the document."}]
        }
    
    chunks = fetch_document_chunks_by_ids_local(ids)
    
    if not chunks:
        return {
            "messages": [{"role": "assistant", "content": f"No document chunks found with the provided IDs: {ids}. Please check if the document has been properly ingested."}]
        }
    
    context = "\n".join(chunk.get("text", "") for chunk in chunks)
    
    answer_response = llm.invoke(f"Answer the following question: {question} based on the following context: {context}")
    answer = answer_response.content if hasattr(answer_response, 'content') else str(answer_response)
    
    return {
        "messages": [{"role": "assistant", "content": answer}]
    }




ingest_tool = StructuredTool(
    name="ingest_document",
    func=ingest_document_tool,
    description="Ingests a document by splitting, embedding, and storing it in local session storage. Use this tool when a user provides a document to analyze.",
    args_schema=IngestDocumentTool
)

summarize_tool = StructuredTool(
    name="summarize_document",
    func=summarize_document_tool,
    description="Summarizes a document by fetching its chunks from local session storage using the provided chunk IDs. Use this tool to get a summary of an ingested document.",
    args_schema=SummarizeDocumentTool
)

question_answer_tool = StructuredTool(
    name="question_answer",
    func=question_answer_tool,
    description="Answers questions based on an ingested document using the provided chunk IDs. Use this tool to ask specific questions about the document content.",
    args_schema=QA_Tool
)