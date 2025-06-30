from langchain.tools import Tool
from app.agents.utils.document_loader import process_and_store_document

def ingest_document_tool(text: str) -> str:
    """
    Tool for ingesting a document: splits, embeds, and stores in Pinecone.
    Args:
        text (str): The document text to process.
    Returns:
        str: Success message with number of chunks stored.
    """
    num_chunks = process_and_store_document(text)
    return f"Document ingested and {num_chunks} chunks stored in Pinecone."

ingest_tool = Tool(
    name="Ingest Document",
    func=ingest_document_tool,
    description="Splits, embeds, and stores a document in Pinecone."
)