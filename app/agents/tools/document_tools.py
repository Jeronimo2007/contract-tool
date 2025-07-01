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
    return {"messages": [{"role": "assistant", "content": f"Document ingested successfully with {num_chunks} chunks stored."}]}

def sumarize_document_tool(text: str) -> str:
    """
    Tool for summarizing a document.
    Args:
        text (str): The document text to summarize.
    Returns:
        str: Summary of the document.
    """
    # Placeholder for summarization logic
    # In a real implementation, you would use an LLM or other method to summarize the text
    summary = f"Summary of the document: {text[:100]}..."  # Example summary
    return summary


ingest_tool = Tool(
    name="Ingest_Document",
    func=ingest_document_tool,
    description="Ingests, stores, or saves a document by splitting, embedding, and saving it in Pinecone."
)