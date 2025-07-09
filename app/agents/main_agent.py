from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from app.agents.tools.document_tools import ingest_document_tool, summarize_tool, question_answer_tool
from PyPDF2 import PdfReader
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage
from PyPDF2 import PdfReader
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    chunk_ids: list 


initial_message = {
    "messages": [SystemMessage(content="""You are a helpful assistant specialized in analyzing and summarizing documents. Your task is to answer any question about the document that has been pre-processed and stored as chunk IDs in the state.
    
   Answer only with the answer to question. """)]
}

load_dotenv()







llm = init_chat_model(
    model="google/gemini-2.0-flash-001",
    openai_api_key=os.getenv("api_key"),
    openai_api_base=os.getenv("base_url"),
)

