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




memory = MemorySaver()

graph_builder = StateGraph(State)

initial_message = {
    "messages": [SystemMessage(content="""You are a helpful assistant specialized in analyzing and summarizing documents. Your task is to answer any question about the document that has been pre-processed and stored as chunk IDs in the state.
    
    IMPORTANT: The document has already been ingested and is available as chunk IDs in the state. You do NOT need to ingest it again.
    
    CRITICAL: When using tools, you MUST pass the chunk_ids from the state. The chunk_ids are stored in the state and contain the actual document chunks.
    
    Use the available tools:
    - summarize_document: Pass the chunk_ids from state as the 'ids' parameter
    - question_answer: Pass the chunk_ids from state as the 'ids' parameter and provide the question
    
    Example: If the state contains chunk_ids: ['anon_1_0', 'anon_1_1'], use those exact IDs when calling tools.
    
    If the user asks for a document and no chunk IDs are available in the state, respond with "I do not have a document to analyze. Please provide a document first."
    """)]
}

load_dotenv()



def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}



# Path to your PDF
pdf_path = os.path.join(os.path.dirname(__file__), "../contrato_falso_test.pdf")

# Extract text from PDF
reader = PdfReader(pdf_path)
pdf_text = ""
for page in reader.pages:
    pdf_text += page.extract_text() or ""

tools = [summarize_tool, question_answer_tool]


llm = init_chat_model(
    model="gpt-4.1-mini-2025-04-14",
    openai_api_key=os.getenv("api_key"),
    openai_api_base=os.getenv("base_url"),

)

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition )
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

chunks = ingest_document_tool(pdf_text)['chunk_ids']
print(f"Generated chunk IDs: {chunks}")


def __init__():


    state = {
        "messages": initial_message["messages"] + [{"role": "user", "content": "Please summarize the document and then answer: What is this document about? Is it a contract?"}],
        "chunk_ids": chunks
    }

    events = graph.stream(
        state,
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

__init__()

