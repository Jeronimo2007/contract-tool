from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from app.agents.tools.document_tools import ingest_tool, summarize_tool
from PyPDF2 import PdfReader
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
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

tools = [ingest_tool, summarize_tool]


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




def __init__():

    events = graph.stream(
        {"messages": [{"role": "user", "content": f"ingest and summarize the next document: {pdf_text}"}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

__init__()

