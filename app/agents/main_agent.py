from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from ..agents.utils.document_loader import is_pdf_loaded, search_query
from langchain_core.tools import tool

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    pdf_loaded: bool




initial_message = {
    "messages": [SystemMessage(content="""You are a helpful assistant specialized in analyzing and summarizing documents. Your task is to answer any question about the document that has been pre-processed and stored as chunk IDs in the state.
    
   Answer only with the answer to question. """)]
}

load_dotenv()


tools = []

llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    openai_api_key=os.getenv("api_key"),
    openai_api_base=os.getenv("base_url"),
)



llm = llm.bind_tools(tools)

def chatbot(state: State):
    user_query = None
    for msg in state["messages"]:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_query = msg.get("content")
        elif hasattr(msg, 'content') and getattr(msg, 'role', None) == 'user':
            user_query = msg.content

    # Call the LLM
    response = llm.invoke(state["messages"])

    # Check if the response is a tool call
    tool_calls = getattr(response, "tool_calls", None) or response.additional_kwargs.get("tool_calls", None)
    if tool_calls:
        # For each tool call, execute the tool and return the result
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name") or tool_call.get("name")
            args = tool_call.get("function", {}).get("arguments") or tool_call.get("args")
            if tool_name == "document_search_tool":
                # args may be a JSON string or dict
                import json
                if isinstance(args, str):
                    args = json.loads(args)
                query = args.get("query")
                answer = document_search_tool(query)
                return {"messages": state["messages"] + [{"role": "assistant", "content": answer}]}
    # If no tool call, just return the LLM's response
    return {"messages": state["messages"] + [{"role": "assistant", "content": getattr(response, 'content', str(response))}]}


def chat_with_chunks(user_query: str):
    """
    Retrieves relevant chunks for the user query and passes them to the LLM for response.
    """
    chunks = search_query(user_query)
    if isinstance(chunks, list):
        chunk_texts = [chunk.page_content for chunk in chunks]
    else:
        chunk_texts = [getattr(chunks, 'page_content', str(chunks))]
    messages = [
        SystemMessage(content="You are a helpful assistant specialized in analyzing and summarizing documents. Use the following document context to answer the user's question."),
        {"role": "system", "content": "\n\n".join(chunk_texts)},
        {"role": "user", "content": user_query}
    ]
    response = llm.invoke(messages)
    return response

# Define a tool for document search
@tool
def document_search_tool(query: str) -> str:
    """Search the document for relevant information and answer the query."""
    return chat_with_chunks(query)

# Register the tool in the tools list
# If you want the agent to use this tool, add it to the tools list
# You may add more tools as needed

tools = [document_search_tool]

llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    openai_api_key=os.getenv("api_key"),
    openai_api_base=os.getenv("base_url"),
)

llm = llm.bind_tools(tools)


graph_builder = StateGraph(State)

tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,  
    {
        "use_tool": "tools",
        "end": END,
    }
)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()

# # --- Graph Visualization ---
# # Get the mermaid syntax from the compiled graph
# mermaid_code = graph.get_graph().draw_mermaid()

# # Render and save as PNG
# png_bytes = draw_mermaid_png(mermaid_code)
# with open("output.png", "wb") as f:
#     f.write(png_bytes)

if __name__ == "__main__":
    # Example user query
    user_query = "What is the main subject of the document?"
    # Prepare the initial state
    state = {
        "messages": [{"role": "user", "content": user_query}],
        "pdf_loaded": True  # or use is_pdf_loaded() if you want to check
    }
    # Run the chatbot
    result = chatbot(state)
    print("Agent response:", result["messages"][-1]["content"])
