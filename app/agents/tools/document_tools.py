
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model_name="google/gemini-2.0-flash-001",
    openai_api_key=os.getenv("api_key"),
    openai_api_base=os.getenv("base_url"),
)

