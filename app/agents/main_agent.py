from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from app.agents.tools.document_tools import ingest_tool


load_dotenv()


llm = ChatOpenAI()



agent = initialize_agent(
    tools=[ingest_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example usage: agent receives a task to ingest a document
result = agent.run("Ingest this document: 'This is a sample contract text to be embedded and stored.'")
print(result)

from PyPDF2 import PdfReader

# Path to your PDF
pdf_path = os.path.join(os.path.dirname(__file__), "../contrato_falso_test.pdf")

# Extract text from PDF
reader = PdfReader(pdf_path)
pdf_text = ""
for page in reader.pages:
    pdf_text += page.extract_text() or ""

# Now use the agent to ingest the PDF text
result = agent.run(f"Ingest this document: '{pdf_text}'")
print(result)