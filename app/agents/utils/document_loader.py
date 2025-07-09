
import os 
import uuid
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from typing import List


load_dotenv()




output_path = "app/pdf/"
file_path = "app/pdf/3.pdf"




chunks = partition_pdf(
    filename= file_path,
    infer_table_structure=True,
    strategy="hi_res",


    extract_image_block_types=["Image"],


    extract_image_block_to_payload=True,

    chunking_strategy="by_title",
    max_characters = 10000,
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000
)

tables = []
texts = []

for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)

    if "CompositeElement" in str(type(chunk)):
        texts.append(chunk)




# elements = chunks[2].metadata.orig_elements

# chunk_images = [el for el in elements if 'Image' in str(type(el))]

# chunkimg=chunk_images[0].to_dict()

# print(chunkimg)




def get_images_base64(chunks):
    images_base64 = []

    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_base64.append(el.metadata.image_base64)
    return images_base64


images = get_images_base64(chunks)







initial_message = """
    You are a helpful assistant specialized in analyzing and summarizing documents.
    
   Answer only with the answer to question.Do not answer with 'here is the summary' or anything like that.
   Just give the summary as it is.
   
   Table or text to chunk: {element}"""

prompt = ChatPromptTemplate.from_template(initial_message)


llm = ChatOpenAI(
    model="mistralai/mistral-small-3.2-24b-instruct:free",
    openai_api_key=os.getenv("api_key"),
    openai_api_base=os.getenv("base_url"),
)

llm_img = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    openai_api_key=os.getenv("api_key"),
    openai_api_base=os.getenv("base_url"),
)

chain = prompt | llm | StrOutputParser()


text_summaries =  chain.batch(texts,{"max_concurrency": 3})

tables_html = [table.metadata.text_as_html for table in tables]
table_summaries = chain.batch(tables_html, {"max_concurrency": 3})


initial_message_img = """

    Describe the iamge in detail. For context, the image is part of a pdf document to analyze"""


messages = [
    (
        "user",
        [
            {"type": "text", "text": initial_message_img},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image}"}
            }
        ]
    )
]

prompt_img = ChatPromptTemplate.from_messages(messages)

chain_img = prompt_img | llm_img | StrOutputParser()

image_summaries = chain_img.batch(images, {"max_concurrency": 3})


vectore_store = Chroma(collection_name = "Multi_moda_agent", embedding_function=OpenAIEmbeddings())


store = InMemoryStore()
id_key = 'doc_id'

retriever = MultiVectorRetriever(
    vectorstore = vectore_store,
    docstore = store,
    id_key = id_key,
)


doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables if any exist
if tables:
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))
else:
    print("No tables found in the PDF. Skipping table indexing.")

# Add images if any exist
if images:
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, images)))
else:
    print("No images found in the PDF. Skipping image indexing.")


new = retriever.invoke(
    "What is langGraph?"
)

