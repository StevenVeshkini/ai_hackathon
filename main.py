import os
import click
import faiss
import pickle
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from notion import NotionPageReader
from gpt_index.schema import Document
from pprint import pprint

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NOTION_API_KEY = os.getenv('NOTION_API_KEY')

def load_notion_documents() -> List[Document]:
    print("Retrieving Notion documents...")
    notion_client = NotionPageReader(integration_token=NOTION_API_KEY)
    pages = notion_client.search("")
    documents = notion_client.load_data(pages=pages)
    return documents

def preprocess_documents(documents: Document) -> Tuple[List[List[str]], Dict[str, Any]]:
    print("Processing documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs = []
    metadatas = []
    for document in documents:
        content = document.text
        if not content:
            continue
        metadata = document.extra_info
        text_splits = text_splitter.split_text(content)
        docs.extend(text_splits)
        metadatas.extend([metadata] * len(text_splits))  
    return docs, metadatas

def create_embeddings_and_vector_db(documents: List[List[str]], metadatas: Dict[str, Any]) -> None:
    print("Creating embeddings of documents and loading into vector db...")
    vector_db = FAISS.from_texts(documents, OpenAIEmbeddings(), metadatas=metadatas)
    faiss.write_index(vector_db.index, "docs.index")
    vector_db.index = None
    with open("db.pkl", "wb") as f:
        pickle.dump(vector_db, f)

@click.command()
def ingest(): 
    # 1. Retrieve documents and metadata from source (Notion API)
    documents = load_notion_documents()

    # 1.5 Pre-process documents and document metadata
    # Split documents into smaller chunks due to the context limits of the LLMs.
    docs, metadatas = preprocess_documents(documents)
    print(docs)
    print(metadatas)
    # 2. Create embeddings of documents 
    # 3. Store embeddings in vector database
    create_embeddings_and_vector_db(docs, metadatas)

@click.command()
@click.argument('question')
def question(question: str) -> None:
    index = faiss.read_index("docs.index")
    with open("db.pkl", "rb") as f:
        vector_db = pickle.load(f)

    vector_db.index = index
    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=vector_db)
    result = chain({"question": question})
    pprint(f"Answer: {result['answer']}")
    pprint(f"Sources: {result['sources']}")

if __name__ == '__main__':
    question()