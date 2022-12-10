import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from pprint import pprint
from datetime import datetime
from notion import NotionPageReader

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NOTION_API_KEY = os.getenv('NOTION_API_KEY')

notion_client = NotionPageReader(integration_token=NOTION_API_KEY)
def load_notion_documents():
    pages = notion_client.search("")
    documents = notion_client.load_data(pages=pages)
    return documents

pprint(load_notion_documents())
