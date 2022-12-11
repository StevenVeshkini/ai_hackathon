import os
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
import argparse
import gradio as gr
import slack_sdk
from flask import Flask
from slackeventsapi import SlackEventAdapter
from threading import Thread

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NOTION_API_KEY = os.getenv('NOTION_API_KEY')
SLACK_TOKEN=os.getenv('SLACK_TOKEN')
SIGNING_SECRET=os.getenv('SIGNING_SECRET')

def load_notion_documents() -> List[Document]:
    print("Retrieving Notion documents...")
    notion_client = NotionPageReader(integration_token=NOTION_API_KEY)
    pages = notion_client.search("")
    documents = notion_client.load_data(pages=pages)
    return documents

def preprocess_documents(documents: Document) -> Tuple[List[List[str]], Dict[str, Any]]:
    print("Processing documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    metadatas = []
    for document in documents:
        content = document.text
        metadata = document.extra_info
        text_splits = text_splitter.split_text(content)
        text_splits = list(filter(lambda x: x != "", text_splits))
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

def ingest(): 
    # 1. Retrieve documents and metadata from source (Notion API)
    documents = load_notion_documents()

    # 1.5 Pre-process documents and document metadata
    # Split documents into smaller chunks due to the context limits of the LLMs.
    docs, metadatas = preprocess_documents(documents)

    # 2. Create embeddings of documents 
    # 3. Store embeddings in vector database
    create_embeddings_and_vector_db(docs, metadatas)

def question(question: str) -> None:
    index = faiss.read_index("docs.index")
    with open("db.pkl", "rb") as f:
        vector_db = pickle.load(f)

    vector_db.index = index
    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=vector_db)
    result = chain({"question": question})
    pprint(f"Question: {question}")
    pprint(f"Answer: {result['answer']}")
    pprint(f"Sources: {result['sources']}")
    return (result['answer'], "References used:\n"+"\n".join(result['sources'].split(',')))

client = slack_sdk.WebClient(token=SLACK_TOKEN)
app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(SIGNING_SECRET, '/slack/events', app)


@slack_event_adapter.on('message')
def message(payload):
    try:
        event = payload.get('event', {})
        text = event.get('text')
        cleantext = text.strip()
        if "iai:" in cleantext:
            print(cleantext)
            client.chat_postMessage(channel='#internal-ai',text='thinking...')
            resp = question("Let's think step by step. " + text)
            client.chat_postMessage(channel='#internal-ai',text=resp[0])
            client.chat_postMessage(channel='#internal-ai',text=resp[1])
    except Exception as err:
        print(err)
        pass

if __name__ == "__main__":
    # ingest() #run this first

    # parser = argparse.ArgumentParser(description='Ask a question.')
    # parser.add_argument('question', type=str)
    # args = parser.parse_args()
    # question(args.question)

    appdemo = gr.Interface(title="InternalAI v0.1 - C̶h̶a̶t̶B̶o̶t̶ ChatBrain for [FFMPEG co.]",description="InternalAI gathers information across all sources of knowledge at a company, from writeups in Notion and Google Docs to messages in apps like Slack & Discord and makes it easily searchable for onboarding of new employees or getting existing employees caught up with new projects. Coming soon is our Github integration to allow ramp-up on engineering tasks as well.", fn=question, inputs=gr.Textbox(placeholder="Ask InternalAI a question here..."), outputs=["text","text"], allow_flagging="auto", examples=["How much am I allowed to expense for dinner in the office, and where do I report those expenses? How about at events?", "Which todos are left in our Social Media Monitoring project?", "Who should I contact about the coffee machine and stocking extra flavors?"], article='<h2>Here\'s a video showing the efficacy of InternalAI when compared to searching across slack, notion and google docs manually.</h1> <iframe width="560" height="315" src="https://www.youtube.com/embed/g4oP0GiBt_g" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> <h2>FFMPEG co\'s Knowledge Sources</h1> <div>Notion: <a href="https://www.notion.so/FFMPEG-Company-Home-c8b76f986e2c4e4ab289fdd72627e50c">Link to FFMPEG co\'s Notion Homepage</a></div> <div>Google Docs: aslkdj;flajsd;lfjasl;jdfl</div> <div>Slack: aspdfowked;owiio;wimd;osmd;ocs</div> <h2>We\'ve also added a Slack integration so employees can ask InternalAI questions without even leaving their communication tools. </h2> <div>Join the Slack here: ashdfjiasdfksdfafds</div>')


    print("FOR SLACK to WORK YOU MUST RUN LOCALTUNNEL (on port 5000 where flask runs) and then change the request URL in slack settings to https://LOCALTUNNELGIVENURL.loca.lt/slack/events")

    # flask_thread = Thread(target=app.run,kwargs={'debug':False})
    # flask_thread.start()

    # appdemo.launch(share=True)

    # flask_thread.join()


    # appdemo.launch(share=True)   
    # # appdemo.launch()   
    # app.run(debug=True)
