import os
import pickle
import time
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

# Read API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# If using Streamlit Cloud Secrets
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="RockyBot", layout="wide")
st.title("📰 News Research Tool")

st.sidebar.header("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_button = st.sidebar.button("Process URLs")

FILE_PATH = "faiss_store.pkl"

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    api_key=OPENAI_API_KEY
)

placeholder = st.empty()
