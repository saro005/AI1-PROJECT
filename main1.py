import os
import pickle
import time
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

# Read OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# If running on Streamlit Cloud, read from Secrets
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
st.write("OPENAI_API_KEY:", OPENAI_API_KEY)
st.write("Type:", type(OPENAI_API_KEY))
st.write("Length:", len(OPENAI_API_KEY) if OPENAI_API_KEY else 0)

try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=OPENAI_API_KEY
    )
except Exception as e:
    st.exception(e)
placeholder = st.empty()
