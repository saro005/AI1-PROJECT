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

st.set_page_config(page_title="RockyBot", layout="wide")
st.title("📰 RockyBot : News Research Tool")

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
    temperature=0.3
)

placeholder = st.empty()

if process_button:

    if len(urls) == 0:
        st.warning("Please enter at least one URL.")
        st.stop()

    loader = UnstructuredURLLoader(urls=urls)

    placeholder.info("Loading articles...")
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    docs = splitter.split_documents(data)

    placeholder.info("Creating embeddings...")

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    with open(FILE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

    placeholder.success("Articles processed successfully!")
    time.sleep(1)

query = st.text_input("Ask a question")

if query:

    if not os.path.exists(FILE_PATH):
        st.warning("Please process URLs first.")
        st.stop()

    with open(FILE_PATH, "rb") as f:
        vectorstore = pickle.load(f)

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    result = chain.invoke(
        {"question": query}
    )

    st.subheader("Answer")
    st.write(result["answer"])

    if result.get("sources"):
        st.subheader("Sources")
        st.write(result["sources"])
