import os
import streamlit as st
import pickle
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RockyBot", layout="wide")
st.title("📰 RockyBot: News Research Tool")

st.sidebar.header("Enter News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_store.pkl"
main_placeholder = st.empty()

llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-3.5-turbo"
)

if process_url_clicked and urls:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.info("Loading data...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    main_placeholder.info("Splitting text...")
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    main_placeholder.info("Creating embeddings...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    main_placeholder.success("Processing completed ✅")
    time.sleep(1)

query = st.text_input("Ask a question about the articles")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        result = chain({"question": query}, return_only_outputs=True)

        st.subheader("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources")
            for src in sources.split("\n"):
                st.write(src)
    else:
        st.warning("Please process URLs first.")
