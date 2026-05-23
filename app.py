import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings

import time

from dotenv import load_dotenv
load_dotenv()

# Set up Groq API key   
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model="qwen/qwen3-32b", temperature=0.8)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context provided.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers/")
        st.session_state.docs = st.session_state.loader.load() # Documents loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  
        st.session_state.vectors = vectors

user_prompt = st.text_input("Enter your question from research papers:")

if st.button("Documents Embedding"):
    create_vector_embedding()
    st.write("Vectors created")

if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please click 'Documents Embedding' first to create the vector database.")
    else:
        with st.spinner("Generating response..."):
            documents_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, documents_chain)
            start_time = time.process_time()
            response = retrieval_chain.invoke({"input": user_prompt})
            end_time = time.process_time()
            execution_time = end_time - start_time
            st.write("Execution time:", execution_time, "seconds")
            st.write(response["answer"])

            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("=====================================")
