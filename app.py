import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import requests
from bs4 import BeautifulSoup
import os

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa' not in st.session_state:
    st.session_state.qa = None

@st.cache_resource
def load_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="path/to/llama-2-7b-chat.ggmlv3.q4_0.bin",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_website(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=load_embedding_model())
    return vectorstore

def get_sitemap(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a')
        sitemap = [link.get('href') for link in links if link.get('href') and link.get('href').startswith('/')]
        return list(set(sitemap))
    except Exception as e:
        st.error(f"Error fetching sitemap: {e}")
        return []

def main():
    st.title("Website RAG Chatbot")

    # Get website URL from user
    website_url = st.text_input("Enter the website URL:")

    if website_url and st.button("Load Website"):
        with st.spinner("Loading website content..."):
            st.session_state.vectorstore = load_website(website_url)
            llm = load_llm()
            st.session_state.qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever()
            )
        st.success("Website loaded successfully!")

        # Display sitemap
        sitemap = get_sitemap(website_url)
        if sitemap:
            st.subheader("Website Navigation")
            for link in sitemap:
                st.write(f"- [{link}]({website_url}{link})")

    if st.session_state.qa:
        st.subheader("Chat with the Website Bot")
        user_question = st.text_input("Ask a question about the website:")
        if user_question:
            with st.spinner("Generating response..."):
                response = st.session_state.qa.run(user_question)
            st.write("Answer:", response)

if __name__ == "__main__":
    main()
