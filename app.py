import streamlit as st
import openai
import os
import tempfile
import pdfplumber
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="📚 Multi-PDF Chatbot", page_icon="🧠")
st.title("🧠 Chat with Multiple PDFs")

# Session state setup
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()

clear_chat = st.checkbox("Clear chat history when uploading new PDF", value=False)

# Function to process and add PDF to vectorstore
def ingest_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        pdf_path = tmp_file.name

    # Extract text
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Initialize or add to existing vector store
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        st.session_state.vectorstore.add_documents(docs)

    # Update the retrieval chain
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=st.session_state.vectorstore.as_retriever(),
        return_source_documents=False
    )

# Upload and ingest PDFs
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
    ingest_pdf(uploaded_file)
    st.session_state.uploaded_files.add(uploaded_file.name)

    if clear_chat:
        st.session_state.chat_history = []

    st.success(f"Ingested '{uploaded_file.name}' successfully!")

# Chat interface
if st.session_state.qa_chain:
    user_input = st.chat_input("Ask your question:")
    if user_input:
        result = st.session_state.qa_chain({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })
        answer = result["answer"]
        st.session_state.chat_history.append((user_input, answer))

# Show chat history
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
