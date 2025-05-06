import streamlit as st
import openai
import os
import pdfplumber
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="📚 Multi-PDF Chatbot", page_icon="🧠")
st.title("🤖 Chat with PDFs")

# Session state setup
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Function to read and chunk PDFs ---
def load_and_chunk_pdfs(folder_path="./pdfs"):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            with pdfplumber.open(pdf_path) as pdf:
                full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                if full_text.strip():
                    doc = Document(page_content=full_text, metadata={"source": filename})
                    chunks = splitter.split_documents([doc])
                    all_chunks.extend(chunks)

    return all_chunks

# --- Ingest all PDFs at app startup ---
if st.session_state.vectorstore is None:
    with st.spinner("Loading PDFs..."):
        all_chunks = load_and_chunk_pdfs("./pdfs")

        if all_chunks:
            embeddings = OpenAIEmbeddings()
            st.session_state.vectorstore = FAISS.from_documents(all_chunks, embeddings)
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0),
                retriever=st.session_state.vectorstore.as_retriever(),
                return_source_documents=False
            )
            st.success("All PDFs loaded and ready for chat!")
        else:
            st.error("No text extracted from any PDFs. Check contents.")

# --- Chat UI ---
if st.session_state.qa_chain:
    user_input = st.chat_input("Ask a question based on the PDFs...")
    if user_input:
        result = st.session_state.qa_chain({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })
        answer = result["answer"]
        st.session_state.chat_history.append((user_input, answer))

# --- Show chat history ---
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
