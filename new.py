import streamlit as st
from dotenv import load_dotenv
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq

# Load env
load_dotenv()

st.set_page_config(page_title="LincolnBot 🤖", page_icon="🦾")
st.title("Lynne – Ask Me Anything About Lewis")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load biography file & vectorstore
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("my_bio.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_vectorstore()

# LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.7
)

# Prompt
system_prompt = """You are LincolnBot, an assistant that answers questions 
about Lincoln based on the provided biography.

- Only use the biography for answers.
- If the answer is not in the biography, reply: "I don't know".
- Format responses in markdown.
- Always include a **Final Takeaway** section.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{query}")
])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"document_variable_name": "context"}
    
)

# Input box
query = st.text_input("Ask something about me...", key="input")

# When user submits a query
if query:
    with st.spinner("Thinking... 🤔"):
        answer = qa_chain.run({"query": query})

    # Save to session chat history
    st.session_state.chat_history.append((query, answer))

    # Clear input box after submission
    st.session_state.input = ""

# Display chat history
for q, a in st.session_state.chat_history:
    st.markdown(f"**🙋 You:** {q}")
    st.markdown(f"**🤖 LincolnBot:** {a}")
    st.markdown("---")
