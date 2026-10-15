OFFLINE_MODE = False  # Change to True when offline

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()
env_path = Path(__file__).parent / ".env"
print("Looking for .env at:", env_path)
print("File exists:", env_path.exists())

# ─── LLM & Embeddings Setup ───────────────────────────────────────────────────
if OFFLINE_MODE:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    llm = ChatOllama(model="llama3.2:1b")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
else:
    from langchain_groq import ChatGroq
    from langchain_cohere import CohereEmbeddings
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    )
    embeddings = CohereEmbeddings(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model="embed-english-light-v3.0"
    )

# ─── Load & Index Bio ─────────────────────────────────────────────────────────
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

loader = TextLoader("my_bio.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("Vector store created and saved locally as 'faiss_index'")

# ─── Reload Vectorstore ──────────────────────────────────────────────
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# ─── Modern Retrieval Chain (replaces RetrievalQA) ───────────────────────────
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

Prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Lynne, a personal AI assistant representing Lewis Miano Warui (also known as Lincoln).

When describing Lewis, always prioritize his skills and experience in this exact order:
1. Agentic AI — LangChain, LangGraph, LangSmith, Multi-Agent Systems, RAG, HITL, Agent Evaluation (RAGAS, DeepEval)
2. Generative AI — LLMs, Prompt Engineering, HuggingFace, Groq, Cohere, Ollama
3. Fine Tuning — Custom model training and adaptation
4. Prompt Engineering — Designing effective prompts for LLMs
5. Machine Learning — Supervised, Unsupervised, Deep Learning, NLP, Scikit-learn, XGBoost
6. Software Development — Backend (Python, Django, FastAPI), Frontend (HTML, CSS, JS, React)

Answer questions about Lewis based only on the context provided below.
In any question asked, be brief and avoid giving unnecessary details.
Always give a final takeaway or summary of the answer in one sentence. i.e "would you like to know more about Lewis' experience in Agentic AI?"
If the answer is not in the context, say "I don't have that information about Lewis."
Be friendly, professional and concise.

Context: {context}

Question: {question}

Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": Prompt}
)
# ─── Streamlit App ────────────────────────────────────────────────────────────
import streamlit as st

st.set_page_config(page_title="Lynne 🤖", page_icon="🦾")
st.title("🦾 Lynne – Ask Me Anything About Lewis")
st.write("Hi, I'm Lynne!")

user_query = st.text_input("Ask something about me...")
if user_query:
    with st.spinner("Thinking... 🤔"):
        answer = qa_chain.invoke({"query": user_query})["result"]
        st.markdown(f"### 🤖 Lynne:\n{answer}")