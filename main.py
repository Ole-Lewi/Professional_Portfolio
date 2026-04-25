from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Load environment variables from .env file

from pathlib import Path

env_path = Path(__file__).parent / ".env"
print("Looking for .env at:", env_path)
print("File exists:", env_path.exists())



#Load .txt file and split into chunks
loader = TextLoader("my_bio.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

#Create embeddings and store in vector database
embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))
vectorstore = FAISS.from_documents(chunks, embeddings,) 
vectorstore.save_local("faiss_index")       # Save the vector store locally
print("Vector store created and saved locally as 'faiss_index'")

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
#Load LLM
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), 
               model_name="llama-3.1-8b-instant",
               temperature=0.7)

#reload vectorstore
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#Create retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=vectorstore.as_retriever()
                                       )

#Creating  the Streamlit app
import streamlit as st
st.set_page_config(page_title="Lynne 🤖", page_icon="🦾")
st.title("🦾 Lynne – Ask Me Anything About Lewis")
st.write("Hi am Lynne!")

#User input
user_query = st.text_input("Ask something about me...")
if user_query:
    with st.spinner("Thinking... 🤔"):
        answer = qa_chain.invoke({"query": user_query})["result"]
        st.markdown(f"### 🤖 Lincolnbot:\n{answer}")