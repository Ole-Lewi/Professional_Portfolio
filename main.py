from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

#Load .txt file and split into chunks
loader = TextLoader("my_bio.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

#Create embeddings and store in vector database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings,) 
vectorstore.save_local("faiss_index")       # Save the vector store locally
print("Vector store created and saved locally as 'faiss_index'")

from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

#Load LLM
llm = ChatGroq(api_key="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", 
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
        answer = qa_chain.run({'query': user_query})
        st.markdown(f"### 🤖 LincolnBot:\n{answer}")