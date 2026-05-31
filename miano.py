from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


embeddings = OllamaEmbeddings(model="nomic-embed-text")
loader = TextLoader("my_bio.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")


from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
#Load LLM
llm = ChatOllama(model_name="nomic-embed-text")

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