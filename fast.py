from fastapi import FastAPI
from pydantic import BaseModel

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

app = FastAPI()

llm = ChatGroq(model="llama-3.1-8b-instant", api_key="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")
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    response = qa_chain.run({"query": query.question})
    return {"answer": response}
