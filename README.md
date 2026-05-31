# 🦾 Lynne — AI Portfolio Assistant
Lynne is a RAG-powered chatbot that answers questions about Lewis Miano (Lincoln) — his background, skills, projects, and experience.
Instead of a static portfolio page, Lynne lets anyone have a live conversation with an AI that knows Lincoln personally. Built and deployed as a real production application.

# 🔗 Live Demo: https://professional-portfolio-5.onrender.com

# How It Works
User Question
      │
      ▼
 Cohere Embeddings
 (embed the query)
      │
      ▼
 FAISS Vector Search
 (find relevant chunks from bio)
      │
      ▼
 LLaMA 3.1 via Groq
 (generate grounded answer)
      │
      ▼
 Streamlit UI
 (display response)

Lincoln's bio (my_bio.txt) is chunked and embedded using Cohere's embed-english-light-v3.0 model
Embeddings are stored in a FAISS vector index (saved locally)
At query time, the question is embedded and semantically matched against the index
Relevant chunks are retrieved and passed as context to LLaMA 3.1 (8B) via Groq
The answer is streamed back through a Streamlit interface


# Tech Stack
LayerToolLLMllama-3.1-8b-instant via Groq APIEmbeddingsCohere embed-english-light-v3.0Vector StoreFAISS (local)OrchestrationLangChain (RetrievalQA)FrontendStreamlitDeploymentRender

# Key Engineering Decisions
Why Cohere embeddings instead of HuggingFace?
Render's free tier has a 512MB RAM limit. HuggingFace's sentence-transformers models load the full model weights into memory at startup, which caused OOM crashes on Render. Switching to Cohere's API-based embeddings offloads the compute entirely — no model weights in memory, no crash.

Why FAISS over a hosted vector DB?
For a single-document knowledge base this size, a local FAISS index is lighter, faster, and free. No additional service to manage or pay for.

Why Groq for the LLM?
Groq's inference is significantly faster than other free-tier LLM APIs, making the chat feel responsive rather than laggy — important for a portfolio demo where first impressions matter.

# Project Structure
Professional_Portfolio/
├── main.py           # Core RAG pipeline + Streamlit app
├── fast.py           # FastAPI version (alternative serving)
├── my_bio.txt        # Lincoln's bio (the knowledge base)
├── faiss_index/      # Persisted FAISS vector store
├── .streamlit/       # Streamlit configuration
├── requirement.txt   # Dependencies
├── runtime.txt       # Python version for Render
└── .gitignore        # Keeps API keys and venv out of git

# Running Locally
1. Clone the repo
bashgit clone https://github.com/Ole-Lewi/Professional_Portfolio.git
cd Professional_Portfolio
2. Create a virtual environment
bashpython -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
3. Install dependencies
bashpip install -r requirement.txt
4. Set up environment variables
Create a .env file in the root:
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
5. Build the vector index
bashpython main.py  # Runs the indexing step on first launch
6. Launch the app
bashstreamlit run main.py

# What I Learned Building This

Debugging LangChain version conflicts between community and core packages
How Render's 512MB RAM ceiling forces real architectural trade-offs
Why .env path loading order matters when multiple projects share a machine
The value of API-based vs. local embeddings for constrained deployment environments


# Author
Lewis Miano (Lincoln)
ALX Backend Web Dev · ML/NLP · Agentic AI Systems
GitHub · Live App