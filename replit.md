# rag-cyber-chatbot

RAG-powered cybersecurity knowledge chatbot using LangChain, Chroma, and Groq API.

## Stack
- **Language**: Python 3.11
- **LLM**: Groq API (llama3-8b-8192)
- **Embeddings**: HuggingFace all-MiniLM-L6-v2 (sentence-transformers)
- **Vector Store**: Chroma (persistent, local)
- **Framework**: LangChain + langchain-community
- **Interface**: Command line only

## Project Structure
```
main.py                  # Entry point — asks user for a question, prints the answer
rag/
  pipeline.py            # Connects retrieval and generation (Groq LLM)
  retriever.py           # Loads Chroma vector store, retrieves top 3 relevant chunks
  embedder.py            # HuggingFace all-MiniLM-L6-v2 embedding function
ingestion/
  ingest.py              # Loads PDFs from data/raw/, chunks, embeds, saves to Chroma
prompts/
  system_prompt.txt      # System prompt with hallucination guard
data/
  raw/                   # Place PDF files here before running ingestion
  chroma/                # Auto-created by ingest.py (Chroma vector store)
requirements.txt
.env.example
```

## Setup
1. Add `GROQ_API_KEY` as a Replit secret.
2. Place PDF files in `data/raw/`.
3. Run ingestion: `python ingestion/ingest.py`
4. Start the chatbot: `python main.py`

## Environment Variables
- `GROQ_API_KEY` — Groq API key (set as a Replit secret)
