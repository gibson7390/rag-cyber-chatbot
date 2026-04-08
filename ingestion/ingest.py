import os
import sys
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag.embedder import get_embedding_function

RAW_DIR = "data/raw"
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "cybersecurity"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_pdfs(raw_dir: str) -> list:
    pdf_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{raw_dir}'. Add PDFs and re-run.")
        sys.exit(1)

    all_docs = []
    for filename in pdf_files:
        path = os.path.join(raw_dir, filename)
        print(f"Loading: {filename}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} pages from {len(pdf_files)} PDF(s).")
    return all_docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def ingest(raw_dir: str = RAW_DIR, chroma_dir: str = CHROMA_DIR):
    docs = load_pdfs(raw_dir)
    chunks = chunk_documents(docs)

    embed_fn = get_embedding_function()

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print("Embedding chunks — this may take a moment...")
    embeddings = embed_fn.embed_documents(texts)

    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Cleared existing collection.")
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"Ingestion complete. {len(chunks)} chunks stored in '{chroma_dir}'.")


if __name__ == "__main__":
    ingest()
