import os
from rag.embedder import get_embedding_function

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "cybersecurity"
TOP_K = 5


class Retriever:
    def __init__(self):
        import chromadb

        if not os.path.isdir(CHROMA_DIR):
            raise FileNotFoundError(
                f"Chroma vector store not found at '{CHROMA_DIR}'. "
                "Run 'python ingestion/ingest.py' first to ingest documents."
            )

        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        self.embed_fn = get_embedding_function()

    def retrieve(self, query: str) -> tuple[list[str], list[dict]]:
        query_embedding = self.embed_fn.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return documents, metadatas
