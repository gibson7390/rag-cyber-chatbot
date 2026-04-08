from langchain_community.embeddings import HuggingFaceEmbeddings

MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
