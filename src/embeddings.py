from langchain.embeddings import HuggingFaceEmbeddings
from torch import cuda

EMBEDDING_250 = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_500 = "sentence-transformers/msmarco-distilbert-dot-v5"


def get_local_embeddings(embed_model_id):
    device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={"device": device},
        encode_kwargs={"device": device, "batch_size": 32},
    )

    return embed_model
