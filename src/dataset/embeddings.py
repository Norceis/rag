from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dataset.parsing_data_utils import load_all_documents
from pathlib import Path
from pathlib import Path
from langchain.vectorstores import FAISS
from dataset.parsing_data import load_all_documents
import os
from tqdm import tqdm
import pinecone
from langchain.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from torch import cuda

# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from torch import cuda

EMBEDDING_250 = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_500 = "sentence-transformers/msmarco-distilbert-dot-v5"
EMBEDDING_OPENAI = "openai"
EMBEDDING_TYPE = EMBEDDING_OPENAI
SPLIT_DATASET_DIR = Path("../../data/split_dataset/4")
EMBEDDED_DATASET_DIR = Path("../../data/embedded_dataset/faiss/2")


def get_local_embeddings(embed_model_id):
    device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={"device": device},
        encode_kwargs={"device": device, "batch_size": 32},
    )

    return embed_model


def embed_dataset_locally(docs: list[Document], embed_model: Embeddings):
    if isinstance(embed_model, OpenAIEmbeddings):
        db = FAISS.from_documents(docs, embed_model)
        db.save_local(EMBEDDED_DATASET_DIR / f"faiss_idx")

    elif isinstance(embed_model, HuggingFaceEmbeddings):
        global_db = FAISS.from_documents([docs[0]], embed_model)

        batch_size = 1000
        counter = 0
        for idx in tqdm(range(1, len(docs), batch_size), desc="Embedding in progress"):
            global_db.add_documents(docs[idx : idx + batch_size])
            if not counter % 500:
                global_db.save_local(EMBEDDED_DATASET_DIR / f"faiss_idx_ckpt_{idx}")
            counter += 1

        global_db.save_local(EMBEDDED_DATASET_DIR / f"faiss_idx")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    load_dotenv()
    docs = load_all_documents(SPLIT_DATASET_DIR)

    if EMBEDDING_TYPE == EMBEDDING_OPENAI:
        embed_model = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"), show_progress_bar=True
        )
    elif EMBEDDING_TYPE == EMBEDDING_250:
        embed_model = get_local_embeddings("sentence-transformers/all-MiniLM-L6-v2")

    elif EMBEDDING_TYPE == EMBEDDING_500:
        embed_model = get_local_embeddings(
            "sentence-transformers/msmarco-distilbert-dot-v5"
        )

    embed_dataset_locally(docs, embed_model)
