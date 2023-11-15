import os
from pathlib import Path
from torch import cuda
import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings


@st.cache_resource
def load_llm(llm_name: str = "openai_1000"):
    if llm_name == "openai_1000":
        return OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    elif llm_name == "orca":
        return load_local_llm("orca")
    elif llm_name == "llama2":
        return load_local_llm("llama2")
    else:
        raise NotImplementedError


@st.cache_resource
def load_local_llm(local_llm_name: str):
    if local_llm_name == "orca":
        model_path = "../../models/mistral-7b-openorca.Q4_0.gguf"
    elif local_llm_name == "llama2":
        model_path = "../../models/nous-hermes-llama2-13b.Q4_0.gguf"
    else:
        raise NotImplementedError

    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=40,
        n_batch=256,
        n_ctx=2048,
        f16_kv=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
        streaming=False,
    )

    return llm


@st.cache_resource
def get_local_embeddings(embed_model_id):
    device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={"device": device},
        encode_kwargs={"device": device, "batch_size": 32},
    )

    return embed_model


@st.cache_resource
def load_db(store_name: str = "faiss", db_name: str = "local_500"):
    if store_name == "faiss":
        return load_faiss(db_name)
    else:
        raise NotImplementedError


@st.cache_resource
def load_faiss(db_name: str = "local_500"):
    faiss_local_path = Path(f"../../data/embedded_dataset/faiss/{db_name}/faiss_idx")

    if db_name == "openai_1000":
        embed_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    elif db_name == "local_500":
        embed_model = get_local_embeddings(
            "sentence-transformers/msmarco-distilbert-dot-v5"
        )
    elif db_name == "local_250":
        embed_model = get_local_embeddings("sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise NotImplementedError

    return FAISS.load_local(str(faiss_local_path), embed_model)
