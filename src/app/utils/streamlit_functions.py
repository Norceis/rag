import os
from pathlib import Path

import pandas as pd
from langchain.memory import ConversationBufferMemory
from torch import cuda
import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings

from utils.formatting import display_clickable_table, display_clickable_text


def get_available_llms():
    directory_path = Path("../../models")
    files = directory_path.glob(f"*.gguf")
    file_names = [str(file.name) for file in files]
    return file_names


@st.cache_resource
def load_llm(llm_name: str = "openai", n_gpu_layers: int = 100, context_len: int = 8192):
    if llm_name == "openai":
        return OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    elif llm_name in get_available_llms():
        return load_local_llm(local_llm_name=llm_name, n_gpu_layers=n_gpu_layers, context_len=context_len)
    else:
        raise NotImplementedError


@st.cache_resource
def load_local_llm(
    local_llm_name: str = "mistral-7b-openorca.Q4_0.gguf", n_gpu_layers: int = 100, context_len: int = 8192
):
    model_path = "../../models/" + local_llm_name

    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=256,
        n_ctx=context_len,
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

    if "openai" in db_name:
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


def display_sidebar_with_links():
    with st.sidebar:
        set_of_docs = set()
        for message in st.session_state.messages:
            try:
                for source in message["source_names"]:
                    set_of_docs.add(source)
            except KeyError:
                pass

        if set_of_docs:
            display_clickable_table(set_of_docs)


def display_text_with_links(docs_table: pd.DataFrame):
    set_of_docs = set(docs_table.iloc[0, 0])
    if set_of_docs:
        display_clickable_text(set_of_docs)


def initialize_session_state_variables():
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Hello, how can I help you?",
            }
        )

    if "input_password" not in st.session_state:
        st.session_state.input_password = ""


def authorize_user(password: str):
    if password == "rag-test-1337":
        st.session_state.input_password = password
        st.rerun()
    elif password:
        st.markdown("Password incorrect")
