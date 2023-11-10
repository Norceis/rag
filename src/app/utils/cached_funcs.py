import os
from pathlib import Path

import streamlit as st
from langchain.llms import OpenAI
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings


@st.cache_resource
def load_llm(llm_name: str = "openai"):
    if llm_name == "openai":
        return OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise NotImplementedError


@st.cache_resource
def load_db(db_name: str = "faiss"):
    if db_name == "faiss":
        return load_faiss()
    else:
        raise NotImplementedError


@st.cache_resource
def load_faiss():
    faiss_local_path = Path(f"../../data/embedded_dataset/faiss/openai_1000/faiss_idx")
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local(faiss_local_path, embeddings)
