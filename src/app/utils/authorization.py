from pathlib import Path

import streamlit as st

from utils.streamlit_functions import get_available_llms, get_available_dbs


def authorize_user(password: str):
    match password:
        case "rag-test-1337":  # ADMIN
            st.session_state.input_password = password
            llms = get_available_llms(Path("../../models"))
            llms.append("openai")
            dbs = get_available_dbs(Path("../../data/embedded_dataset/faiss"))

            number_of_ss_docs_returned = st.slider(
                label="Choose number of docs returned",
                min_value=1,
                max_value=9,
                value=3,
            )
            context_len_of_llm = st.slider(
                label="Choose length of LLM context",
                min_value=4096,
                max_value=32768,
                value=8192,
            )
            n_gpu_layers = st.slider(
                label="Choose number of layers to offload from RAM to GPU",
                min_value=10,
                max_value=100,
                value=40,
            )
            llm_name = st.radio("Choose LLM", llms, index=len(llms) - 1)
            db_name = st.radio(
                "Choose DB",
                dbs,
                index=len(llms) - 1,
            )

            if st.button("Confirm choices"):
                st.session_state.number_of_ss_docs_returned = number_of_ss_docs_returned
                st.session_state.context_len_of_llm = context_len_of_llm
                st.session_state.n_gpu_layers = n_gpu_layers
                st.session_state.llm_name = llm_name
                st.session_state.db_name = db_name
                st.rerun()

        case "rag-test-8369":  # GROUP 1
            st.session_state.input_password = password
            st.session_state.llm_name = "openai"
            st.session_state.db_name = "openai_1500"
            st.rerun()
        case "rag-test-7555":  # GROUP 2
            st.session_state.input_password = password
            st.session_state.llm_name = "openai"
            st.session_state.db_name = "local_500"
            st.rerun()
        case "rag-test-9952":  # GROUP 3
            st.session_state.input_password = password
            st.session_state.llm_name = "nous-hermes-llama2-13b.Q4_0.gguf"
            st.session_state.db_name = "openai_1500"
            st.rerun()
        case "rag-test-9632":  # GROUP 4
            st.session_state.input_password = password
            st.session_state.llm_name = "nous-hermes-llama2-13b.Q4_0.gguf"
            st.session_state.db_name = "local_500"
            st.rerun()
        case _ if password:
            st.markdown("Password incorrect")
