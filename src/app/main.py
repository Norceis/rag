import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

from utils.streamlit_functions import (
    load_llm,
    load_db,
    display_text_with_links,
    authorize_user,
    initialize_session_state_variables,
)
from utils.formatting import (
    format_as_table,
    format_document_name,
    markdown_justified,
)
from utils.pipelines import get_retrieval_chat_pipeline

load_dotenv()
st.set_page_config(page_title="Chat", page_icon="🗣️", layout="centered")

st.markdown(
    "<h3 style='text-align: center; color: white;'>RFC Assistant</h3>",
    unsafe_allow_html=True,
)

initialize_session_state_variables()

if not st.session_state.input_password:
    password = st.text_input("Enter password:", key="user_input_password")
    authorize_user(password)
else:
    llm = load_llm(llm_name="llama-2-70b-chat.Q4_K_M.gguf", n_gpu_layers=100)
    db = load_db(store_name="faiss", db_name="openai_1500")

    _, _, _, col, _, _, _ = st.columns(7)
    with col:
        if st.button("Reset"):
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Hello, how can I help you?",
                }
            )

    chat_pipeline = get_retrieval_chat_pipeline(
        llm,
        db,
        st.session_state.memory,
        3,
    )

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            try:
                source_docs_table = format_as_table(message, message_type=True)
                with st.chat_message(message["role"]):
                    markdown_justified(message["content"])
                    display_text_with_links(source_docs_table)
            except KeyError:
                with st.chat_message(message["role"]):
                    markdown_justified(message["content"])

    user_input = st.chat_input("What do you want to know?")
    if user_input:
        st.session_state.chat_feedback = ""
        st.session_state.chat_input = ""
        st.session_state.chat_result = ""

        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        response = dict()
        response["answer"] = ""
        while not response["answer"]:
            response = chat_pipeline(user_input)

        answer = response["answer"]
        source_docs_table = format_as_table(response, message_type=False)
        with st.chat_message("assistant"):
            markdown_justified(answer)
            display_text_with_links(source_docs_table)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "source_names": [
                    format_document_name(doc.metadata["Source"])
                    for doc in response["source_documents"]
                ],
            }
        )
        st.session_state.chat_result = response
        st.session_state.chat_input = user_input
