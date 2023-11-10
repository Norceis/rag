import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

from utils.cached_funcs import load_llm, load_db
from utils.formatting import format_as_table, format_document_name
from utils.pipelines import get_retrieval_chat_pipeline

load_dotenv()
st.set_page_config(page_title="Chat", page_icon="🗣️", layout="centered")

st.markdown(
    "<h3 style='text-align: center; color: white;'>RFC Assistant</h3>",
    unsafe_allow_html=True,
)

_, _, _, col, _, _, _ = st.columns(7)
with col:
    if st.button("Reset"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

llm = load_llm()
db = load_db()

chat_pipeline = get_retrieval_chat_pipeline(
    llm,
    db,
    st.session_state.memory,
    3,
)

# Display chat messages from history on app rerun
if "messages" in st.session_state:
    for message in st.session_state.messages:
        try:
            source_docs_table = format_as_table(message, message_type=True)
            with st.chat_message(message["role"]):
                st.markdown(f'{message["content"]}')
                st.dataframe(source_docs_table, hide_index=True)
        except KeyError:
            with st.chat_message(message["role"]):
                st.markdown(f'{message["content"]}')

# React to user input
user_input = st.chat_input("What do you want to know?")
if user_input:
    st.session_state.chat_feedback = ""
    st.session_state.chat_input = ""
    st.session_state.chat_result = ""

    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    response = chat_pipeline(user_input)

    answer = response["answer"]
    source_docs_table = format_as_table(response, message_type=False)
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.dataframe(source_docs_table, hide_index=True)

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
