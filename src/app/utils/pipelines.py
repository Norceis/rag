from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import BaseLLM
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.vectorstores import VectorStore

from utils.prompts import rag_prompt

SYSTEM_TEMPLATE = rag_prompt


def get_retrieval_chat_pipeline(
    llm: BaseLLM, db: VectorStore, memory, n_documents: int
):
    """
    :param llm: LLM to use
    :param db: Vector Store to use
    :param memory: Memory buffer from langchain framework
    :param n_documents: Amount of documents to return
    :return: Q&A chain for Chat functionality
    """

    messages = [
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    return ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(search_kwargs={"k": n_documents}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "document_prompt": PromptTemplate(
                input_variables=["page_content", "Source"],
                template="Document name: {Source}\nContext:\n{page_content}",
            ),
            "prompt": prompt,
        },
    )
