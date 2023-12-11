import os
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm
from langchain.llms import LlamaCpp
from torch import cuda
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from tvalmetrics import RagScoresCalculator
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.manager import CallbackManager

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
N_TESTS_FOR_QUESTION = 10

SYSTEM_TEMPLATE = """Create an informative and comprehensive answer for a given question based solely on the given documents. You must only use information from the given documents.
Use an unbiased and journalistic tone. Do not repeat text.
Cite the documents using [Document name] notation.
If multiple documents contain the answer, cite those documents like ‘as stated in [Document name 1], [Document name 2], etc.’.
You must include citations in your answer.
If the documents do not contain the answer to the question, say that  ‘answering is not possible given the available information.’
{context}

"""
MESSAGES = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
]

PROMPT = ChatPromptTemplate.from_messages(MESSAGES)

QUESTIONS = [
    "How does OAuth facilitate federated authentication in RDAP?",
    "How is command numbering managed in iscsi protocol?",
    "What actions should an SCTP endpoint take when initializing an association?",
    "Does the error response have to be signed by the same key as the original request in TSIG error handling?",
    "Explain to me how to perform secure addition of a new SEP key to a trust point DNSKEY RRSet",
    "What is the difference between very large, but finite delay and packet loss?",
    "What should I do with my TCP implementation if I'm encountering an ICMP Destination Unreachable message with codes 2-4?",
    "What is the difference between ACP point-to-point and multi-access virtual interfaces?",
    "How does the use of DNSSEC impact the interaction between DNS64 and DNS recursive resolvers?",
    "How does YANG support conditional augmentation of data nodes?",
]

REFERENCE_ANSWERS = [
    "Using OAuth, multiple RDAP servers can form a federation, and the clients can access any server in the same federation by providing one credential registered in any server in that federation.  The OAuth authorization framework is designed for use with HTTP and thus can be used with RDAP.",
    "Command numbering starts with the first Login Request on the first connection of a session (the leading login on the leading connection), and the CmdSN MUST be incremented by 1 in a Serial Number Arithmetic sense, as defined in [RFC1982], for every non-immediate command issued afterwards.",
    "During the association initialization, an endpoint uses the following rules to discover and collect the destination transport address(es) of its peer. If there are no address parameters present in the received INIT or INIT ACK chunk, the endpoint MUST take the source IP address from which the chunk arrives and record it, in combination with the SCTP Source Port Number, as the only destination transport address for this peer. If there is a Host Name Address parameter present in the received INIT or INIT ACK chunk, the endpoint MUST immediately send an ABORT chunk and MAY include an 'Unresolvable Address' error cause to its peer. The ABORT chunk SHOULD be sent to the source IP address from which the last peer packet was received. If there are only IPv4/IPv6 addresses present in the received INIT or INIT ACK chunk, the receiver MUST derive and record all the transport addresses from the received chunk AND the source IP address that sent the INIT or INIT ACK chunk. The transport addresses are derived by the combination of SCTP Source Port Number (from the common header) and the IP Address parameter(s) carried in the INIT or INIT ACK chunk and the source IP address of the IP datagram. The receiver SHOULD use only these transport addresses as destination transport addresses when sending subsequent packets to its peer. An INIT or INIT ACK chunk MUST be treated as belonging to an already established association (or one in the process of being established) if the use of any of the valid address parameters contained within the chunk would identify an existing TCB.",
    "Yes",
    "Operator adds new SEP key to trust point DNSKey, it is being validated based on the self-signed RRSet. If no other new SEP key is seen in a validated trust points, resolver starts acceptance after proper amount of time expired",
    "Depends on the application, there is a defined range of delay with a specified upper bound (Tmax)",
    "TCP implementations SHOULD abort the connection (SHLD-26)",
    "Implementation of mapping secure channels: point-to-point interfaces create a separate virtual interface for each secure channel, while multi-access interfaces consolidate multiple secure channels into a single virtual interface associated with the underlying subnet",
    "Validating DNS64 resolver increases the confidence on the synthetic AAAA records, as it has validated that a non-synthetic AAAA record doesn't exist. However, if the client device is oblivious to NAT64 (the most common case) and performs DNSSEC validation on the AAAA record, it will fail as it is a synthesized record.",
    "conditional augmentation of data nodes is supported through the use of the 'augment' statement in combination with the 'when' statement. The 'augment' statement allows you to add or extend data nodes in an existing data tree, and the 'when' statement provides a condition under which the augmentation takes effect.",
]


def generate_answers(llm, db, n_documents: int) -> tuple:
    questions = []
    reference_answers = []
    answers = []
    retrieved_contexts = []

    total_iterations = len(QUESTIONS) * N_TESTS_FOR_QUESTION
    progress_bar = tqdm(total=total_iterations, desc="Generating answers in progress")

    for idx in range(len(QUESTIONS)):
        for _ in range(N_TESTS_FOR_QUESTION):
            response = dict()
            response["answer"] = ""
            while not response["answer"]:
                conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm,
                    db.as_retriever(search_kwargs={"k": n_documents}),
                    memory=ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True,
                        output_key="answer",
                    ),
                    return_source_documents=True,
                    combine_docs_chain_kwargs={
                        "document_prompt": PromptTemplate(
                            input_variables=["page_content", "Source"],
                            template="Document name: {Source}\nContext:\n{page_content}",
                        ),
                        "prompt": PROMPT,
                    },
                )
                response = conversation_chain(QUESTIONS[idx])

            questions.append(QUESTIONS[idx])
            reference_answers.append(REFERENCE_ANSWERS[idx])
            answers.append(response["answer"])
            retrieved_contexts.append(response["source_documents"])
            progress_bar.update(1)

    progress_bar.close()

    return questions, reference_answers, answers, retrieved_contexts


def generate_scores(
    questions: list, reference_answers: list, answers: list, retrieved_contexts: list
) -> pd.DataFrame:
    llm_evaluator = "gpt-4"
    score_calculator = RagScoresCalculator(
        model=llm_evaluator,
        answer_similarity_score=True,
        retrieval_precision=True,
        augmentation_precision=True,
        augmentation_accuracy=True,
        answer_consistency=True,
    )

    scores = score_calculator.score_batch(
        questions, reference_answers, answers, retrieved_contexts
    )

    return scores.to_dataframe()


def rename_scores(scores: pd.DataFrame) -> pd.DataFrame:
    renamed_df = scores.rename(
        columns={
            "question": "Question",
            "reference_answer": "Reference answer",
            "llm_answer": "LLM answer",
            "retrieved_context": "Retrieved context",
            "answer_similarity_score": "Answer similarity score",
            "retrieval_precision": "Retrieval precision",
            "augmentation_precision": "Augmentation precision",
            "augmentation_accuracy": "Augmentation accuracy",
            "answer_consistency": "Answer consistency",
            "overall_score": "Overall score",
        }
    )

    return renamed_df


def calculate_score_means(scores: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = scores.select_dtypes(include=[np.number]).columns
    mean_df = scores.groupby(np.arange(len(scores)) // N_TESTS_FOR_QUESTION)[
        numeric_columns
    ].mean()

    result_df = pd.concat(
        [
            pd.DataFrame(QUESTIONS, columns=["Question"]),
            mean_df[
                [
                    "Answer similarity score",
                    "Retrieval precision",
                    "Augmentation precision",
                    "Augmentation accuracy",
                    "Answer consistency",
                    "Overall score",
                ]
            ],
        ],
        axis=1,
    )

    return result_df


def get_llm(llm_name: str):
    match llm_name:
        case "openai":
            llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
            return llm

        case "local":
            llm = LlamaCpp(
                model_path="models/nous-hermes-llama2-13b.Q4_0.gguf",
                n_gpu_layers=45,
                n_batch=256,
                n_ctx=8192,
                f16_kv=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                verbose=True,
                streaming=False,
            )

            return llm


def get_db(db_name: str):
    match db_name:
        case "openai":
            faiss_local_path = "data/embedded_dataset/faiss/openai_1500/faiss_idx"
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            db = FAISS.load_local(faiss_local_path, embeddings)
            return db

        case "local":
            faiss_local_path = "data/embedded_dataset/faiss/local_500/faiss_idx"
            device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/msmarco-distilbert-dot-v5",
                model_kwargs={"device": device},
                encode_kwargs={"device": device, "batch_size": 32},
            )
            db = FAISS.load_local(faiss_local_path, embeddings)
            return db


if __name__ == "__main__":
    COMBINATIONS = {
        "openai_openai": ("openai", "openai"),
        "openai_local": ("openai", "local"),
        "local_openai": ("local", "openai"),
        "local_local": (
            "local",
            "local",
        ),
    }

    for name, values in COMBINATIONS.items():
        db = get_db(values[0])
        llm = get_llm(values[1])

        print(f"-------------------------- Starting {name} -------------------------- ")
        if values[0] == "local":
            (
                questions,
                reference_answers,
                answers,
                retrieved_contexts,
            ) = generate_answers(llm, db, n_documents=7)

        else:
            (
                questions,
                reference_answers,
                answers,
                retrieved_contexts,
            ) = generate_answers(llm, db, n_documents=3)

        start_time = time.time()
        scores = generate_scores(
            questions, reference_answers, answers, retrieved_contexts
        )

        print(
            f"--------------------------  Scoring took {round((time.time() - start_time) / 60, 2)} minutes -------------------------- "
        )

        renamed_scores = rename_scores(scores)
        renamed_scores.to_json(
            f"data/evaluation/automatic/LONG_{name}.json", orient="records"
        )

        short_scores = calculate_score_means(renamed_scores)
        short_scores.to_json(
            f"data/evaluation/automatic/SHORT_{name}.json", orient="records"
        )
        print(
            f"--------------------------  {name} results saved successfully -------------------------- "
        )
