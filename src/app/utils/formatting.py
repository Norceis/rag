from pathlib import Path
import pandas as pd
import streamlit as st


def format_document_name(name: str):
    return Path(name).name


def format_as_table(response, message_type: bool = True):
    table = pd.DataFrame()
    if message_type:
        table["Source documents"] = [response["source_names"]]
    else:
        table["Source documents"] = [
            [
                format_document_name(doc.metadata["Source"])
                for doc in response["source_documents"]
            ]
        ]
    return table


def display_clickable_table(set_of_docs: set):
    df = pd.DataFrame(columns=["Source documents"])

    for doc_name in set_of_docs:
        url = f"https://www.rfc-editor.org/rfc/{doc_name}.html"
        new_row_data = {
            "Source documents": f'<a target="_blank" href="{url}">{doc_name}</a>'
        }
        df.loc[len(df)] = new_row_data
    return st.write(df.to_html(escape=False), unsafe_allow_html=True)


def display_clickable_text(set_of_docs: set):
    links = []

    for doc_name in set_of_docs:
        url = f"https://www.rfc-editor.org/rfc/{doc_name}.html"
        new_row_data = f'<a target="_blank" href="{url}">RFC {doc_name[3:]}</a>'
        links.append(new_row_data)

    return markdown_prettified(f'Source documents: {" ".join(links)}')


def markdown_justified(text):
    return st.markdown(
        f'<div style="text-align: justify;">{text}</div>',
        unsafe_allow_html=True,
    )


def markdown_prettified(text):
    st.markdown(
        """
    <style>
        .fancy-font {
            font-family: 'Roboto', sans-serif;
            color: #A3B763;
            font-size: 12px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    return st.markdown(f'<p class="fancy-font">{text}</p>', unsafe_allow_html=True)
