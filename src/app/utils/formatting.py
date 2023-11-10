from pathlib import Path
import pandas as pd


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
