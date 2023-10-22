from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures

from tqdm import tqdm

from dataset.parsing_data_utils import parse_html_file, save_docs_to_json

CLEAN_DATASET_DIR = Path("../../data/clean_dataset/html")
SPLIT_DATASET_DIR = Path("../../data/split_dataset/3")
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 250


def preprocess_and_save_html_file(path: Path):
    save_path = SPLIT_DATASET_DIR / f"{path.name[:-5]}.json"

    if save_path.exists():
        return

    # check other parsers
    doc = parse_html_file(path)

    # check other splitters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splitted_docs = splitter.split_documents([doc])

    save_docs_to_json(splitted_docs, save_path)


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    list(
        tqdm(
            executor.map(
                preprocess_and_save_html_file,
                CLEAN_DATASET_DIR.glob("*.html"),
            ),
            total=len(list(CLEAN_DATASET_DIR.glob("*.html"))),
            desc="Splitting and saving",
            unit=" file",
        )
    )
