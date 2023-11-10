import concurrent.futures
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader, UnstructuredHTMLLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from dataset.allowed_types import ALLOWED_CATEGORIES
from dataset.regexes import *

CLEAN_DATASET_DIR = Path("data/clean_dataset/html")
SPLIT_DATASET_DIR = Path("data/split_dataset/6")
CHUNK_SIZE = 250
CHUNK_OVERLAP = 25


def get_title_from_html_file(path: Path):
    with path.open("r", encoding="utf-8") as file:
        html_content = file.read()

        soup = BeautifulSoup(html_content, "html.parser")
        title_tag = soup.title
        if title_tag:
            title_content = title_tag.text
            match = re.search(COLON_SPACE_PATTERN, title_content, re.DOTALL)
            match_phrase = match.group(1) if match else None
            return match_phrase
        return None


def get_dirty_metadata_field(pattern, string):
    match = re.search(pattern, string, re.DOTALL)
    matching_element = match.group(1) if match else None
    return matching_element


def get_clean_metadata_field(pattern: str, string: str):
    if string:
        correct_entry = string.split("    ")[0]
        match = re.findall(pattern, correct_entry, re.DOTALL)
        return match
    return None


def get_category_field(string):
    if string:
        correct_entry = string.split("    ")[0]
        if correct_entry in ALLOWED_CATEGORIES:
            return correct_entry
        return None
    return None


def get_full_metadata_from_html_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        html_content = file.read()
        pattern_contents_dirty = {
            "updates_pattern": get_dirty_metadata_field(UPDATES_PATTERN, html_content),
            "obsoletes_pattern": get_dirty_metadata_field(
                OBSOLETES_PATTERN, html_content
            ),
            "category_pattern": get_dirty_metadata_field(
                CATEGORY_PATTERN, html_content
            ),
            "issn_pattern": get_dirty_metadata_field(ISSN_PATTERN, html_content),
            "updated_by_pattern": get_dirty_metadata_field(
                UPDATED_BY_PATTERN, html_content
            ),
            "NIC_pattern": get_dirty_metadata_field(NIC_PATTERN, html_content),
            "obsoleted_by_pattern": get_dirty_metadata_field(
                OBSOLETED_BY_PATTERN, html_content
            ),
            "related_rfcs_pattern": get_dirty_metadata_field(
                RELATED_RFCS_PATTERN, html_content
            ),
        }

    pattern_contents_clean = {
        "Source": path.name[:-5],  # make this smarter
        "Title": get_title_from_html_file(path),
        "Updates": get_clean_metadata_field(
            NUMBERS_PATTERN, pattern_contents_dirty["updates_pattern"]
        ),
        "Obsoletes": get_clean_metadata_field(
            NUMBERS_PATTERN, pattern_contents_dirty["obsoletes_pattern"]
        ),
        "Category": get_category_field(pattern_contents_dirty["category_pattern"]),
        "ISSN": get_clean_metadata_field(
            ISSN_PATTERN, pattern_contents_dirty["issn_pattern"]
        ),
        "Updated by": get_clean_metadata_field(
            NUMBERS_PATTERN, pattern_contents_dirty["updated_by_pattern"]
        ),
        "NIC": get_clean_metadata_field(
            NIC_PATTERN, pattern_contents_dirty["NIC_pattern"]
        ),
        "Obsoleted by": get_clean_metadata_field(
            NUMBERS_PATTERN, pattern_contents_dirty["obsoleted_by_pattern"]
        ),
        "Related RFCs": get_clean_metadata_field(
            NUMBERS_PATTERN, pattern_contents_dirty["related_rfcs_pattern"]
        ),
    }

    return pattern_contents_clean


def replace_substring_in_path(path: Path, old: str, new: str) -> Path:
    return Path(str(path).replace(old, new))


def parse_via_langchain_txt_loader(path: Path) -> str:
    loader = TextLoader(path)
    return loader.load()[0].page_content


def remove_new_page_separators_from_html(
    path: Path,
) -> str:  # returns html file in string format
    with path.open("r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    elements_to_remove = soup.find_all("span", class_="grey")
    for element in elements_to_remove:
        element.extract()

    elements_to_remove = soup.find_all("hr", class_="noprint")
    for element in elements_to_remove:
        element.extract()

    # pre_tags = soup.find_all('pre', class_='newpage')
    # for pre_tag in pre_tags:
    #     pre_tag.decompose()

    return str(soup)


def extract_visible_text(html_content: str):
    # with path.open('r', encoding='utf-8') as file:
    #     html_content = file.read()
    soup = BeautifulSoup(html_content, "html.parser")
    texts = soup.find_all(string=True)
    visible_texts = []
    for text in texts:
        if text.parent.name not in [
            "style",
            "script",
            "head",
            "title",
            "meta",
            "[document]",
        ]:
            # print(text)
            if text == "NewPage":
                continue
            visible_texts.append(text)
    return " ".join(visible_texts).strip()


# OLD, BAD
# def parse_file(path: Path):
#     metadata_dict = get_full_metadata_from_html_file(path=path)
#     parsed_text = parse_via_langchain_txt_loader(
#         replace_substring_in_path(path=path, old="html", new="str")
#     )
#     return Document(page_content=parsed_text, metadata=metadata_dict)


# OLD, WORKS ONLY FOR FILES 1-8649
# def parse_html_file(path: Path):
#     loader = UnstructuredHTMLLoader(path, mode="single", strategy="fast")
#     doc = loader.load()
#     metadata = get_full_metadata_from_html_file(path)
#     return Document(page_content=doc[0].page_content, metadata=metadata)


def load_raw_text_with_soup(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


def parse_html_file(path: Path):
    content = load_raw_text_with_soup(path)
    metadata = get_full_metadata_from_html_file(path)
    return Document(page_content=content, metadata=metadata)


def split_document(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splitted_docs = splitter.split_documents(documents)
    return splitted_docs


def preprocess_and_save_html_file(path: Path):
    if not SPLIT_DATASET_DIR.exists():
        SPLIT_DATASET_DIR.mkdir(parents=True)

    save_path = SPLIT_DATASET_DIR / f"{path.name[:-5]}.json"

    if save_path.exists():
        return

    doc = parse_html_file(path)
    splitted_docs = split_document([doc])
    save_docs_to_json(splitted_docs, save_path)


def save_docs_to_json(list_of_docs: list, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as json_file:
        for doc in list_of_docs:
            json.dump(doc.json(), json_file)
            json_file.write("\n")


def read_docs_from_json(file_path: str) -> list:
    array = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            data_data = json.loads(data)
            obj = Document(**data_data)
            array.append(obj)
    return array


def load_all_documents(path: Path):
    splitted_documents = []
    for file_path in tqdm(path.glob("*.json"), desc="Processing RFCs", unit="file"):
        splitted_documents.extend(read_docs_from_json(file_path))
    return splitted_documents


if __name__ == "__main__":
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
