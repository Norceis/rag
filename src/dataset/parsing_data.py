from pathlib import Path
import re

from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from dataset.allowed_types import ALLOWED_CATEGORIES
from dataset.regexes import *


def get_title_from_html_file(path: Path):
    with path.open("r", encoding="utf-8") as file:
        html_content = file.read()

        soup = BeautifulSoup(html_content, "html.parser")
        title_tag = soup.title
        if title_tag:
            title_content = title_tag.text
            # match = re.search(COLON_SPACE_PATTERN, title_content, re.DOTALL)
            # match_phrase = match.group(1) if match else None
            return title_content
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


def parse_file(path: Path):
    metadata_dict = get_full_metadata_from_html_file(path=path)
    parsed_text = parse_via_langchain_txt_loader(replace_substring_in_path(path=path, old='html', new='str'))
    return Document(page_content=parsed_text, metadata=metadata_dict)

# parse_file(Path("data/raw_dataset/html/rfc2046.html"))
