from pathlib import Path
import requests
from tqdm import tqdm

BASE_RFC_URL = "https://www.rfc-editor.org/rfc/"
RAW_PATH_DATASET = Path("data/raw_dataset")


def download_website(url: str, output_file: str, logger):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(response.text)

        logger.info(
            f"Website content downloaded successfully and saved to {output_file}"
        )

    except requests.exceptions.HTTPError as errh:
        logger.error(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        logger.error(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        logger.error(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        logger.error(f"Request Exception: {err}")


def download_rfc(rfc_number: int, logger, doc_format: str = "html"):
    if doc_format == "html":
        rfc_name = f"rfc{rfc_number}.html"
    elif doc_format == "txt":
        rfc_name = f"rfc{rfc_number}.txt"
    else:
        raise TypeError("Please specify correct format to download (html or txt)")

    rfc_url = BASE_RFC_URL + rfc_name
    output_file = RAW_PATH_DATASET / doc_format / rfc_name
    download_website(rfc_url, output_file, logger)


def move_dataset(from_dir: Path, to_dir: Path):
    source_dir = from_dir
    destination_dir = to_dir
    files_to_move = list(source_dir.glob("*"))

    for file_path in tqdm(files_to_move):
        if file_path.is_file():
            file_path.rename(destination_dir / file_path.name)

    print("All files moved successfully.")
