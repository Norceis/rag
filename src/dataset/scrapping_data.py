from tqdm import tqdm
from dataset.scrapping_data_utils import download_rfc
import concurrent.futures
from loggers.loggers import get_custom_logger

scrapping_logger = get_custom_logger("scrapping_logger", "logs/scrapping.log")

how_many_docs = 9500
doc_format = "txt"
args_iterable = (
    (rfc_number, scrapping_logger, doc_format) for rfc_number in range(how_many_docs)
)

with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
    for _ in tqdm(
        executor.map(lambda args: download_rfc(*args), args_iterable),
        total=how_many_docs,
    ):
        pass

scrapping_logger.info("All RFC documents downloaded.")
