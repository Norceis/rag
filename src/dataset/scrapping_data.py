from tqdm import tqdm
from dataset.utils import download_rfc
import concurrent.futures
from loggers.scrapping_logger import scrapping_logger

rfc_numbers = range(9500)
args_iterable = ((rfc_number, scrapping_logger) for rfc_number in rfc_numbers)

with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
    for _ in tqdm(executor.map(lambda args: download_rfc(*args), args_iterable), total=9500):
        pass

scrapping_logger.info("All RFC documents downloaded.")