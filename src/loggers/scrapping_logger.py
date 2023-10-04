import logging

logging.basicConfig(
    filename="logs/scrapping.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

scrapping_logger = logging.getLogger()
