import logging


def get_custom_logger(name: str, log_path: str, format: str = None):
    if format:
        formatter = logging.Formatter(fmt=format)
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(handler)

    return logger
