import logging


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    if log_file:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    else:
        logger.propagate = True

    return logger
