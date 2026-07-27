import logging
from pathlib import Path


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    if log_file:
        # logging.FileHandler does NOT create parent directories on its own --
        # on a fresh clone (e.g. "data/" not yet created), this crashed with
        # FileNotFoundError. Create the full parent chain first.
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    else:
        logger.propagate = True

    return logger
