import logging

from src.utils.logger import get_logger


def test_child_logger_without_log_file_propagates_and_has_no_handler():
    logger = get_logger("scraper.test_child_propagates")
    assert logger.propagate is True
    assert logger.handlers == []


def test_logger_with_log_file_has_file_handler_and_writes(tmp_path):
    log_file = tmp_path / "test.log"
    logger = get_logger("scraper.test_root_with_handler", log_file=str(log_file))
    assert logger.propagate is False
    assert len(logger.handlers) == 1

    logger.info("hello")
    for handler in logger.handlers:
        handler.flush()

    assert "hello" in log_file.read_text(encoding="utf-8")
