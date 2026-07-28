import logging

from src.utils.logger import get_logger


def test_child_logger_without_log_file_propagates_and_has_no_handler():
    logger = get_logger("scraper.test_child_propagates")
    assert logger.propagate is True
    assert logger.handlers == []


def test_get_logger_creates_missing_parent_directories_for_log_file(tmp_path):
    # Regression test: logging.FileHandler does NOT create parent
    # directories. On a fresh clone, "data/" doesn't exist yet (gitignored,
    # no .gitkeep), so get_logger("scraper", log_file="data/scraper.log")
    # crashed with FileNotFoundError immediately on scrape()/classify(). Use
    # a deeply nested missing path to prove get_logger creates the full
    # parent chain, not just the immediate directory.
    log_file = tmp_path / "nested" / "dir" / "test.log"
    assert not log_file.parent.exists()

    logger = get_logger("scraper.test_nested_dir_creation", log_file=str(log_file))
    logger.info("hello nested")
    for handler in logger.handlers:
        handler.flush()

    assert log_file.parent.exists()
    assert "hello nested" in log_file.read_text(encoding="utf-8")


def test_logger_with_log_file_has_file_handler_and_writes(tmp_path):
    log_file = tmp_path / "test.log"
    logger = get_logger("scraper.test_root_with_handler", log_file=str(log_file))
    assert logger.propagate is False
    assert len(logger.handlers) == 1

    logger.info("hello")
    for handler in logger.handlers:
        handler.flush()

    assert "hello" in log_file.read_text(encoding="utf-8")
