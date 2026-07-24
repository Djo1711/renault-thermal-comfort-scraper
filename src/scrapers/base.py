import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.utils.rate_limiter import wait as rate_limit_wait


class FetchError(Exception):
    pass


class HTTPStatusError(FetchError):
    def __init__(self, status_code: int, url: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code} sur {url}")


class BaseScraper:
    def __init__(
        self,
        source_name: str,
        min_delay: float = 3,
        max_delay: float = 8,
        timeout: float = 15,
        max_retries: int = 3,
        user_agent: str = "Mozilla/5.0 (compatible; ThermalComfortResearchBot/1.0)",
        logger: Optional[logging.Logger] = None,
        session: Optional[requests.Session] = None,
    ):
        self.source_name = source_name
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger(f"scraper.{source_name}")
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def fetch(self, url: str) -> BeautifulSoup:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                self.logger.warning(
                    "Tentative %d/%d échouée pour %s: %s", attempt, self.max_retries, url, exc
                )
                continue

            if 400 <= response.status_code < 500:
                raise HTTPStatusError(response.status_code, url)

            if response.status_code >= 500:
                last_error = FetchError(f"HTTP {response.status_code} sur {url}")
                self.logger.warning("Tentative %d/%d: %s", attempt, self.max_retries, last_error)
                continue

            rate_limit_wait(self.min_delay, self.max_delay)
            return BeautifulSoup(response.text, "lxml")

        raise FetchError(f"Échec après {self.max_retries} tentatives sur {url}") from last_error
