from pathlib import Path

from src.utils.config import load_yaml


class KeywordPrefilter:
    def __init__(self, keywords_path: str | Path):
        data = load_yaml(keywords_path)
        self.categories: dict[str, list[str]] = data["categories"]

    def is_relevant(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            keyword.lower() in lowered
            for keywords in self.categories.values()
            for keyword in keywords
        )
