import json
from pathlib import Path
from typing import Any


class CheckpointStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_progress(self, scope: str) -> dict[str, Any]:
        return self._data.get(scope, {"last_page": 0, "seen_ids": []})

    def update_progress(self, scope: str, last_page: int, seen_ids: set[str]) -> None:
        self._data[scope] = {"last_page": last_page, "seen_ids": sorted(seen_ids)}
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
