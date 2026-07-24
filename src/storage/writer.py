import math
from pathlib import Path
from typing import Type, TypeVar

import pandas as pd
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_existing_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["id"])
    return set(df["id"].tolist())


def append_posts(path: str | Path, posts: list[BaseModel]) -> int:
    if not posts:
        return 0
    path = Path(path)
    new_rows = [p.model_dump(mode="json") for p in posts]
    new_df = pd.DataFrame(new_rows)

    if path.exists():
        existing_df = pd.read_parquet(path)
        existing_ids = set(existing_df["id"])
        new_df = new_df[~new_df["id"].isin(existing_ids)]
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return len(new_df)


def _clean_row(row: dict) -> dict:
    return {
        key: (None if isinstance(value, float) and math.isnan(value) else value)
        for key, value in row.items()
    }


def read_posts(path: str | Path, model: Type[T]) -> list[T]:
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return [model(**_clean_row(row)) for row in df.to_dict(orient="records")]
