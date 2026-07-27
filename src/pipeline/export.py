from pathlib import Path

import pandas as pd

from src.storage.models import ClassifiedPost


def to_dataframe(posts: list[ClassifiedPost]) -> pd.DataFrame:
    return pd.DataFrame([p.model_dump(mode="json") for p in posts])


def relevant_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        # An empty input (e.g. from to_dataframe([]) or an empty classified
        # parquet file) has zero columns, so indexing "pertinent_confort_thermique"
        # would raise KeyError. Return it unchanged rather than crash.
        return df
    return df[df["pertinent_confort_thermique"] & ~df["classification_echec"]]


def stats_by_brand_model_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    relevant = relevant_only(df)
    if relevant.empty:
        return pd.DataFrame(columns=["vehicule_marque", "vehicule_modele", "sentiment", "count"])
    return (
        relevant.groupby(["vehicule_marque", "vehicule_modele", "sentiment"])
        .size()
        .reset_index(name="count")
    )


def stats_by_source(df: pd.DataFrame) -> pd.DataFrame:
    relevant = relevant_only(df)
    if relevant.empty:
        return pd.DataFrame(columns=["source", "count"])
    return relevant.groupby("source").size().reset_index(name="count")


def stats_by_month(df: pd.DataFrame) -> pd.DataFrame:
    relevant = relevant_only(df).copy()
    if relevant.empty:
        return pd.DataFrame(columns=["month", "count"])
    relevant["published_at"] = pd.to_datetime(
        relevant["published_at"], format="ISO8601", utc=True
    ).dt.tz_localize(None)
    relevant["month"] = relevant["published_at"].dt.to_period("M").astype(str)
    return relevant.groupby("month").size().reset_index(name="count")


def export_all(classified_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(classified_path)

    outputs = {
        "verbatims": output_dir / "verbatims_classifies.parquet",
        "stats_marque_modele_sentiment": output_dir / "stats_marque_modele_sentiment.parquet",
        "stats_source": output_dir / "stats_source.parquet",
        "stats_mois": output_dir / "stats_mois.parquet",
    }
    df.to_parquet(outputs["verbatims"], index=False)
    stats_by_brand_model_sentiment(df).to_parquet(outputs["stats_marque_modele_sentiment"], index=False)
    stats_by_source(df).to_parquet(outputs["stats_source"], index=False)
    stats_by_month(df).to_parquet(outputs["stats_mois"], index=False)
    return outputs
