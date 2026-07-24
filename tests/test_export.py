from datetime import datetime, timezone

import pandas as pd

from src.pipeline.export import (
    export_all,
    stats_by_brand_model_sentiment,
    stats_by_month,
    stats_by_source,
    to_dataframe,
)
from src.storage.models import ClassifiedPost


def make_classified(**overrides) -> ClassifiedPost:
    defaults = dict(
        id=overrides.get("id", "id1"),
        source="automobile_propre",
        text="texte",
        url="https://x/1",
        scraped_at=datetime.now(timezone.utc),
        published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        pertinent_confort_thermique=True,
        pertinence_confidence="haute",
        vehicule_marque="Renault",
        vehicule_modele="Megane E-Tech",
        sentiment=2,
        sentiment_confidence="haute",
        classification_echec=False,
    )
    defaults.update(overrides)
    return ClassifiedPost(**defaults)


def test_stats_by_brand_model_sentiment_counts_relevant_posts():
    posts = [
        make_classified(id="1", vehicule_marque="Renault", vehicule_modele="Zoe", sentiment=2),
        make_classified(id="2", vehicule_marque="Renault", vehicule_modele="Zoe", sentiment=2),
        make_classified(id="3", vehicule_marque="Peugeot", vehicule_modele="308", sentiment=4),
    ]
    df = to_dataframe(posts)
    stats = stats_by_brand_model_sentiment(df)
    row = stats[(stats["vehicule_marque"] == "Renault") & (stats["sentiment"] == 2)]
    assert row["count"].iloc[0] == 2


def test_stats_excludes_non_relevant_and_failed_posts():
    posts = [
        make_classified(id="1", pertinent_confort_thermique=False),
        make_classified(id="2", classification_echec=True),
        make_classified(id="3"),
    ]
    df = to_dataframe(posts)
    stats = stats_by_source(df)
    assert stats["count"].sum() == 1


def test_stats_by_month_groups_by_publication_month():
    posts = [
        make_classified(id="1", published_at=datetime(2024, 1, 10, tzinfo=timezone.utc)),
        make_classified(id="2", published_at=datetime(2024, 1, 20, tzinfo=timezone.utc)),
        make_classified(id="3", published_at=datetime(2024, 2, 5, tzinfo=timezone.utc)),
    ]
    df = to_dataframe(posts)
    stats = stats_by_month(df)
    counts = dict(zip(stats["month"], stats["count"]))
    assert counts["2024-01"] == 2
    assert counts["2024-02"] == 1


def test_export_all_writes_four_parquet_files_with_unfiltered_and_filtered_counts(tmp_path):
    posts = [
        make_classified(id="1", vehicule_marque="Renault", vehicule_modele="Zoe", sentiment=2),
        make_classified(id="2", pertinent_confort_thermique=False),
        make_classified(id="3", classification_echec=True),
    ]
    classified_path = tmp_path / "classified.parquet"
    to_dataframe(posts).to_parquet(classified_path, index=False)

    output_dir = tmp_path / "exports"
    outputs = export_all(classified_path, output_dir)

    assert set(outputs.keys()) == {
        "verbatims",
        "stats_marque_modele_sentiment",
        "stats_source",
        "stats_mois",
    }
    for path in outputs.values():
        assert path.exists()

    verbatims = pd.read_parquet(outputs["verbatims"])
    assert len(verbatims) == 3

    stats_source = pd.read_parquet(outputs["stats_source"])
    assert stats_source["count"].sum() == 1

    stats_marque = pd.read_parquet(outputs["stats_marque_modele_sentiment"])
    assert stats_marque["count"].sum() == 1


def test_export_all_on_empty_classified_parquet_does_not_crash(tmp_path):
    classified_path = tmp_path / "classified_empty.parquet"
    to_dataframe([]).to_parquet(classified_path, index=False)

    output_dir = tmp_path / "exports_empty"
    outputs = export_all(classified_path, output_dir)

    for path in outputs.values():
        assert path.exists()

    assert len(pd.read_parquet(outputs["verbatims"])) == 0
    assert len(pd.read_parquet(outputs["stats_marque_modele_sentiment"])) == 0
    assert len(pd.read_parquet(outputs["stats_source"])) == 0
    assert len(pd.read_parquet(outputs["stats_mois"])) == 0
