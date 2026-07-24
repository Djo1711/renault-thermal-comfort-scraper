from datetime import datetime, timezone

from src.pipeline.export import (
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
