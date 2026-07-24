from datetime import datetime, timezone

import pytest

from src.pipeline.llm_classifier import LLMClassifier
from src.storage.models import RawPost, make_post_id


@pytest.mark.integration
def test_classify_against_real_ollama_returns_valid_schema():
    raw = RawPost(
        id=make_post_id("automobile_propre", "https://x/topic/999", "auteur", "2024-01-01"),
        source="automobile_propre",
        brand_source="Renault",
        model_source="Megane E-Tech",
        text="La buée sur le pare-brise ne part jamais même avec la clim à fond, c'est énervant.",
        author="auteur",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        url="https://x/topic/999",
        scraped_at=datetime.now(timezone.utc),
    )
    classifier = LLMClassifier()
    result = classifier.classify(raw)
    assert result.pertinent_confort_thermique is True
    assert 1 <= result.sentiment <= 5
