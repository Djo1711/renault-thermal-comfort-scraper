import json
from datetime import datetime, timezone

import pytest
import responses

from src.pipeline.llm_classifier import (
    LLMResponseInvalidError,
    LLMClassifier,
    OllamaUnavailableError,
)
from src.storage.models import RawPost, make_post_id


def make_raw_post() -> RawPost:
    return RawPost(
        id=make_post_id("automobile_propre", "https://x/topic/1", "auteur", "2024-01-01"),
        source="automobile_propre",
        brand_source="Renault",
        model_source="Megane E-Tech",
        text="j'ai toujours de la buée sur le pare-brise",
        author="auteur",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        url="https://x/topic/1",
        scraped_at=datetime.now(timezone.utc),
    )


VALID_LLM_JSON = {
    "pertinent_confort_thermique": True,
    "pertinence_confidence": "haute",
    "vehicule": {
        "marque": "Renault",
        "marque_confidence": "haute",
        "modele": "Megane E-Tech",
        "modele_confidence": "haute",
    },
    "type_probleme": ["buee_desembuage"],
    "type_probleme_confidence": "haute",
    "zone_vehicule": "pare-brise avant",
    "sentiment": 2,
    "sentiment_confidence": "haute",
    "resume_probleme": "Buée persistante",
    "extrait_source": "j'ai toujours de la buée",
}


@responses.activate
def test_classify_parses_valid_llm_response():
    responses.add(
        responses.POST,
        "http://localhost:11434/api/chat",
        json={"message": {"content": json.dumps(VALID_LLM_JSON)}},
        status=200,
    )
    classifier = LLMClassifier()
    result = classifier.classify(make_raw_post())
    assert result.pertinent_confort_thermique is True
    assert result.vehicule.marque == "Renault"
    assert result.type_probleme == ["buee_desembuage"]


@responses.activate
def test_classify_retries_once_on_invalid_json_then_succeeds():
    responses.add(
        responses.POST,
        "http://localhost:11434/api/chat",
        json={"message": {"content": "ceci n'est pas du json"}},
        status=200,
    )
    responses.add(
        responses.POST,
        "http://localhost:11434/api/chat",
        json={"message": {"content": json.dumps(VALID_LLM_JSON)}},
        status=200,
    )
    classifier = LLMClassifier(max_retries=2)
    result = classifier.classify(make_raw_post())
    assert result.pertinent_confort_thermique is True
    assert len(responses.calls) == 2


@responses.activate
def test_classify_raises_llm_response_invalid_after_max_retries():
    responses.add(
        responses.POST,
        "http://localhost:11434/api/chat",
        json={"message": {"content": "toujours invalide"}},
        status=200,
    )
    responses.add(
        responses.POST,
        "http://localhost:11434/api/chat",
        json={"message": {"content": "toujours invalide"}},
        status=200,
    )
    classifier = LLMClassifier(max_retries=2)
    with pytest.raises(LLMResponseInvalidError):
        classifier.classify(make_raw_post())


@responses.activate
def test_classify_raises_llm_response_invalid_when_http_envelope_itself_is_not_json():
    # Regression test: a 200 response whose BODY is not valid JSON at all (not
    # just a malformed "content" field inside a valid envelope) previously
    # raised an unhandled JSONDecodeError from response.json(), propagating out
    # of classify() instead of being treated as one invalid-response attempt.
    responses.add(
        responses.POST,
        "http://localhost:11434/api/chat",
        body="not json at all",
        status=200,
        content_type="text/plain",
    )
    responses.add(
        responses.POST,
        "http://localhost:11434/api/chat",
        body="not json at all",
        status=200,
        content_type="text/plain",
    )
    classifier = LLMClassifier(max_retries=2)
    with pytest.raises(LLMResponseInvalidError):
        classifier.classify(make_raw_post())
    assert len(responses.calls) == 2


@responses.activate
def test_classify_raises_ollama_unavailable_immediately_without_retry():
    classifier = LLMClassifier(max_retries=2)
    with pytest.raises(OllamaUnavailableError):
        classifier.classify(make_raw_post())
    assert len(responses.calls) == 1
