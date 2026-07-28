from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.storage.models import (
    ClassifiedPost,
    LLMClassificationResult,
    RawPost,
    VehicleInfo,
    make_post_id,
)


def make_raw_post(**overrides) -> RawPost:
    defaults = dict(
        id=make_post_id("automobile_propre", "https://x/topic/1", "jdupont", "2024-01-01T00:00:00"),
        source="automobile_propre",
        brand_source="Renault",
        model_source=None,
        year_source=None,
        text="j'ai toujours de la buée sur le pare-brise",
        author="jdupont",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        url="https://x/topic/1",
        scraped_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return RawPost(**defaults)


def test_make_post_id_is_stable_and_deterministic():
    id1 = make_post_id("caradisiac", "https://x/avis/1", "auteur", "2024-01-01")
    id2 = make_post_id("caradisiac", "https://x/avis/1", "auteur", "2024-01-01")
    assert id1 == id2
    assert len(id1) == 16


def test_make_post_id_differs_on_different_input():
    id1 = make_post_id("caradisiac", "https://x/avis/1", "auteur", "2024-01-01")
    id2 = make_post_id("caradisiac", "https://x/avis/2", "auteur", "2024-01-01")
    assert id1 != id2


def test_llm_result_rejects_sentiment_out_of_range():
    with pytest.raises(ValidationError):
        LLMClassificationResult(
            pertinent_confort_thermique=True,
            pertinence_confidence="haute",
            vehicule=VehicleInfo(),
            sentiment=6,
            sentiment_confidence="haute",
        )


def test_llm_result_rejects_unknown_problem_type():
    with pytest.raises(ValidationError):
        LLMClassificationResult(
            pertinent_confort_thermique=True,
            pertinence_confidence="haute",
            vehicule=VehicleInfo(),
            type_probleme=["probleme_invente"],
            sentiment=3,
            sentiment_confidence="haute",
        )


def test_llm_result_accepts_valid_payload():
    result = LLMClassificationResult(
        pertinent_confort_thermique=True,
        pertinence_confidence="haute",
        vehicule=VehicleInfo(marque="Renault", marque_confidence="haute"),
        type_probleme=["buee_desembuage"],
        type_probleme_confidence="haute",
        zone_vehicule="pare-brise avant",
        sentiment=2,
        sentiment_confidence="haute",
        resume_probleme="Buée persistante",
        extrait_source="j'ai toujours de la buée",
    )
    assert result.sentiment == 2
    assert result.vehicule.marque == "Renault"


def test_llm_result_auto_corrects_pertinence_when_type_probleme_present():
    # Regression test: the LLM occasionally self-contradicts, returning a
    # non-empty type_probleme while pertinent_confort_thermique is False. The
    # SYSTEM_PROMPT already asks the model not to do this, but prompt-only
    # enforcement is probabilistic. This model_validator enforces the rule
    # structurally: it must auto-correct pertinent_confort_thermique to True
    # rather than raise, since raising would waste a retry on an otherwise
    # usable classification.
    result = LLMClassificationResult(
        pertinent_confort_thermique=False,
        pertinence_confidence="haute",
        vehicule=VehicleInfo(),
        type_probleme=["buee_desembuage"],
        type_probleme_confidence="haute",
        sentiment=3,
        sentiment_confidence="haute",
    )
    assert result.pertinent_confort_thermique is True


def test_classified_post_from_raw_and_llm_flattens_vehicle_fields():
    raw = make_raw_post(year_source="2022")
    llm = LLMClassificationResult(
        pertinent_confort_thermique=True,
        pertinence_confidence="haute",
        vehicule=VehicleInfo(marque="Renault", marque_confidence="haute", modele="Megane E-Tech"),
        type_probleme=["buee_desembuage"],
        type_probleme_confidence="haute",
        zone_vehicule="pare-brise avant",
        sentiment=2,
        sentiment_confidence="haute",
        resume_probleme="Buée persistante",
        extrait_source="j'ai toujours de la buée",
    )

    classified = ClassifiedPost.from_raw_and_llm(raw, llm)

    assert classified.id == raw.id
    assert classified.year_source == "2022"
    assert classified.vehicule_marque == "Renault"
    assert classified.vehicule_modele == "Megane E-Tech"
    assert classified.type_probleme == ["buee_desembuage"]
    assert classified.classification_echec is False


def test_classified_post_failed_marks_classification_echec():
    raw = make_raw_post()
    classified = ClassifiedPost.failed(raw)
    assert classified.classification_echec is True
    assert classified.pertinent_confort_thermique is False
    assert classified.id == raw.id
