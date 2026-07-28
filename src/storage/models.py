import hashlib
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Confidence = Literal["haute", "moyenne", "basse"]

PROBLEM_TYPES = [
    "chauffage",
    "climatisation",
    "ventilation",
    "buee_desembuage",
    "isolation_thermique",
    "odeur_hvac",
    "pompe_a_chaleur",
    "prechauffage",
    "autre",
]

ProblemType = Literal[
    "chauffage",
    "climatisation",
    "ventilation",
    "buee_desembuage",
    "isolation_thermique",
    "odeur_hvac",
    "pompe_a_chaleur",
    "prechauffage",
    "autre",
]


def make_post_id(source: str, url: str, author: str, published_at: str) -> str:
    raw = f"{source}|{url}|{author}|{published_at}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class RawPost(BaseModel):
    id: str
    source: Literal["automobile_propre", "caradisiac"]
    brand_source: Optional[str] = None
    model_source: Optional[str] = None
    year_source: Optional[str] = None
    text: str
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    url: str
    scraped_at: datetime


class VehicleInfo(BaseModel):
    marque: Optional[str] = None
    marque_confidence: Optional[Confidence] = None
    modele: Optional[str] = None
    modele_confidence: Optional[Confidence] = None
    annee: Optional[int] = None
    annee_confidence: Optional[Confidence] = None
    motorisation: Optional[str] = None
    motorisation_confidence: Optional[Confidence] = None


class LLMClassificationResult(BaseModel):
    pertinent_confort_thermique: bool
    pertinence_confidence: Confidence
    vehicule: VehicleInfo
    type_probleme: list[ProblemType] = []
    type_probleme_confidence: Optional[Confidence] = None
    zone_vehicule: Optional[str] = None
    sentiment: int = Field(ge=1, le=5)
    sentiment_confidence: Confidence
    resume_probleme: Optional[str] = None
    extrait_source: Optional[str] = None

    @model_validator(mode="after")
    def _coherence_pertinence_type_probleme(self) -> "LLMClassificationResult":
        # Enforce programmatically what the SYSTEM_PROMPT only asks for
        # probabilistically: if the model identified at least one problem
        # type, it must have also flagged the post as relevant. We
        # auto-correct rather than raise, since the rest of the response is
        # otherwise usable and raising would waste a retry attempt on an
        # otherwise-good classification (see llm_classifier.py's retry loop).
        if self.type_probleme and not self.pertinent_confort_thermique:
            self.pertinent_confort_thermique = True
        return self


class ClassifiedPost(BaseModel):
    id: str
    source: Literal["automobile_propre", "caradisiac"]
    brand_source: Optional[str] = None
    model_source: Optional[str] = None
    year_source: Optional[str] = None
    text: str
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    url: str
    scraped_at: datetime

    pertinent_confort_thermique: bool
    pertinence_confidence: Confidence
    vehicule_marque: Optional[str] = None
    vehicule_marque_confidence: Optional[Confidence] = None
    vehicule_modele: Optional[str] = None
    vehicule_modele_confidence: Optional[Confidence] = None
    vehicule_annee: Optional[int] = None
    vehicule_annee_confidence: Optional[Confidence] = None
    vehicule_motorisation: Optional[str] = None
    vehicule_motorisation_confidence: Optional[Confidence] = None
    type_probleme: list[str] = []
    type_probleme_confidence: Optional[Confidence] = None
    zone_vehicule: Optional[str] = None
    sentiment: int
    sentiment_confidence: Confidence
    resume_probleme: Optional[str] = None
    extrait_source: Optional[str] = None
    classification_echec: bool = False

    @classmethod
    def from_raw_and_llm(cls, raw: RawPost, llm: LLMClassificationResult) -> "ClassifiedPost":
        return cls(
            id=raw.id,
            source=raw.source,
            brand_source=raw.brand_source,
            model_source=raw.model_source,
            year_source=raw.year_source,
            text=raw.text,
            author=raw.author,
            published_at=raw.published_at,
            url=raw.url,
            scraped_at=raw.scraped_at,
            pertinent_confort_thermique=llm.pertinent_confort_thermique,
            pertinence_confidence=llm.pertinence_confidence,
            vehicule_marque=llm.vehicule.marque,
            vehicule_marque_confidence=llm.vehicule.marque_confidence,
            vehicule_modele=llm.vehicule.modele,
            vehicule_modele_confidence=llm.vehicule.modele_confidence,
            vehicule_annee=llm.vehicule.annee,
            vehicule_annee_confidence=llm.vehicule.annee_confidence,
            vehicule_motorisation=llm.vehicule.motorisation,
            vehicule_motorisation_confidence=llm.vehicule.motorisation_confidence,
            type_probleme=llm.type_probleme,
            type_probleme_confidence=llm.type_probleme_confidence,
            zone_vehicule=llm.zone_vehicule,
            sentiment=llm.sentiment,
            sentiment_confidence=llm.sentiment_confidence,
            resume_probleme=llm.resume_probleme,
            extrait_source=llm.extrait_source,
            classification_echec=False,
        )

    @classmethod
    def failed(cls, raw: RawPost) -> "ClassifiedPost":
        return cls(
            id=raw.id,
            source=raw.source,
            brand_source=raw.brand_source,
            model_source=raw.model_source,
            year_source=raw.year_source,
            text=raw.text,
            author=raw.author,
            published_at=raw.published_at,
            url=raw.url,
            scraped_at=raw.scraped_at,
            pertinent_confort_thermique=False,
            pertinence_confidence="basse",
            sentiment=3,
            sentiment_confidence="basse",
            classification_echec=True,
        )
