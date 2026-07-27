import json
from typing import Optional

import requests

from src.storage.models import LLMClassificationResult, RawPost

SYSTEM_PROMPT = (
    "Tu es un assistant qui analyse des verbatims d'automobilistes français à propos du confort "
    "thermique de leur véhicule (chauffage, climatisation, ventilation, buée/désembuage, isolation "
    "thermique, odeur des systèmes de ventilation/climatisation, pompe à chaleur, préchauffage "
    "habitacle/batterie). "
    "Règle de cohérence OBLIGATOIRE entre les champs pertinent_confort_thermique et type_probleme : "
    "si tu identifies au moins un type de problème de confort thermique dans type_probleme, alors "
    "pertinent_confort_thermique DOIT être true. Si aucun type de problème de confort thermique "
    "n'est identifié, alors type_probleme doit être une liste vide ET pertinent_confort_thermique "
    "doit être false. Ces deux champs ne doivent jamais se contredire. "
    "Réponds STRICTEMENT au format JSON demandé, sans texte autour."
)

RESPONSE_SCHEMA = LLMClassificationResult.model_json_schema()


class LLMClassificationError(Exception):
    pass


class OllamaUnavailableError(LLMClassificationError):
    pass


class LLMResponseInvalidError(LLMClassificationError):
    pass


class LLMClassifier:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        max_retries: int = 2,
        timeout: float = 60,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()

    def _build_prompt(self, raw: RawPost) -> str:
        contexte = (
            f"marque={raw.brand_source or 'inconnu'}, "
            f"modèle={raw.model_source or 'inconnu'}, "
            f"année={raw.year_source or 'inconnue'}, "
            f"site={raw.source}"
        )
        return f'Texte du verbatim : "{raw.text}"\nContexte source : {contexte}'

    def classify(self, raw: RawPost) -> LLMClassificationResult:
        last_error: Optional[Exception] = None
        prompt = self._build_prompt(raw)

        for attempt in range(1, self.max_retries + 1):
            system = SYSTEM_PROMPT
            if attempt > 1:
                system += " Le format précédent était invalide, corrige-le."

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "format": RESPONSE_SCHEMA,
                "stream": False,
            }

            try:
                response = self.session.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise OllamaUnavailableError(
                    f"Ollama injoignable sur {self.base_url} — lance `ollama serve`"
                ) from exc

            try:
                content = response.json().get("message", {}).get("content", "")
                data = json.loads(content)
                return LLMClassificationResult(**data)
            except (json.JSONDecodeError, ValueError) as exc:
                # Covers both a malformed "content" field inside a valid JSON
                # envelope AND the envelope itself not being valid JSON at all
                # (requests.exceptions.JSONDecodeError subclasses
                # json.JSONDecodeError) -- either case counts as one invalid
                # attempt, not an unhandled crash.
                last_error = exc

        raise LLMResponseInvalidError(
            f"Réponse LLM invalide après {self.max_retries} tentatives pour le post {raw.id}"
        ) from last_error
