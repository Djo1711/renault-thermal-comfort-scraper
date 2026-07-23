# Pipeline scraping + classification LLM confort thermique — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruire from scratch un pipeline Python qui scrape des verbatims auto sur 2 sources (forums.automobile-propre.com, caradisiac.com/avis) filtrés sur 15 marques ciblées, les classifie via un LLM local (Ollama/Qwen2.5) selon un schéma JSON structuré (pertinence confort thermique, véhicule, type de problème, sentiment), puis exporte un parquet + des stats.

**Architecture:** Trois commandes CLI indépendantes (`scrape`, `classify`, `export`) communiquant via des fichiers parquet sur disque, chacune reprenable seule. Scraping avec checkpoint JSON + dédup par id stable ; pré-filtre keywords bon marché avant l'appel LLM ; classification LLM avec structured output + retry ; export pandas → parquet + stats.

**Tech Stack:** Python 3.11+, requests + BeautifulSoup4 + lxml, pydantic v2, click + rich, pyyaml, pandas + pyarrow, Ollama (Qwen2.5 7B Instruct). Tests : pytest + `responses` (mock HTTP) + fixtures HTML/JSON réelles.

## Global Constraints

- Python 3.11+ uniquement, pas de dépendance à un OS spécifique en dehors d'Ollama en local.
- Rate-limiting scraping : délai aléatoire 3-8 secondes entre requêtes HTTP (jamais moins).
- 15 marques ciblées (exactes, `config/vehicle_list.yaml`) : Renault, Peugeot, Citroën, Dacia, Volkswagen, Toyota, Kia, Hyundai, Audi, BMW, Mercedes, Cupra, Seat, Skoda, Opel.
- Modèle LLM : `qwen2.5:7b-instruct` via Ollama sur `http://localhost:11434` (le 14B est écarté, machine à 16 Go de RAM unifiée).
- Stockage intermédiaire et final en parquet (pas de CSV), colonnes à plat (pas de JSON imbriqué).
- Tests unitaires sans appel réseau réel (fixtures HTML/JSON + mocks HTTP via `responses`) ; un seul test d'intégration marqué `@pytest.mark.integration`, skip par défaut (`pytest.ini` : `addopts = -m "not integration"`).
- TDD strict pour tout module de logique (test avant code) sur l'ensemble des tâches ci-dessous.
- Chaque tâche se termine par un commit git séparé (sauf la tâche d'installation Ollama, purement opérationnelle).

---

### Task 1: Scaffolding du projet (dépendances, config, structure de packages)

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.gitignore`
- Create: `pytest.ini`
- Create: `config/settings.yaml`
- Create: `config/vehicle_list.yaml`
- Create: `config/keywords.yaml`
- Create: `src/__init__.py`, `src/utils/__init__.py`, `src/scrapers/__init__.py`, `src/storage/__init__.py`, `src/pipeline/__init__.py`
- Create: `src/utils/config.py`
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/fixtures/` (dossier)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_yaml(path: str | Path) -> dict` dans `src/utils/config.py`, utilisé par toutes les tâches suivantes qui lisent une config YAML.
- Produces: `load_fixture(name: str) -> str` dans `tests/conftest.py`, utilisé par toutes les tâches de test de scrapers.

- [ ] **Step 1: Créer les fichiers de dépendances**

`requirements.txt` :
```
requests>=2.31
beautifulsoup4>=4.12
lxml>=5.0
pydantic>=2.6
click>=8.1
rich>=13.7
pyyaml>=6.0
pandas>=2.2
pyarrow>=15.0
```

`requirements-dev.txt` :
```
-r requirements.txt
pytest>=8.0
pytest-mock>=3.14
responses>=0.25
```

- [ ] **Step 2: Créer `.gitignore`**

```
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
env/
data/raw/
data/processed/*.parquet
data/checkpoints/
data/output/
*.log
.env
.env.local
.DS_Store
.pytest_cache/
.mypy_cache/
```

- [ ] **Step 3: Créer `pytest.ini`**

```ini
[pytest]
testpaths = tests
markers =
    integration: tests nécessitant un serveur Ollama local (skip par défaut)
addopts = -m "not integration"
```

- [ ] **Step 4: Créer les fichiers de config YAML**

`config/settings.yaml` :
```yaml
rate_limit:
  min_seconds: 3
  max_seconds: 8
http:
  timeout_seconds: 15
  max_retries: 3
  user_agent: "Mozilla/5.0 (compatible; ThermalComfortResearchBot/1.0)"
ollama:
  base_url: "http://localhost:11434"
  model: "qwen2.5:7b-instruct"
  max_retries: 2
paths:
  raw_posts: "data/processed/posts_raw.parquet"
  classified_posts: "data/processed/posts_classified.parquet"
  checkpoint: "data/checkpoints/progress.json"
  output_dir: "data/output"
```

`config/vehicle_list.yaml` :
```yaml
brands:
  - Renault
  - Peugeot
  - Citroën
  - Dacia
  - Volkswagen
  - Toyota
  - Kia
  - Hyundai
  - Audi
  - BMW
  - Mercedes
  - Cupra
  - Seat
  - Skoda
  - Opel
```

`config/keywords.yaml` :
```yaml
categories:
  chauffage:
    - chauffage
    - "ne chauffe pas"
    - "froid dans l'habitacle"
  climatisation:
    - climatisation
    - "clim "
    - climatiseur
  ventilation:
    - ventilation
    - aération
    - soufflerie
  buee_desembuage:
    - buée
    - désembuage
    - embué
    - embuée
  isolation_thermique:
    - "isolation thermique"
    - "déperdition de chaleur"
    - "pont thermique"
  odeur_hvac:
    - "odeur de moisi"
    - "mauvaise odeur clim"
    - "odeur ventilation"
  pompe_a_chaleur:
    - "pompe à chaleur"
    - " pac "
  prechauffage:
    - préchauffage
    - préconditionnement
    - "préchauffer la batterie"
    - "préchauffer l'habitacle"
```

- [ ] **Step 5: Créer la structure de packages Python**

```bash
mkdir -p src/utils src/scrapers src/storage src/pipeline tests/fixtures data/processed data/checkpoints data/output
touch src/__init__.py src/utils/__init__.py src/scrapers/__init__.py src/storage/__init__.py src/pipeline/__init__.py tests/__init__.py
```

- [ ] **Step 6: Écrire le test du config loader (`tests/test_config.py`)**

```python
from src.utils.config import load_yaml


def test_load_yaml_returns_dict(tmp_path):
    p = tmp_path / "sample.yaml"
    p.write_text("a: 1\nb: two\n", encoding="utf-8")
    data = load_yaml(p)
    assert data == {"a": 1, "b": "two"}


def test_load_yaml_reads_project_settings():
    data = load_yaml("config/settings.yaml")
    assert data["rate_limit"]["min_seconds"] == 3
    assert data["ollama"]["model"] == "qwen2.5:7b-instruct"
```

- [ ] **Step 7: Lancer les tests, vérifier qu'ils échouent**

Run: `pytest tests/test_config.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'src.utils.config'`

- [ ] **Step 8: Implémenter `src/utils/config.py`**

```python
from pathlib import Path


import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 9: Créer `tests/conftest.py` (helper de fixtures)**

```python
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")
```

- [ ] **Step 10: Installer les dépendances et lancer les tests**

Run:
```bash
pip install -r requirements-dev.txt
pytest tests/test_config.py -v
```
Expected: `2 passed`

- [ ] **Step 11: Commit**

```bash
git add requirements.txt requirements-dev.txt .gitignore pytest.ini config/ src/ tests/
git commit -m "Scaffolding: dépendances, config YAML, structure de packages"
```

---

### Task 2: Logger utilitaire

**Files:**
- Create: `src/utils/logger.py`
- Test: `tests/test_logger.py`

**Interfaces:**
- Produces: `get_logger(name: str, log_file: str | None = None) -> logging.Logger` — utilisé par les scrapers (Task 7) et le CLI (Task 13).

- [ ] **Step 1: Écrire le test**

```python
import logging

from src.utils.logger import get_logger


def test_child_logger_without_log_file_propagates_and_has_no_handler():
    logger = get_logger("scraper.test_child_propagates")
    assert logger.propagate is True
    assert logger.handlers == []


def test_logger_with_log_file_has_file_handler_and_writes(tmp_path):
    log_file = tmp_path / "test.log"
    logger = get_logger("scraper.test_root_with_handler", log_file=str(log_file))
    assert logger.propagate is False
    assert len(logger.handlers) == 1

    logger.info("hello")
    for handler in logger.handlers:
        handler.flush()

    assert "hello" in log_file.read_text(encoding="utf-8")
```

- [ ] **Step 2: Lancer le test, vérifier qu'il échoue**

Run: `pytest tests/test_logger.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'src.utils.logger'`

- [ ] **Step 3: Implémenter `src/utils/logger.py`**

```python
import logging


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    if log_file:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    else:
        logger.propagate = True

    return logger
```

- [ ] **Step 4: Lancer le test, vérifier qu'il passe**

Run: `pytest tests/test_logger.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/utils/logger.py tests/test_logger.py
git commit -m "Ajoute le logger utilitaire (propagation des loggers scraper.* sans handler)"
```

---

### Task 3: Rate limiter

**Files:**
- Create: `src/utils/rate_limiter.py`
- Test: `tests/test_rate_limiter.py`

**Interfaces:**
- Produces: `wait(min_seconds: float = 3, max_seconds: float = 8) -> float` — utilisé par `BaseScraper.fetch` (Task 7).

- [ ] **Step 1: Écrire le test**

```python
from unittest.mock import patch

from src.utils.rate_limiter import wait


def test_wait_sleeps_for_random_delay_within_bounds():
    with patch("src.utils.rate_limiter.random.uniform", return_value=5.0) as mock_uniform, \
         patch("src.utils.rate_limiter.time.sleep") as mock_sleep:
        delay = wait(min_seconds=3, max_seconds=8)

    mock_uniform.assert_called_once_with(3, 8)
    mock_sleep.assert_called_once_with(5.0)
    assert delay == 5.0
```

- [ ] **Step 2: Lancer le test, vérifier qu'il échoue**

Run: `pytest tests/test_rate_limiter.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'src.utils.rate_limiter'`

- [ ] **Step 3: Implémenter `src/utils/rate_limiter.py`**

```python
import random
import time


def wait(min_seconds: float = 3, max_seconds: float = 8) -> float:
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)
    return delay
```

- [ ] **Step 4: Lancer le test, vérifier qu'il passe**

Run: `pytest tests/test_rate_limiter.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/utils/rate_limiter.py tests/test_rate_limiter.py
git commit -m "Ajoute le rate limiter (délai aléatoire 3-8s)"
```

---

### Task 4: Checkpoint store

**Files:**
- Create: `src/utils/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Produces: `CheckpointStore(path: str | Path)` avec `.get_progress(scope: str) -> dict` (retourne `{"last_page": int, "seen_ids": list[str]}`) et `.update_progress(scope: str, last_page: int, seen_ids: set[str]) -> None` (persiste immédiatement sur disque). Utilisé par les scrapers (Task 8, 9).

- [ ] **Step 1: Écrire le test**

```python
from src.utils.checkpoint import CheckpointStore


def test_new_checkpoint_returns_empty_progress(tmp_path):
    store = CheckpointStore(tmp_path / "progress.json")
    progress = store.get_progress("automobile_propre:renault:megane-e-tech")
    assert progress == {"last_page": 0, "seen_ids": []}


def test_update_progress_persists_to_disk_and_is_reloadable(tmp_path):
    checkpoint_path = tmp_path / "progress.json"
    store = CheckpointStore(checkpoint_path)
    store.update_progress(
        "automobile_propre:renault:megane-e-tech", last_page=2, seen_ids={"id1", "id2"}
    )

    reloaded = CheckpointStore(checkpoint_path)
    progress = reloaded.get_progress("automobile_propre:renault:megane-e-tech")
    assert progress["last_page"] == 2
    assert sorted(progress["seen_ids"]) == ["id1", "id2"]


def test_different_scopes_are_independent(tmp_path):
    store = CheckpointStore(tmp_path / "progress.json")
    store.update_progress("scope_a", last_page=1, seen_ids={"a"})
    store.update_progress("scope_b", last_page=3, seen_ids={"b"})

    assert store.get_progress("scope_a")["last_page"] == 1
    assert store.get_progress("scope_b")["last_page"] == 3
```

- [ ] **Step 2: Lancer le test, vérifier qu'il échoue**

Run: `pytest tests/test_checkpoint.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'src.utils.checkpoint'`

- [ ] **Step 3: Implémenter `src/utils/checkpoint.py`**

```python
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
```

- [ ] **Step 4: Lancer le test, vérifier qu'il passe**

Run: `pytest tests/test_checkpoint.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/utils/checkpoint.py tests/test_checkpoint.py
git commit -m "Ajoute le checkpoint store (reprise scraping par scope: marque/sous-forum)"
```

---

### Task 5: Modèles de données pydantic (RawPost, ClassifiedPost, schéma LLM)

**Files:**
- Create: `src/storage/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `make_post_id(source: str, url: str, author: str, published_at: str) -> str`
- Produces: `RawPost` (champs : `id, source, brand_source, model_source, year_source, text, author, published_at, url, scraped_at`)
- Produces: `VehicleInfo`, `LLMClassificationResult` (schéma JSON exact attendu du LLM)
- Produces: `ClassifiedPost` avec `.from_raw_and_llm(raw: RawPost, llm: LLMClassificationResult) -> ClassifiedPost` et `.failed(raw: RawPost) -> ClassifiedPost`
- Produces: `PROBLEM_TYPES: list[str]` (catégories reprises de `config/keywords.yaml`)
- Consumes: rien (module de base, pas de dépendance interne)

- [ ] **Step 1: Écrire les tests**

```python
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
```

- [ ] **Step 2: Lancer les tests, vérifier qu'ils échouent**

Run: `pytest tests/test_models.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'src.storage.models'`

- [ ] **Step 3: Implémenter `src/storage/models.py`**

```python
import hashlib
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

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
    type_probleme: list[str] = []
    type_probleme_confidence: Optional[Confidence] = None
    zone_vehicule: Optional[str] = None
    sentiment: int
    sentiment_confidence: Confidence
    resume_probleme: Optional[str] = None
    extrait_source: Optional[str] = None

    @field_validator("sentiment")
    @classmethod
    def sentiment_in_range(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError("sentiment must be between 1 and 5")
        return v

    @field_validator("type_probleme")
    @classmethod
    def type_probleme_known(cls, v: list[str]) -> list[str]:
        unknown = [t for t in v if t not in PROBLEM_TYPES]
        if unknown:
            raise ValueError(f"type_probleme inconnu: {unknown}")
        return v


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
```

- [ ] **Step 4: Lancer les tests, vérifier qu'ils passent**

Run: `pytest tests/test_models.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/storage/models.py tests/test_models.py
git commit -m "Ajoute les modèles pydantic (RawPost, ClassifiedPost, schéma JSON LLM)"
```

---

### Task 6: Storage writer (parquet, dédup, reprise)

**Files:**
- Create: `src/storage/writer.py`
- Test: `tests/test_writer.py`

**Interfaces:**
- Consumes: `RawPost`, `ClassifiedPost` (Task 5)
- Produces: `append_posts(path: str | Path, posts: list[BaseModel]) -> int` (retourne le nombre de lignes réellement ajoutées), `read_existing_ids(path: str | Path) -> set[str]`, `read_posts(path: str | Path, model: type) -> list` — utilisés par le CLI (Task 13) et les scrapers.

- [ ] **Step 1: Écrire les tests**

```python
from datetime import datetime, timezone

from src.storage.models import RawPost, make_post_id
from src.storage.writer import append_posts, read_existing_ids, read_posts


def make_post(post_id_suffix: str) -> RawPost:
    return RawPost(
        id=make_post_id("automobile_propre", f"https://x/topic/{post_id_suffix}", "auteur", "2024-01-01"),
        source="automobile_propre",
        brand_source="Renault",
        text=f"texte {post_id_suffix}",
        author="auteur",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        url=f"https://x/topic/{post_id_suffix}",
        scraped_at=datetime.now(timezone.utc),
    )


def test_append_posts_writes_new_file(tmp_path):
    path = tmp_path / "posts_raw.parquet"
    written = append_posts(path, [make_post("1"), make_post("2")])
    assert written == 2
    assert path.exists()
    assert read_existing_ids(path) == {make_post("1").id, make_post("2").id}


def test_append_posts_skips_already_present_ids(tmp_path):
    path = tmp_path / "posts_raw.parquet"
    append_posts(path, [make_post("1")])
    written = append_posts(path, [make_post("1"), make_post("2")])
    assert written == 1
    assert len(read_posts(path, RawPost)) == 2


def test_read_existing_ids_empty_when_no_file(tmp_path):
    assert read_existing_ids(tmp_path / "missing.parquet") == set()


def test_read_posts_reconstructs_pydantic_models(tmp_path):
    path = tmp_path / "posts_raw.parquet"
    append_posts(path, [make_post("1")])
    posts = read_posts(path, RawPost)
    assert len(posts) == 1
    assert isinstance(posts[0], RawPost)
    assert posts[0].id == make_post("1").id
    assert posts[0].brand_source == "Renault"


def test_read_posts_empty_list_when_no_file(tmp_path):
    assert read_posts(tmp_path / "missing.parquet", RawPost) == []
```

- [ ] **Step 2: Lancer les tests, vérifier qu'ils échouent**

Run: `pytest tests/test_writer.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'src.storage.writer'`

- [ ] **Step 3: Implémenter `src/storage/writer.py`**

```python
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
```

- [ ] **Step 4: Lancer les tests, vérifier qu'ils passent**

Run: `pytest tests/test_writer.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/storage/writer.py tests/test_writer.py
git commit -m "Ajoute le writer parquet (dédup par id, reprise, reconstruction pydantic)"
```

---

### Task 7: Base scraper (HTTP, rate-limiting, retry)

**Files:**
- Create: `src/scrapers/base.py`
- Test: `tests/test_base_scraper.py`

**Interfaces:**
- Consumes: `wait()` de `src/utils/rate_limiter.py` (Task 3)
- Produces: `BaseScraper(source_name, min_delay=3, max_delay=8, timeout=15, max_retries=3, user_agent=..., logger=None, session=None)` avec `.fetch(url: str) -> BeautifulSoup`. Produces aussi `FetchError` et `HTTPStatusError(status_code, url)` (sous-classe de `FetchError`, levée immédiatement sans retry sur 4xx). Utilisé par `AutomobilePropreScraper` et `CaradisiacScraper` (Task 8, 9).

- [ ] **Step 1: Écrire les tests**

```python
import responses

from src.scrapers.base import BaseScraper, FetchError, HTTPStatusError


@responses.activate
def test_fetch_returns_parsed_soup_on_success():
    responses.add(
        responses.GET,
        "https://example.test/page",
        body="<html><body><p>ok</p></body></html>",
        status=200,
    )
    scraper = BaseScraper("test_source", min_delay=0, max_delay=0)
    soup = scraper.fetch("https://example.test/page")
    assert soup.find("p").text == "ok"


@responses.activate
def test_fetch_raises_http_status_error_on_400_without_retry():
    responses.add(responses.GET, "https://example.test/page", status=400)
    scraper = BaseScraper("test_source", min_delay=0, max_delay=0, max_retries=3)

    try:
        scraper.fetch("https://example.test/page")
        assert False, "aurait dû lever HTTPStatusError"
    except HTTPStatusError as exc:
        assert exc.status_code == 400

    assert len(responses.calls) == 1


@responses.activate
def test_fetch_retries_on_500_then_succeeds():
    responses.add(responses.GET, "https://example.test/page", status=500)
    responses.add(
        responses.GET,
        "https://example.test/page",
        body="<html><body><p>ok</p></body></html>",
        status=200,
    )
    scraper = BaseScraper("test_source", min_delay=0, max_delay=0, max_retries=3)
    soup = scraper.fetch("https://example.test/page")
    assert soup.find("p").text == "ok"
    assert len(responses.calls) == 2


@responses.activate
def test_fetch_raises_after_max_retries_on_persistent_500():
    responses.add(responses.GET, "https://example.test/page", status=500)
    responses.add(responses.GET, "https://example.test/page", status=500)
    scraper = BaseScraper("test_source", min_delay=0, max_delay=0, max_retries=2)

    try:
        scraper.fetch("https://example.test/page")
        assert False, "aurait dû lever FetchError"
    except FetchError:
        pass

    assert len(responses.calls) == 2
```

- [ ] **Step 2: Lancer les tests, vérifier qu'ils échouent**

Run: `pytest tests/test_base_scraper.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'src.scrapers.base'`

- [ ] **Step 3: Implémenter `src/scrapers/base.py`**

```python
import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.utils.rate_limiter import wait as rate_limit_wait


class FetchError(Exception):
    pass


class HTTPStatusError(FetchError):
    def __init__(self, status_code: int, url: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code} sur {url}")


class BaseScraper:
    def __init__(
        self,
        source_name: str,
        min_delay: float = 3,
        max_delay: float = 8,
        timeout: float = 15,
        max_retries: int = 3,
        user_agent: str = "Mozilla/5.0 (compatible; ThermalComfortResearchBot/1.0)",
        logger: Optional[logging.Logger] = None,
        session: Optional[requests.Session] = None,
    ):
        self.source_name = source_name
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger(f"scraper.{source_name}")
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def fetch(self, url: str) -> BeautifulSoup:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                self.logger.warning(
                    "Tentative %d/%d échouée pour %s: %s", attempt, self.max_retries, url, exc
                )
                continue

            if 400 <= response.status_code < 500:
                raise HTTPStatusError(response.status_code, url)

            if response.status_code >= 500:
                last_error = FetchError(f"HTTP {response.status_code} sur {url}")
                self.logger.warning("Tentative %d/%d: %s", attempt, self.max_retries, last_error)
                continue

            rate_limit_wait(self.min_delay, self.max_delay)
            return BeautifulSoup(response.text, "lxml")

        raise FetchError(f"Échec après {self.max_retries} tentatives sur {url}") from last_error
```

- [ ] **Step 4: Lancer les tests, vérifier qu'ils passent**

Run: `pytest tests/test_base_scraper.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/scrapers/base.py tests/test_base_scraper.py
git commit -m "Ajoute BaseScraper (fetch avec retry sur 5xx, HTTPStatusError immédiat sur 4xx)"
```

---

### Task 8: Scraper automobile-propre.com

**Files:**
- Create: `src/scrapers/automobile_propre.py`
- Create fixtures: `tests/fixtures/automobile_propre_forums_list.html`, `tests/fixtures/automobile_propre_brand_page.html`, `tests/fixtures/automobile_propre_topic_list_page1.html`, `tests/fixtures/automobile_propre_topic_list_page2_repeat.html`, `tests/fixtures/automobile_propre_topic_page.html`
- Test: `tests/test_automobile_propre.py`

**Interfaces:**
- Consumes: `BaseScraper`, `FetchError`, `HTTPStatusError` (Task 7) ; `CheckpointStore` (Task 4) ; `RawPost`, `make_post_id` (Task 5) ; `load_fixture` (Task 1)
- Produces: `AutomobilePropreScraper(target_brands: list[str], checkpoint_store: CheckpointStore, **kwargs)` avec `.discover_brand_forums() -> dict[str, str]`, `.discover_subforums(brand_url: str) -> dict[str, str]`, `.iter_topic_urls(subforum_url: str, scope: str) -> Iterator[str]`, `.scrape_topic(topic_url: str, brand: str, model: str | None) -> list[RawPost]`. Utilisé par le CLI (Task 13).

- [ ] **Step 1: Créer les fixtures HTML**

`tests/fixtures/automobile_propre_forums_list.html` :
```html
<!DOCTYPE html>
<html>
<body>
<div class="ipsDataList">
  <h4 data-ips-hook="title"><a href="https://forums.automobile-propre.com/forums/forum/12-renault/">Renault</a></h4>
  <h4 data-ips-hook="title"><a href="https://forums.automobile-propre.com/forums/forum/34-tesla/">Tesla</a></h4>
  <h4 data-ips-hook="title"><a href="https://forums.automobile-propre.com/forums/forum/56-peugeot/">Peugeot</a></h4>
</div>
</body>
</html>
```

`tests/fixtures/automobile_propre_brand_page.html` :
```html
<!DOCTYPE html>
<html>
<body>
<div class="ipsDataList">
  <h4 data-ips-hook="title"><a href="https://forums.automobile-propre.com/forums/forum/45-megane-e-tech/">Megane E-Tech</a></h4>
  <h4 data-ips-hook="title"><a href="https://forums.automobile-propre.com/forums/forum/46-zoe/">Zoe</a></h4>
</div>
</body>
</html>
```

`tests/fixtures/automobile_propre_topic_list_page1.html` :
```html
<!DOCTYPE html>
<html>
<body>
<div class="ipsDataList">
  <h4 data-ips-hook="title"><a href="https://forums.automobile-propre.com/forums/topic/1001-buee-persistante/">Buée persistante sur pare-brise</a></h4>
  <h4 data-ips-hook="title"><a href="https://forums.automobile-propre.com/forums/topic/1002-autonomie-hiver/">Autonomie en hiver</a></h4>
</div>
<li class="ipsPagination_next"><a href="https://forums.automobile-propre.com/forums/forum/45-megane-e-tech/?page=2">Suivant</a></li>
</body>
</html>
```

`tests/fixtures/automobile_propre_topic_list_page2_repeat.html` (piège de pagination : mêmes sujets, lien "suivant" toujours présent) :
```html
<!DOCTYPE html>
<html>
<body>
<div class="ipsDataList">
  <h4 data-ips-hook="title"><a href="https://forums.automobile-propre.com/forums/topic/1001-buee-persistante/">Buée persistante sur pare-brise</a></h4>
  <h4 data-ips-hook="title"><a href="https://forums.automobile-propre.com/forums/topic/1002-autonomie-hiver/">Autonomie en hiver</a></h4>
</div>
<li class="ipsPagination_next"><a href="https://forums.automobile-propre.com/forums/forum/45-megane-e-tech/?page=3">Suivant</a></li>
</body>
</html>
```

`tests/fixtures/automobile_propre_topic_page.html` :
```html
<!DOCTYPE html>
<html>
<body>
<article id="elPost_5001">
  <div data-role="commentContent"><p>J'ai toujours de la buée sur le pare-brise même clim à fond.</p></div>
  <a class="ipsUsername" href="https://forums.automobile-propre.com/forums/profile/123-jdupont/">jdupont</a>
  <time datetime="2024-01-15T10:30:00+00:00">15 janvier</time>
</article>
<article id="elComment_5002">
  <div data-role="commentContent"><p>Pareil chez moi, ça sent le moisi en plus.</p></div>
  <a class="ipsUsername" href="https://forums.automobile-propre.com/forums/profile/456-mdurand/">mdurand</a>
  <time datetime="2024-01-16T09:00:00+00:00">16 janvier</time>
</article>
</body>
</html>
```

- [ ] **Step 2: Écrire le test**

```python
import responses

from src.scrapers.automobile_propre import AutomobilePropreScraper
from src.utils.checkpoint import CheckpointStore
from tests.conftest import load_fixture


def make_scraper(tmp_path, target_brands=None):
    checkpoint = CheckpointStore(tmp_path / "progress.json")
    return AutomobilePropreScraper(
        target_brands=target_brands or ["Renault", "Peugeot"],
        checkpoint_store=checkpoint,
        min_delay=0,
        max_delay=0,
    )


@responses.activate
def test_discover_brand_forums_filters_target_brands(tmp_path):
    responses.add(
        responses.GET,
        "https://forums.automobile-propre.com/forums/",
        body=load_fixture("automobile_propre_forums_list.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    brands = scraper.discover_brand_forums()
    assert brands == {
        "Renault": "https://forums.automobile-propre.com/forums/forum/12-renault/",
        "Peugeot": "https://forums.automobile-propre.com/forums/forum/56-peugeot/",
    }


@responses.activate
def test_discover_subforums_returns_model_links(tmp_path):
    responses.add(
        responses.GET,
        "https://forums.automobile-propre.com/forums/forum/12-renault/",
        body=load_fixture("automobile_propre_brand_page.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    subforums = scraper.discover_subforums("https://forums.automobile-propre.com/forums/forum/12-renault/")
    assert subforums == {
        "Megane E-Tech": "https://forums.automobile-propre.com/forums/forum/45-megane-e-tech/",
        "Zoe": "https://forums.automobile-propre.com/forums/forum/46-zoe/",
    }


@responses.activate
def test_iter_topic_urls_stops_when_page_repeats_same_topics(tmp_path):
    subforum_url = "https://forums.automobile-propre.com/forums/forum/45-megane-e-tech/"
    responses.add(
        responses.GET, subforum_url, body=load_fixture("automobile_propre_topic_list_page1.html"), status=200
    )
    responses.add(
        responses.GET,
        f"{subforum_url}?page=2",
        body=load_fixture("automobile_propre_topic_list_page2_repeat.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    urls = list(scraper.iter_topic_urls(subforum_url, scope="automobile_propre:renault:megane-e-tech"))
    assert urls == [
        "https://forums.automobile-propre.com/forums/topic/1001-buee-persistante/",
        "https://forums.automobile-propre.com/forums/topic/1002-autonomie-hiver/",
    ]
    assert len(responses.calls) == 2


@responses.activate
def test_scrape_topic_extracts_posts(tmp_path):
    topic_url = "https://forums.automobile-propre.com/forums/topic/1001-buee-persistante/"
    responses.add(responses.GET, topic_url, body=load_fixture("automobile_propre_topic_page.html"), status=200)
    scraper = make_scraper(tmp_path)
    posts = scraper.scrape_topic(topic_url, brand="Renault", model="Megane E-Tech")
    assert len(posts) == 2
    assert "buée" in posts[0].text
    assert posts[0].author == "jdupont"
    assert posts[0].brand_source == "Renault"
    assert posts[0].model_source == "Megane E-Tech"
```

- [ ] **Step 3: Lancer les tests, vérifier qu'ils échouent**

Run: `pytest tests/test_automobile_propre.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'src.scrapers.automobile_propre'`

- [ ] **Step 4: Implémenter `src/scrapers/automobile_propre.py`**

```python
from datetime import datetime, timezone
from typing import Iterator, Optional

from src.scrapers.base import BaseScraper, HTTPStatusError
from src.storage.models import RawPost, make_post_id
from src.utils.checkpoint import CheckpointStore

FORUMS_URL = "https://forums.automobile-propre.com/forums/"


class AutomobilePropreScraper(BaseScraper):
    def __init__(self, target_brands: list[str], checkpoint_store: CheckpointStore, **kwargs):
        super().__init__("automobile_propre", **kwargs)
        self.target_brands = {b.lower() for b in target_brands}
        self.checkpoint = checkpoint_store

    def discover_brand_forums(self) -> dict[str, str]:
        soup = self.fetch(FORUMS_URL)
        links = soup.select("[data-ips-hook='title'] a")
        return {
            link.get_text(strip=True): link["href"]
            for link in links
            if link.get_text(strip=True).lower() in self.target_brands
        }

    def discover_subforums(self, brand_url: str) -> dict[str, str]:
        soup = self.fetch(brand_url)
        links = soup.select("[data-ips-hook='title'] a")
        return {link.get_text(strip=True): link["href"] for link in links}

    def iter_topic_urls(self, subforum_url: str, scope: str) -> Iterator[str]:
        progress = self.checkpoint.get_progress(scope)
        seen_ids = set(progress["seen_ids"])
        page = progress["last_page"] + 1
        url = subforum_url if page == 1 else f"{subforum_url}?page={page}"

        while True:
            try:
                soup = self.fetch(url)
            except HTTPStatusError:
                break

            links = soup.select("[data-ips-hook='title'] a")
            topic_urls = [link["href"] for link in links if "/topic/" in link["href"]]
            new_urls = [u for u in topic_urls if u not in seen_ids]

            if not new_urls:
                break

            for topic_url in new_urls:
                seen_ids.add(topic_url)
                yield topic_url

            self.checkpoint.update_progress(scope, last_page=page, seen_ids=seen_ids)

            next_link = soup.select_one(
                "li.ipsPagination_next a, [data-role='tablePagination'] a[rel='next']"
            )
            if not next_link:
                break
            page += 1
            url = next_link["href"]

    def scrape_topic(self, topic_url: str, brand: str, model: Optional[str]) -> list[RawPost]:
        soup = self.fetch(topic_url)
        posts = []

        for article in soup.select("article[id^='elPost'], article[id^='elComment']"):
            body = article.select_one("[data-role='commentContent']")
            if not body:
                continue

            author_el = article.select_one(
                ".ipsEntry__username a, .ipsUsername, .ipsComment_author a[href*='/profile/']"
            )
            time_el = article.select_one("time[datetime]")

            text = body.get_text(strip=True)
            author = author_el.get_text(strip=True) if author_el else None
            published_at = time_el["datetime"] if time_el else None
            post_id = make_post_id("automobile_propre", topic_url, author or "", published_at or "")

            posts.append(
                RawPost(
                    id=post_id,
                    source="automobile_propre",
                    brand_source=brand,
                    model_source=model,
                    text=text,
                    author=author,
                    published_at=published_at,
                    url=topic_url,
                    scraped_at=datetime.now(timezone.utc),
                )
            )

        return posts
```

- [ ] **Step 5: Lancer les tests, vérifier qu'ils passent**

Run: `pytest tests/test_automobile_propre.py -v`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add src/scrapers/automobile_propre.py tests/test_automobile_propre.py tests/fixtures/automobile_propre_*.html
git commit -m "Ajoute le scraper automobile-propre.com (découverte dynamique, pagination anti-boucle)"
```

---

### Task 9: Scraper caradisiac.com (avis propriétaires)

**Files:**
- Create: `src/scrapers/caradisiac.py`
- Create fixtures: `tests/fixtures/caradisiac_avis_index.html`, `tests/fixtures/caradisiac_brand_hub_page1.html`, `tests/fixtures/caradisiac_model_hub_page1.html`, `tests/fixtures/caradisiac_model_hub_page2_repeat.html`, `tests/fixtures/caradisiac_avis_page.html`
- Test: `tests/test_caradisiac.py`

**Interfaces:**
- Consumes: `BaseScraper`, `HTTPStatusError` (Task 7) ; `CheckpointStore` (Task 4) ; `RawPost`, `make_post_id` (Task 5) ; `load_fixture` (Task 1)
- Produces: `CaradisiacScraper(target_brands: list[str], checkpoint_store: CheckpointStore, **kwargs)` avec `.discover_brand_hubs() -> dict[str, str]`, `.iter_model_hub_urls(brand_hub_url: str, scope: str) -> Iterator[str]`, `.iter_avis_urls(model_hub_url: str, scope: str) -> Iterator[str]`, `.scrape_avis(avis_url: str) -> RawPost | None`. Utilisé par le CLI (Task 13).

- [ ] **Step 1: Créer les fixtures HTML**

`tests/fixtures/caradisiac_avis_index.html` :
```html
<!DOCTYPE html>
<html>
<body>
<a href="https://www.caradisiac.com/auto--renault/">Renault</a>
<a href="https://www.caradisiac.com/auto--tesla/">Tesla</a>
<a href="https://www.caradisiac.com/auto--peugeot/">Peugeot</a>
</body>
</html>
```

`tests/fixtures/caradisiac_brand_hub_page1.html` :
```html
<!DOCTYPE html>
<html>
<body>
<a href="https://www.caradisiac.com/modele--renault-megane-e-tech/">Megane E-Tech</a>
<a href="https://www.caradisiac.com/modele--renault-zoe/">Zoe</a>
</body>
</html>
```

`tests/fixtures/caradisiac_model_hub_page1.html` :
```html
<!DOCTYPE html>
<html>
<body>
<a href="https://www.caradisiac.com/avis--modele--renault-megane-e-tech/abc123/">Avis 1</a>
<a href="https://www.caradisiac.com/avis--modele--renault-megane-e-tech/def456/">Avis 2</a>
</body>
</html>
```

`tests/fixtures/caradisiac_model_hub_page2_repeat.html` (piège de pagination : mêmes avis répétés) :
```html
<!DOCTYPE html>
<html>
<body>
<a href="https://www.caradisiac.com/avis--modele--renault-megane-e-tech/abc123/">Avis 1</a>
<a href="https://www.caradisiac.com/avis--modele--renault-megane-e-tech/def456/">Avis 2</a>
</body>
</html>
```

`tests/fixtures/caradisiac_avis_page.html` (JSON-LD avec `vehicleModelDate` en int, pour tester le cast défensif) :
```html
<!DOCTYPE html>
<html>
<body>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Review",
  "reviewBody": "Bonne voiture mais la clim peine à désembuer en hiver, ça met un temps fou.",
  "datePublished": "2023-11-02",
  "itemReviewed": {
    "brand": {"name": "Renault"},
    "model": "Megane E-Tech",
    "vehicleModelDate": 2022
  },
  "reviewRating": {"ratingValue": "13", "bestRating": "20"},
  "author": {"name": "Marc D."}
}
</script>
<div class="blocLesPlus">
  <ul><li>Confort de conduite</li><li>Autonomie correcte</li></ul>
</div>
<div class="blocLesMoins">
  <ul><li>Désembuage lent en hiver</li></ul>
</div>
</body>
</html>
```

- [ ] **Step 2: Écrire le test**

```python
import responses

from src.scrapers.caradisiac import CaradisiacScraper
from src.utils.checkpoint import CheckpointStore
from tests.conftest import load_fixture


def make_scraper(tmp_path, target_brands=None):
    checkpoint = CheckpointStore(tmp_path / "progress.json")
    return CaradisiacScraper(
        target_brands=target_brands or ["Renault", "Peugeot"],
        checkpoint_store=checkpoint,
        min_delay=0,
        max_delay=0,
    )


@responses.activate
def test_discover_brand_hubs_filters_target_brands(tmp_path):
    responses.add(
        responses.GET,
        "https://www.caradisiac.com/avis/",
        body=load_fixture("caradisiac_avis_index.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    hubs = scraper.discover_brand_hubs()
    assert hubs == {
        "renault": "https://www.caradisiac.com/auto--renault/",
        "peugeot": "https://www.caradisiac.com/auto--peugeot/",
    }


@responses.activate
def test_iter_model_hub_urls_stops_when_page_repeats(tmp_path):
    brand_hub = "https://www.caradisiac.com/auto--renault/"
    responses.add(
        responses.GET, f"{brand_hub}avis/", body=load_fixture("caradisiac_brand_hub_page1.html"), status=200
    )
    responses.add(
        responses.GET,
        f"{brand_hub}avis_2/",
        body=load_fixture("caradisiac_brand_hub_page1.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    urls = list(scraper.iter_model_hub_urls(brand_hub, scope="caradisiac:renault"))
    assert urls == [
        "https://www.caradisiac.com/modele--renault-megane-e-tech/",
        "https://www.caradisiac.com/modele--renault-zoe/",
    ]
    assert len(responses.calls) == 2


@responses.activate
def test_iter_avis_urls_stops_when_page_repeats(tmp_path):
    model_hub = "https://www.caradisiac.com/modele--renault-megane-e-tech/"
    responses.add(
        responses.GET, f"{model_hub}avis/", body=load_fixture("caradisiac_model_hub_page1.html"), status=200
    )
    responses.add(
        responses.GET,
        f"{model_hub}avis_2/",
        body=load_fixture("caradisiac_model_hub_page2_repeat.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    urls = list(scraper.iter_avis_urls(model_hub, scope="caradisiac:renault:megane-e-tech"))
    assert urls == [
        "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/abc123/",
        "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/def456/",
    ]
    assert len(responses.calls) == 2


@responses.activate
def test_scrape_avis_extracts_from_json_ld_and_casts_year_to_str(tmp_path):
    avis_url = "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/abc123/"
    responses.add(responses.GET, avis_url, body=load_fixture("caradisiac_avis_page.html"), status=200)
    scraper = make_scraper(tmp_path)
    post = scraper.scrape_avis(avis_url)
    assert post is not None
    assert post.brand_source == "Renault"
    assert post.model_source == "Megane E-Tech"
    assert post.year_source == "2022"
    assert "désembuer" in post.text
    assert "Confort de conduite" in post.text
    assert "Désembuage lent" in post.text


@responses.activate
def test_scrape_avis_returns_none_for_non_target_brand(tmp_path):
    avis_url = "https://www.caradisiac.com/avis--modele--tesla-model3/xyz/"
    responses.add(
        responses.GET,
        avis_url,
        body=load_fixture("caradisiac_avis_page.html").replace("Renault", "Tesla"),
        status=200,
    )
    scraper = make_scraper(tmp_path, target_brands=["Renault"])
    post = scraper.scrape_avis(avis_url)
    assert post is None
```

- [ ] **Step 3: Lancer les tests, vérifier qu'ils échouent**

Run: `pytest tests/test_caradisiac.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'src.scrapers.caradisiac'`

- [ ] **Step 4: Implémenter `src/scrapers/caradisiac.py`**

```python
import json
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional

from src.scrapers.base import BaseScraper, HTTPStatusError
from src.storage.models import RawPost, make_post_id
from src.utils.checkpoint import CheckpointStore

AVIS_INDEX_URL = "https://www.caradisiac.com/avis/"


class CaradisiacScraper(BaseScraper):
    def __init__(self, target_brands: list[str], checkpoint_store: CheckpointStore, **kwargs):
        super().__init__("caradisiac", **kwargs)
        self.target_brands = {b.lower() for b in target_brands}
        self.checkpoint = checkpoint_store

    def discover_brand_hubs(self) -> dict[str, str]:
        soup = self.fetch(AVIS_INDEX_URL)
        hubs = {}
        for link in soup.select("a[href*='/auto--']"):
            href = link["href"]
            slug = href.strip("/").split("--")[-1]
            if slug.lower() in self.target_brands:
                hubs[slug.lower()] = href
        return hubs

    def _iter_paginated_links(
        self, page_url_fn: Callable[[int], str], link_selector: str, scope: str
    ) -> Iterator[str]:
        progress = self.checkpoint.get_progress(scope)
        seen = set(progress["seen_ids"])
        page = progress["last_page"] + 1

        while True:
            url = page_url_fn(page)
            try:
                soup = self.fetch(url)
            except HTTPStatusError:
                break

            links = [a["href"] for a in soup.select(link_selector)]
            new_links = [u for u in links if u not in seen]

            if not new_links:
                break

            for link in new_links:
                seen.add(link)
                yield link

            self.checkpoint.update_progress(scope, last_page=page, seen_ids=seen)
            page += 1

    def iter_model_hub_urls(self, brand_hub_url: str, scope: str) -> Iterator[str]:
        def page_url(page: int) -> str:
            return f"{brand_hub_url}avis/" if page == 1 else f"{brand_hub_url}avis_{page}/"

        yield from self._iter_paginated_links(page_url, "a[href*='/modele--']", scope)

    def iter_avis_urls(self, model_hub_url: str, scope: str) -> Iterator[str]:
        def page_url(page: int) -> str:
            return f"{model_hub_url}avis/" if page == 1 else f"{model_hub_url}avis_{page}/"

        yield from self._iter_paginated_links(page_url, "a[href*='/avis--modele--']", scope)

    def scrape_avis(self, avis_url: str) -> Optional[RawPost]:
        soup = self.fetch(avis_url)
        script = soup.select_one("script[type='application/ld+json']")
        if not script or not script.string:
            return None

        data = json.loads(script.string)
        if data.get("@type") != "Review":
            return None

        item = data.get("itemReviewed", {})
        brand = item.get("brand", {}).get("name")
        if not brand or brand.lower() not in self.target_brands:
            return None

        model = item.get("model")
        year_raw = item.get("vehicleModelDate")
        year_source = str(year_raw) if year_raw is not None else None

        text = data.get("reviewBody", "")
        plus_items = [li.get_text(strip=True) for li in soup.select("div.blocLesPlus li")]
        minus_items = [li.get_text(strip=True) for li in soup.select("div.blocLesMoins li")]
        if plus_items:
            text += "\nPoints positifs: " + "; ".join(plus_items)
        if minus_items:
            text += "\nPoints négatifs: " + "; ".join(minus_items)

        author = data.get("author", {}).get("name")
        published_at = data.get("datePublished")
        post_id = make_post_id("caradisiac", avis_url, author or "", published_at or "")

        return RawPost(
            id=post_id,
            source="caradisiac",
            brand_source=brand,
            model_source=model,
            year_source=year_source,
            text=text,
            author=author,
            published_at=published_at,
            url=avis_url,
            scraped_at=datetime.now(timezone.utc),
        )
```

- [ ] **Step 5: Lancer les tests, vérifier qu'ils passent**

Run: `pytest tests/test_caradisiac.py -v`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add src/scrapers/caradisiac.py tests/test_caradisiac.py tests/fixtures/caradisiac_*.html
git commit -m "Ajoute le scraper caradisiac.com (avis propriétaires, extraction JSON-LD)"
```

---

### Task 10: Pré-filtre par mots-clés

**Files:**
- Create: `src/pipeline/prefilter.py`
- Test: `tests/test_prefilter.py`

**Interfaces:**
- Consumes: `load_yaml` (Task 1)
- Produces: `KeywordPrefilter(keywords_path: str | Path)` avec `.is_relevant(text: str) -> bool`. Utilisé par le CLI (Task 13) pour décider si un post part vers le LLM.

- [ ] **Step 1: Écrire le test**

```python
from src.pipeline.prefilter import KeywordPrefilter

KEYWORDS_YAML = """
categories:
  buee_desembuage:
    - buée
    - désembuage
  chauffage:
    - chauffage
    - "ne chauffe pas"
"""


def make_prefilter(tmp_path):
    path = tmp_path / "keywords.yaml"
    path.write_text(KEYWORDS_YAML, encoding="utf-8")
    return KeywordPrefilter(path)


def test_is_relevant_true_when_keyword_present(tmp_path):
    prefilter = make_prefilter(tmp_path)
    assert prefilter.is_relevant("J'ai de la buée sur le pare-brise") is True


def test_is_relevant_false_when_no_keyword(tmp_path):
    prefilter = make_prefilter(tmp_path)
    assert prefilter.is_relevant("La voiture consomme peu sur autoroute") is False


def test_is_relevant_is_case_insensitive(tmp_path):
    prefilter = make_prefilter(tmp_path)
    assert prefilter.is_relevant("BUÉE partout ce matin") is True


def test_is_relevant_true_on_real_project_keywords():
    prefilter = KeywordPrefilter("config/keywords.yaml")
    assert prefilter.is_relevant("le chauffage ne fonctionne plus depuis hier") is True
    assert prefilter.is_relevant("la peinture s'écaille sur le capot") is False
```

- [ ] **Step 2: Lancer le test, vérifier qu'il échoue**

Run: `pytest tests/test_prefilter.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'src.pipeline.prefilter'`

- [ ] **Step 3: Implémenter `src/pipeline/prefilter.py`**

```python
from pathlib import Path

from src.utils.config import load_yaml


class KeywordPrefilter:
    def __init__(self, keywords_path: str | Path):
        data = load_yaml(keywords_path)
        self.categories: dict[str, list[str]] = data["categories"]

    def is_relevant(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            keyword.lower() in lowered
            for keywords in self.categories.values()
            for keyword in keywords
        )
```

- [ ] **Step 4: Lancer le test, vérifier qu'il passe**

Run: `pytest tests/test_prefilter.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/prefilter.py tests/test_prefilter.py
git commit -m "Ajoute le pré-filtre mots-clés confort thermique"
```

---

### Task 11: Installation d'Ollama + classifieur LLM

**Files:**
- Create: `src/pipeline/llm_classifier.py`
- Test: `tests/test_llm_classifier.py`
- Test: `tests/test_llm_classifier_integration.py`

**Interfaces:**
- Consumes: `RawPost`, `LLMClassificationResult` (Task 5)
- Produces: `LLMClassifier(base_url="http://localhost:11434", model="qwen2.5:7b-instruct", max_retries=2, timeout=60, session=None)` avec `.classify(raw: RawPost) -> LLMClassificationResult`. Produces aussi `LLMClassificationError` (base), `OllamaUnavailableError` (Ollama injoignable, levée immédiatement sans retry), `LLMResponseInvalidError` (JSON invalide après épuisement des retries). Utilisé par le CLI (Task 13).

- [ ] **Step 1: Installer Ollama et tirer le modèle**

Run:
```bash
brew install ollama
ollama serve &
ollama pull qwen2.5:7b-instruct
```
Expected: le pull se termine sans erreur (peut prendre plusieurs minutes selon la connexion).

- [ ] **Step 2: Vérifier que le serveur répond**

Run: `curl -s http://localhost:11434/api/tags`
Expected: une réponse JSON listant `qwen2.5:7b-instruct` dans `models`.

- [ ] **Step 3: Écrire les tests unitaires (mockés, aucun appel réseau réel)**

```python
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
def test_classify_raises_ollama_unavailable_immediately_without_retry():
    classifier = LLMClassifier(max_retries=2)
    with pytest.raises(OllamaUnavailableError):
        classifier.classify(make_raw_post())
    assert len(responses.calls) == 1
```

- [ ] **Step 4: Lancer les tests, vérifier qu'ils échouent**

Run: `pytest tests/test_llm_classifier.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'src.pipeline.llm_classifier'`

- [ ] **Step 5: Implémenter `src/pipeline/llm_classifier.py`**

```python
import json
from typing import Optional

import requests

from src.storage.models import LLMClassificationResult, RawPost

SYSTEM_PROMPT = (
    "Tu es un assistant qui analyse des verbatims d'automobilistes français à propos du confort "
    "thermique de leur véhicule (chauffage, climatisation, ventilation, buée/désembuage, isolation "
    "thermique, odeur des systèmes de ventilation/climatisation, pompe à chaleur, préchauffage "
    "habitacle/batterie). Réponds STRICTEMENT au format JSON demandé, sans texte autour."
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

            content = response.json().get("message", {}).get("content", "")
            try:
                data = json.loads(content)
                return LLMClassificationResult(**data)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc

        raise LLMResponseInvalidError(
            f"Réponse LLM invalide après {self.max_retries} tentatives pour le post {raw.id}"
        ) from last_error
```

- [ ] **Step 6: Lancer les tests unitaires, vérifier qu'ils passent**

Run: `pytest tests/test_llm_classifier.py -v`
Expected: `4 passed`

- [ ] **Step 7: Écrire le test d'intégration (contre le vrai Ollama installé)**

```python
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
```

- [ ] **Step 8: Lancer le test d'intégration contre le vrai Ollama**

Run: `pytest -m integration tests/test_llm_classifier_integration.py -v`
Expected: `1 passed` (nécessite `ollama serve` actif et le modèle `qwen2.5:7b-instruct` tiré, étapes 1-2 ci-dessus)

- [ ] **Step 9: Commit**

```bash
git add src/pipeline/llm_classifier.py tests/test_llm_classifier.py tests/test_llm_classifier_integration.py
git commit -m "Ajoute le classifieur LLM (Ollama/Qwen2.5, structured output, retry, distinction erreurs)"
```

---

### Task 12: Export et statistiques

**Files:**
- Create: `src/pipeline/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `ClassifiedPost` (Task 5)
- Produces: `to_dataframe(posts: list[ClassifiedPost]) -> pd.DataFrame`, `stats_by_brand_model_sentiment(df) -> pd.DataFrame`, `stats_by_source(df) -> pd.DataFrame`, `stats_by_month(df) -> pd.DataFrame`, `export_all(classified_path, output_dir) -> dict[str, Path]`. Utilisé par le CLI (Task 13).

- [ ] **Step 1: Écrire les tests**

```python
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
```

- [ ] **Step 2: Lancer les tests, vérifier qu'ils échouent**

Run: `pytest tests/test_export.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'src.pipeline.export'`

- [ ] **Step 3: Implémenter `src/pipeline/export.py`**

```python
from pathlib import Path

import pandas as pd

from src.storage.models import ClassifiedPost


def to_dataframe(posts: list[ClassifiedPost]) -> pd.DataFrame:
    return pd.DataFrame([p.model_dump(mode="json") for p in posts])


def relevant_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["pertinent_confort_thermique"] & ~df["classification_echec"]]


def stats_by_brand_model_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    relevant = relevant_only(df)
    return (
        relevant.groupby(["vehicule_marque", "vehicule_modele", "sentiment"])
        .size()
        .reset_index(name="count")
    )


def stats_by_source(df: pd.DataFrame) -> pd.DataFrame:
    relevant = relevant_only(df)
    return relevant.groupby("source").size().reset_index(name="count")


def stats_by_month(df: pd.DataFrame) -> pd.DataFrame:
    relevant = relevant_only(df).copy()
    relevant["published_at"] = pd.to_datetime(relevant["published_at"])
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
```

- [ ] **Step 4: Lancer les tests, vérifier qu'ils passent**

Run: `pytest tests/test_export.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/export.py tests/test_export.py
git commit -m "Ajoute l'export parquet + stats (marque/modèle/sentiment, source, mois)"
```

---

### Task 13: CLI (main.py)

**Files:**
- Create: `main.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: tout ce qui précède (`AutomobilePropreScraper`, `CaradisiacScraper`, `CheckpointStore`, `KeywordPrefilter`, `LLMClassifier`, `OllamaUnavailableError`, `LLMResponseInvalidError`, `append_posts`, `read_existing_ids`, `read_posts`, `export_all`, `get_logger`, `load_yaml`)
- Produces: `cli` (groupe click) avec les sous-commandes `scrape`, `classify`, `export` ; `SETTINGS`, `VEHICLE_LIST` (dicts de config chargés au niveau module).

- [ ] **Step 1: Écrire les tests**

```python
from datetime import datetime, timezone

from click.testing import CliRunner

import main
from src.storage.models import ClassifiedPost
from src.storage.writer import append_posts


def make_classified_post(post_id: str) -> ClassifiedPost:
    return ClassifiedPost(
        id=post_id,
        source="automobile_propre",
        text="texte",
        url=f"https://x/{post_id}",
        scraped_at=datetime.now(timezone.utc),
        published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        pertinent_confort_thermique=True,
        pertinence_confidence="haute",
        vehicule_marque="Renault",
        vehicule_modele="Zoe",
        sentiment=3,
        sentiment_confidence="haute",
        classification_echec=False,
    )


def test_cli_registers_all_three_commands():
    assert set(main.cli.commands.keys()) == {"scrape", "classify", "export"}


def test_export_command_writes_output_files(tmp_path, monkeypatch):
    classified_path = tmp_path / "posts_classified.parquet"
    output_dir = tmp_path / "output"
    append_posts(classified_path, [make_classified_post("1"), make_classified_post("2")])

    monkeypatch.setitem(main.SETTINGS["paths"], "classified_posts", str(classified_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "output_dir", str(output_dir))

    runner = CliRunner()
    result = runner.invoke(main.cli, ["export"])

    assert result.exit_code == 0
    assert (output_dir / "verbatims_classifies.parquet").exists()
    assert (output_dir / "stats_marque_modele_sentiment.parquet").exists()
```

- [ ] **Step 2: Lancer les tests, vérifier qu'ils échouent**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Implémenter `main.py`**

```python
import sys

import click
from rich.console import Console
from rich.table import Table

from src.pipeline.export import export_all
from src.pipeline.llm_classifier import (
    LLMClassifier,
    LLMResponseInvalidError,
    OllamaUnavailableError,
)
from src.pipeline.prefilter import KeywordPrefilter
from src.scrapers.automobile_propre import AutomobilePropreScraper
from src.scrapers.caradisiac import CaradisiacScraper
from src.storage.models import ClassifiedPost, RawPost
from src.storage.writer import append_posts, read_existing_ids, read_posts
from src.utils.checkpoint import CheckpointStore
from src.utils.config import load_yaml
from src.utils.logger import get_logger

console = Console()
SETTINGS = load_yaml("config/settings.yaml")
VEHICLE_LIST = load_yaml("config/vehicle_list.yaml")["brands"]


@click.group()
def cli():
    pass


def _scrape_automobile_propre(scraper: AutomobilePropreScraper) -> list[RawPost]:
    posts: list[RawPost] = []
    for brand, brand_url in scraper.discover_brand_forums().items():
        for model, model_url in scraper.discover_subforums(brand_url).items():
            scope = f"automobile_propre:{brand}:{model}"
            for topic_url in scraper.iter_topic_urls(model_url, scope):
                posts.extend(scraper.scrape_topic(topic_url, brand=brand, model=model))
    return posts


def _scrape_caradisiac(scraper: CaradisiacScraper) -> list[RawPost]:
    posts: list[RawPost] = []
    for brand, brand_hub_url in scraper.discover_brand_hubs().items():
        model_scope = f"caradisiac:{brand}"
        for model_hub_url in scraper.iter_model_hub_urls(brand_hub_url, scope=model_scope):
            avis_scope = f"caradisiac:{brand}:{model_hub_url}"
            for avis_url in scraper.iter_avis_urls(model_hub_url, scope=avis_scope):
                post = scraper.scrape_avis(avis_url)
                if post:
                    posts.append(post)
    return posts


@cli.command()
@click.option(
    "--source", type=click.Choice(["automobile_propre", "caradisiac", "all"]), default="all"
)
def scrape(source):
    logger = get_logger("scraper", log_file="data/scraper.log")
    checkpoint = CheckpointStore(SETTINGS["paths"]["checkpoint"])
    sources = ["automobile_propre", "caradisiac"] if source == "all" else [source]

    common_kwargs = dict(
        checkpoint_store=checkpoint,
        min_delay=SETTINGS["rate_limit"]["min_seconds"],
        max_delay=SETTINGS["rate_limit"]["max_seconds"],
        timeout=SETTINGS["http"]["timeout_seconds"],
        max_retries=SETTINGS["http"]["max_retries"],
        user_agent=SETTINGS["http"]["user_agent"],
    )

    for src_name in sources:
        if src_name == "automobile_propre":
            scraper = AutomobilePropreScraper(target_brands=VEHICLE_LIST, **common_kwargs)
            posts = _scrape_automobile_propre(scraper)
        else:
            scraper = CaradisiacScraper(target_brands=VEHICLE_LIST, **common_kwargs)
            posts = _scrape_caradisiac(scraper)

        written = append_posts(SETTINGS["paths"]["raw_posts"], posts)
        logger.info("%s: %d nouveaux posts écrits", src_name, written)
        console.print(f"[green]{src_name}[/green] : {written} nouveaux posts")


@cli.command()
def classify():
    logger = get_logger("scraper", log_file="data/scraper.log")
    prefilter = KeywordPrefilter("config/keywords.yaml")
    llm = LLMClassifier(
        base_url=SETTINGS["ollama"]["base_url"],
        model=SETTINGS["ollama"]["model"],
        max_retries=SETTINGS["ollama"]["max_retries"],
    )

    raw_posts = read_posts(SETTINGS["paths"]["raw_posts"], RawPost)
    already_classified = read_existing_ids(SETTINGS["paths"]["classified_posts"])
    to_classify = [p for p in raw_posts if p.id not in already_classified]

    console.print(f"{len(to_classify)} posts à classifier ({len(raw_posts) - len(to_classify)} déjà faits)")

    results: list[ClassifiedPost] = []
    for post in to_classify:
        if not prefilter.is_relevant(post.text):
            continue
        try:
            llm_result = llm.classify(post)
            results.append(ClassifiedPost.from_raw_and_llm(post, llm_result))
        except OllamaUnavailableError:
            written = append_posts(SETTINGS["paths"]["classified_posts"], results)
            console.print(
                f"[red]Ollama injoignable — lance `ollama serve`. "
                f"{written} posts classifiés avant l'arrêt.[/red]"
            )
            sys.exit(1)
        except LLMResponseInvalidError as exc:
            logger.warning("Classification échouée pour %s: %s", post.id, exc)
            results.append(ClassifiedPost.failed(post))

    written = append_posts(SETTINGS["paths"]["classified_posts"], results)
    console.print(f"[green]{written}[/green] posts classifiés écrits")


@cli.command()
def export():
    outputs = export_all(SETTINGS["paths"]["classified_posts"], SETTINGS["paths"]["output_dir"])
    table = Table(title="Export terminé")
    table.add_column("Fichier")
    table.add_column("Chemin")
    for name, path in outputs.items():
        table.add_row(name, str(path))
    console.print(table)


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Lancer les tests, vérifier qu'ils passent**

Run: `pytest tests/test_cli.py -v`
Expected: `2 passed`

- [ ] **Step 5: Lancer toute la suite de tests unitaires**

Run: `pytest -v`
Expected: tous les tests passent (les tests marqués `integration` sont skippés par défaut)

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_cli.py
git commit -m "Ajoute le CLI (scrape/classify/export) qui assemble tout le pipeline"
```

---

### Task 14: Validation dry-run bout-en-bout

**Files:** aucun fichier créé — validation manuelle du pipeline complet contre les vrais sites.

**Interfaces:** aucune (tâche de vérification finale, pas de nouveau code).

- [ ] **Step 1: Lancer un scrape limité pour valider la découverte dynamique en conditions réelles**

Run: `python main.py scrape --source automobile_propre`
Expected: le run démarre, log des marques/sous-forums découverts en console, écrit des posts dans `data/processed/posts_raw.parquet`. Interrompre après quelques minutes (Ctrl+C) une fois que plusieurs marques ont été traitées — le checkpoint permet de reprendre plus tard sans perte.

- [ ] **Step 2: Vérifier le contenu collecté**

Run:
```bash
python -c "import pandas as pd; df = pd.read_parquet('data/processed/posts_raw.parquet'); print(len(df)); print(df['brand_source'].value_counts())"
```
Expected: un nombre de posts > 0, réparti sur au moins 2 marques ciblées.

- [ ] **Step 3: Lancer le même test sur caradisiac**

Run: `python main.py scrape --source caradisiac`
Expected: mêmes vérifications que Step 1-2, posts caradisiac ajoutés au même fichier `posts_raw.parquet` (pas de doublons avec ceux d'automobile_propre grâce à l'id stable incluant la source).

- [ ] **Step 4: Lancer la classification sur l'échantillon collecté**

Run: `python main.py classify`
Expected: le pré-filtre réduit le volume envoyé au LLM, `data/processed/posts_classified.parquet` est créé, aucune erreur `OllamaUnavailableError` (Ollama doit tourner depuis la Task 11).

- [ ] **Step 5: Lancer l'export et inspecter les stats**

Run: `python main.py export`
Expected: table `rich` affichant les 4 fichiers générés dans `data/output/` ; ouvrir `stats_marque_modele_sentiment.parquet` pour confirmer que les répartitions sont cohérentes avec l'échantillon collecté.

- [ ] **Step 6: Relancer `scrape` une seconde fois pour valider la reprise**

Run: `python main.py scrape --source automobile_propre`
Expected: le nombre de "nouveaux posts" est inférieur ou égal au premier run (dédup + checkpoint fonctionnent, pas de doublons ni de re-parcours complet depuis zéro).

Pas de commit pour cette tâche (aucun fichier de code modifié) — c'est la validation finale que le pipeline fonctionne de bout en bout sur les vrais sites, condition posée par l'utilisateur avant de lancer le run complet à grande échelle sur les 15 marques.
