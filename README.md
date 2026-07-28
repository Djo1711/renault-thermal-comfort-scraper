# Pipeline confort thermique — scraping + classification LLM

Ce projet scrape des verbatims d'automobilistes français sur deux forums/sites
auto (forums.automobile-propre.com et caradisiac.com), pour Renault et 14
marques concurrentes, puis les classifie via un LLM local (Ollama) pour
détecter et qualifier les problèmes de confort thermique (chauffage,
climatisation, ventilation, buée/désembuage, isolation thermique, odeur HVAC,
pompe à chaleur, préchauffage habitacle/batterie). Les résultats sont ensuite
exportés en fichiers parquet (verbatims classifiés + statistiques agrégées).

## Prérequis

- Python 3.11+
- [Ollama](https://ollama.com) installé et lancé (`ollama serve`), avec le
  modèle `qwen2.5:7b-instruct` déjà téléchargé :

  ```bash
  ollama pull qwen2.5:7b-instruct
  ```

## Installation

```bash
pip install -r requirements.txt
```

(Pour lancer les tests : `pip install -r requirements-dev.txt`.)

## Commandes CLI

Les 3 commandes doivent être lancées **depuis la racine du dépôt** (les
chemins de `config/` et `data/` sont relatifs).

```bash
python main.py scrape --source {automobile_propre,caradisiac,all}   # scrape les posts bruts vers data/processed/posts_raw.parquet
python main.py classify                                              # classifie les posts non encore traités via le LLM Ollama
python main.py export                                                # exporte les verbatims classifiés + stats vers data/output/
```

`scrape` et `classify` reprennent automatiquement où ils se sont arrêtés
(checkpoint / dédoublonnage par id) — ils peuvent être relancés sans risque
après une interruption.
