# Design — Pipeline scraping + classification LLM confort thermique

**Date :** 2026-07-23
**Statut :** Approuvé

## Contexte et objectif

Reconstruction from scratch d'un pipeline de scraping et d'analyse de verbatims
auto français, focalisé sur le confort thermique (chauffage, climatisation,
ventilation, buée/désembuage, isolation thermique, odeur HVAC, pompe à
chaleur, préchauffage habitacle/batterie) pour Renault et 14 marques
concurrentes ciblées.

Un premier prototype (v1) avait déjà été construit et partiellement validé en
conditions réelles (scraping fonctionnel sur les deux sources, classification
par regex/distilcamembert). Cette v1 a été volontairement remise à zéro
(dossier local vidé, historique git et repo GitHub recréés) pour repartir sur
une base propre, en remplaçant le classifieur regex par un agent LLM open
source. Les connaissances techniques vérifiées sur les vrais sites lors de la
v1 (sélecteurs CSS, pièges de pagination, structure JSON-LD) restent valables
et sont réutilisées telles quelles ci-dessous — ce sont des faits observés en
direct sur les sites, pas des hypothèses.

## Sources de données

### Source 1 : forums.automobile-propre.com (forum Invision, orienté VE/hybride)

- Liste des marques : `GET /forums/`, liens de sous-forum via le sélecteur
  CSS `[data-ips-hook='title'] a` (les anciennes classes
  `ipsDataItem_title`/`cForumTitle` ne matchent plus rien). Filtrer les liens
  obtenus sur les 15 marques ciblées (`config/vehicle_list.yaml`),
  comparaison case-insensitive sur le nom exact.
- Chaque marque a souvent des sous-forums par modèle un niveau plus bas
  (même sélecteur) — à crawler aussi, pas juste le niveau marque.
- Les IDs numériques dans les URLs de sous-forums dérivent dans le temps
  (renumérotation par le site) — ne jamais hardcoder une liste d'URLs,
  toujours découvrir dynamiquement à chaque run.
- Liens de sujets dans une page de listing : même sélecteur, filtré sur
  `/topic/` dans le href.
- Page de sujet : posts dans `article[id^='elPost'], article[id^='elComment']`,
  corps du message dans `[data-role='commentContent']`, auteur dans
  `.ipsEntry__username a, .ipsUsername, .ipsComment_author a[href*='/profile/']`,
  date dans `time[datetime]`.
- Pagination : lien suivant via
  `li.ipsPagination_next a, [data-role='tablePagination'] a[rel='next']`.
  Piège confirmé : au-delà de la dernière page réelle, le site peut renvoyer
  une page qui répète le contenu de la dernière page (ou une page vide) tout
  en exposant encore un lien "suivant" → boucle infinie si non gérée. On
  garde un set des sujets déjà vus par sous-forum et on arrête la pagination
  dès qu'une page ne ramène aucun sujet nouveau.
- Les compteurs `reply_count`/`view_count` (`[data-stat-key=...]`) ne sont
  pas toujours présents en vue anonyme — traités comme best-effort.

### Source 2 : caradisiac.com — section "avis propriétaires"

- L'endpoint `/recherche/?q=` est mort (HTTP 410) — non utilisé.
- Point d'entrée : `https://www.caradisiac.com/avis/`, liens marque
  `/auto--{marque}/`, filtrés sur les 15 marques ciblées (Cupra n'a pas de
  hub sur ce site, c'est normal).
- Flux paginé par marque : `/auto--{marque}/avis/`, `/avis_2/`, `/avis_3/`...
  liste des hubs modèle : `/modele--{marque}-{modele}/avis/`.
- Chaque hub modèle paginé de la même façon, liste des avis individuels :
  `/avis--modele--{marque}-{modele}/{hash}/`.
- Piège identique à la source 1 : arrêt de pagination dès qu'une page ne
  ramène aucun lien nouveau (set cumulatif). Une page hors limites peut
  aussi renvoyer HTTP 400 — signal d'arrêt valide, pas une erreur.
- Chaque page d'avis contient un bloc `<script type="application/ld+json">`
  avec `"@type": "Review"` — **source de données principale**, préférée au
  DOM visible :
  - `reviewBody`, `datePublished`
  - `itemReviewed.brand.name`, `itemReviewed.model`,
    `itemReviewed.vehicleModelDate` (caster défensivement en `str`, peut
    être int ou string selon les pages)
  - `reviewRating.ratingValue` / `reviewRating.bestRating` (note /20)
  - `author.name`
- `div.blocLesPlus li` ("A aimé") / `div.blocLesMoins li` ("N'a pas aimé") :
  à concaténer au texte du verbatim.
- Piège : la pagination d'un hub modèle peut afficher des avis d'autres
  modèles/marques (widgets de recommandation) — toujours se fier à la
  marque/modèle du JSON-LD de l'avis lui-même, jamais à l'URL qui y a mené ;
  écarter tout ce qui sort des 15 marques ciblées.

### Source explicitement écartée

`forum-auto.caradisiac.com` : protégé par un challenge Cloudflare moderne
(403 confirmé en direct, curl brut et cloudscraper échouent tous les deux).
Contourner nécessiterait un navigateur headless, sans garantie de succès.
Décision déjà prise : abandonné.

## Architecture générale

Trois commandes CLI indépendantes (`click`), chacune lit/écrit des fichiers
parquet sur disque, chacune reprenable seule :

```
scrape                classify                  export
──────                ────────                  ──────
2 scrapers      →   pré-filtre keywords    →   parquet verbatims
(auto_propre,        (bon marché)               classifiés
 caradisiac)              ↓
     ↓                agent LLM (Ollama         stats (répartition
checkpoint            Qwen2.5 7B) → JSON         marque/modèle/
+ dédup                   ↓                      sentiment/source/
     ↓                validation pydantic        temporel)
posts_raw.parquet    posts_classified.parquet
```

- `python main.py scrape --source {automobile_propre,caradisiac,all}` →
  `data/processed/posts_raw.parquet` + `data/checkpoints/progress.json`
- `python main.py classify` → `data/processed/posts_classified.parquet`
- `python main.py export` → `data/output/*.parquet` + résumé stats (table
  `rich` en console + fichier)

Changement par rapport à la v1 : parquet dès l'étape brute (au lieu de CSV) —
plus robuste sur les types (évite le problème `vehicleModelDate` int/string
déjà rencontré) et plus rapide à relire à chaque reprise.

Objectif de volume : run complet sur les 15 marques dès que le pipeline est
validé en dry-run — le checkpoint/reprise est donc critique dès le départ,
pas une nice-to-have.

## Composants

**`src/scrapers/`** (repris de la v1, logique déjà validée en conditions
réelles — voir section Sources ci-dessus)
- `base.py` : classe abstraite commune (rate-limiting aléatoire 3-8s entre
  requêtes, session HTTP, retry sur erreurs réseau transitoires)
- `automobile_propre.py` : découverte dynamique marque → sous-forums modèle →
  sujets → posts
- `caradisiac.py` : navigation `/avis/` → hub marque → hub modèle → avis
  individuel, extraction JSON-LD

**`src/storage/models.py`** (pydantic)
- `RawPost` : id stable (hash source+url+auteur+date), source, marque/modèle
  si connus de la page (JSON-LD caradisiac, ou nom du sous-forum
  automobile_propre), texte brut, date, url, auteur
- `ClassifiedPost` : `RawPost` + sortie LLM (voir Schéma de classification)

**`src/pipeline/prefilter.py`** — filtrage bon marché par mots-clés
(`config/keywords.yaml`, repris de la v1) décidant si un post part vers le
LLM. Remplace l'ancien `classifier.py` regex qui faisait la classification
complète en v1.

**`src/pipeline/llm_classifier.py`** — construit le prompt (texte + contexte
marque/modèle connu de la source), appelle Ollama via son API HTTP locale
(`/api/chat`, structured output JSON schema de Qwen2.5), parse/valide la
réponse en pydantic, gère le retry si le JSON est invalide.

**`src/storage/writer.py`** — écriture/lecture parquet, dédup par id stable,
reprise (skip des ids déjà présents).

**`src/pipeline/export.py`** — agrégations pandas (marque/modèle/sentiment/
source/mois) → parquet + table récapitulative.

**`main.py`** — CLI click avec les 3 sous-commandes ci-dessus.

## Modèle LLM

Qwen2.5 **7B** Instruct via Ollama : bon compromis français + support natif
du structured output JSON dans Ollama. Le 14B est écarté pour cette machine
(16 Go de RAM unifiée, Apple M5) — trop juste à côté de l'OS et des autres
applications. Installation guidée dans le plan d'implémentation
(`brew install ollama`, `ollama pull qwen2.5:7b-instruct`), en supposant un
serveur Ollama tournant sur `localhost:11434`.

## Schéma de classification LLM

**Entrée** (prompt construit par `llm_classifier.py`) :
```
Texte du verbatim : "{texte brut du post}"
Contexte source : marque={marque_source|inconnu}, modèle={modele_source|inconnu}, site={automobile_propre|caradisiac}
```
Pour caradisiac, marque/modèle/année viennent du JSON-LD fiable → contexte
fort. Pour automobile_propre, seule la marque (parfois le modèle) vient du
nom du sous-forum → contexte plus faible, à charge du LLM de croiser avec le
texte.

**Sortie** (JSON schema imposé via le structured output d'Ollama, validé
ensuite par pydantic) :

```json
{
  "pertinent_confort_thermique": true,
  "pertinence_confidence": "haute",
  "vehicule": {
    "marque": "Renault",
    "marque_confidence": "haute",
    "modele": "Megane E-Tech",
    "modele_confidence": "moyenne",
    "annee": 2023,
    "annee_confidence": "basse",
    "motorisation": "électrique",
    "motorisation_confidence": "haute"
  },
  "type_probleme": ["buee_desembuage"],
  "type_probleme_confidence": "haute",
  "zone_vehicule": "pare-brise avant",
  "sentiment": 2,
  "sentiment_confidence": "haute",
  "resume_probleme": "Buée persistante sur le pare-brise en hiver malgré la clim",
  "extrait_source": "j'ai toujours de la buée sur le pare-brise même clim à fond"
}
```

- `*_confidence` : énumération `haute|moyenne|basse` (pas de score continu —
  plus simple à produire de façon fiable par un modèle 7B, suffisant pour
  filtrer les cas limites a posteriori).
- `sentiment` : entier 1-5 (1 = très négatif, 5 = très positif), toujours
  produit même si `pertinent_confort_thermique=false` (coût marginal nul).
- `type_probleme` : **liste** de valeurs parmi `chauffage, climatisation,
  ventilation, buee_desembuage, isolation_thermique, odeur_hvac,
  pompe_a_chaleur, prechauffage, autre` (repris de `config/keywords.yaml`) —
  un même verbatim peut mentionner plusieurs problèmes.
- `zone_vehicule` : texte libre court (nullable) — la pièce/zone en cause
  (ex. "pare-brise avant", "habitacle", "batterie").
- `extrait_source` : citation courte du texte original (nullable si aucun
  passage précis n'isole le problème) — traçabilité/audit de la
  classification.
- Champs `vehicule.*` nullables si le LLM ne peut vraiment rien déterminer
  (aucune source, aucun indice texte).

`ClassifiedPost` (pydantic, écrit en parquet) = tous les champs de `RawPost`
+ tous les champs ci-dessus, à plat (colonnes préfixées
`vehicule_marque`, `vehicule_marque_confidence`, etc.). `type_probleme`
stocké comme liste de strings (colonne native parquet).

## Export et statistiques

- Parquet des verbatims classifiés (tous les champs `ClassifiedPost`).
- Stats agrégées :
  - répartition par marque/modèle/sentiment
  - répartition par source (automobile_propre vs caradisiac)
  - évolution temporelle (par mois/année de `datePublished`/date du post)
- `resume_probleme` et `extrait_source` servent d'extraits représentatifs
  lisibles directement depuis le parquet classifié (pas de table séparée
  nécessaire).

## Gestion des erreurs

**Scraping** (repris de la v1, déjà validé) :
- Rate-limiting : délai aléatoire 3-8s entre chaque requête HTTP, par
  scraper.
- Retry réseau : 3 tentatives avec backoff exponentiel sur timeout/erreur
  5xx ; abandon du sujet/page en cours (log warning) après épuisement, sans
  arrêter tout le run.
- Pagination : arrêt dès qu'une page ne ramène aucun id nouveau (set
  cumulatif par sous-forum/hub). HTTP 400 sur caradisiac traité comme signal
  d'arrêt valide.
- Checkpoint : après chaque sujet/page traité, persistance (marque,
  sous-forum, dernière page atteinte, ids déjà vus) dans
  `data/checkpoints/progress.json`. Un crash à mi-run reprend à la
  marque/page suivante.
- Dédup : id stable (hash source+url+auteur+date) vérifié contre le parquet
  existant avant ajout — un post déjà vu est ignoré silencieusement (log
  debug).

**Classification LLM** :
- Ollama injoignable (serveur non lancé, timeout) : le run `classify`
  s'arrête immédiatement avec un message clair ("lance `ollama serve`") —
  erreur bloquante, pas de retry en boucle.
- JSON invalide/ne matchant pas le schéma pydantic : 1 retry avec un prompt
  renforcé ("répond STRICTEMENT au format JSON demandé"). Si le 2e essai
  échoue, le post est marqué `classification_echec=true` dans le parquet
  (pas de blocage du run) et loggé en warning avec l'id du post.
- Reprise : `classify` saute les posts déjà présents dans
  `posts_classified.parquet` (par id) — un crash à mi-run se relance en
  ré-exécutant simplement la commande.

**Export** : purement déterministe — échec explicite si
`posts_classified.parquet` est absent/vide, pas de gestion d'erreur
particulière au-delà.

## Stratégie de tests

- **Scrapers** : fixtures HTML réelles sauvegardées (`tests/fixtures/`,
  capturées depuis les vrais sites) pour tester sélecteurs CSS, découverte
  dynamique, et logique d'arrêt de pagination (page vide / ids déjà vus /
  HTTP 400). Aucun appel réseau réel dans les tests.
- **`storage/writer.py`** : tests dédup (id déjà présent → ignoré) et
  reprise (relance après ajout partiel → pas de doublons/perte).
- **`pipeline/prefilter.py`** : tests unitaires purs sur `keywords.yaml`.
- **`pipeline/llm_classifier.py`** : client HTTP Ollama mocké (JSON valide,
  JSON invalide déclenchant retry, échec après 2 essais →
  `classification_echec=true`). Test d'intégration séparé, marqué
  `@pytest.mark.integration` (skip par défaut), contre un vrai Ollama local
  pour vérifier le respect du schéma structuré par le modèle réel.
- **`pipeline/export.py`** : tests sur données `ClassifiedPost` synthétiques
  pour vérifier les agrégations.
- TDD (test avant code) pour tous les modules ci-dessus ; les scrapers
  suivent le même principe avec les fixtures HTML comme point de départ.

## Hors scope

- `forum-auto.caradisiac.com` (Cloudflare, abandonné — voir Sources).
- Parallélisation de la classification LLM (un seul modèle chargé en mémoire
  sur cette machine, pas de gain réel).
- Score de confiance continu (remplacé par l'énumération
  haute/moyenne/basse, plus fiable à produire par un modèle 7B).
