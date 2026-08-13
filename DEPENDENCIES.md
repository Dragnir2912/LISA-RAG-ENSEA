# Dépendances du projet LISA / RAG ENSEA

Ce document explique le rôle des dépendances Python de `requirements.txt` et
`requirements-server.txt`. Pour lancer l’application complète, Docker est la
procédure recommandée : suivre le [guide Windows](docs/deploiement_local.md) ou
le [guide Linux](docs/deploiement_test.md). L’installation Python directe sert
surtout aux notebooks, aux tests et au développement.

## Scraping (collecte des documents source)

- **crawlee[beautifulsoup]** — moteur de crawl utilisé pour parcourir le site de l'ENSEA (`src/scraper.py`) : gère la file d'URLs, la concurrence, le respect du `robots.txt` et l'extraction HTML via BeautifulSoup.
- **apify-client** — client de l'API Apify, utilisé pour piloter l'acteur `facebook-posts-scraper` (`src/facebook_scraper.py`) qui récupère les publications Facebook des pages institutionnelles. Nécessite un jeton `APIFY_TOKEN` en variable d'environnement.
- **beautifulsoup4** — parsing HTML générique (extraction de liens, de tableaux, de contenu structuré) utilisé par le crawler et les nettoyeurs.
- **lxml** — moteur de parsing rapide (C) utilisé en arrière-plan par BeautifulSoup et pdfplumber pour accélérer le traitement HTML/XML.
- **httpx** — client HTTP asynchrone, utilisé pour les téléchargements de pages/PDF pendant le crawl.
- **pypdf** — lecture/extraction basique de PDF (texte, métadonnées, découpage de pages).
- **pdfplumber** — extraction plus fine de PDF (texte positionné, tableaux), utilisée pour les documents économiques et administratifs déposés localement.
- **openpyxl** — lecture/écriture de fichiers Excel (`.xlsx`), utile pour l'import de données tabulaires (banque de questions, exports).
- **python-docx** — lecture/écriture de documents Word (`.docx`), utilisé pour générer les rapports d'évaluation et le "golden set" au format Word.
- **reportlab** — génération de PDF programmatique, utilisée par `src/evaluation_report.py` pour produire les rapports d'audit en PDF.

## Nettoyage des documents

- **trafilatura** — extraction du contenu principal d'une page web (suppression du bruit : menus, publicités) et conversion en texte/Markdown propre avant chunking.
- **markdownify** — conversion HTML → Markdown ; conservée comme utilitaire de nettoyage optionnel disponible dans la chaîne de traitement (non appelée directement dans le code Python actuel, mais utile pour un futur traitement manuel/notebook).
- **python-frontmatter** — lecture/écriture de fichiers Markdown avec en-tête YAML (métadonnées de chaque document nettoyé : source, date, type, etc.).
- **requests** — appel HTTP synchrone simple, utilisé ponctuellement dans les scripts de génération de notebooks (`scripts/build_rag_workflow_notebooks.py`) pour tester un endpoint.

## Chunking (découpage en passages)

- **langchain-text-splitters** — algorithmes de découpage de texte en chunks (par taille, par structure) utilisés avant l'indexation vectorielle.

## Embeddings / reranking

- **torch** — moteur de calcul requis par les modèles d’embedding et de reranking ; le CPU est pleinement pris en charge, tandis qu’un GPU CUDA réduit la latence.
- **transformers** — bibliothèque Hugging Face pour charger les modèles de langage/embeddings (bridée à `<5` car FlagEmbedding n'est pas encore compatible avec Transformers v5).
- **FlagEmbedding** — bibliothèque BAAI pour les modèles d'embeddings denses et le reranking (BGE).
- **sentence-transformers** — génération d'embeddings de phrases/documents, utilisée en alternative ou complément de FlagEmbedding.
- **einops** — opérations de manipulation de tenseurs requises par certains modèles chargés avec `trust_remote_code=True` (ex. nomic-embed-text, jina-embeddings-v3).

## Vector store / recherche hybride

- **lancedb** — base de données vectorielle locale (fichier) utilisée pour stocker et interroger les embeddings des chunks (`src/vectorstore.py`).
- **rank-bm25** — implémentation de BM25 pour la recherche lexicale (sparse), combinée à la recherche vectorielle dans une stratégie de retrieval hybride (`src/retriever.py`).
- **pyarrow** — format de données colonnaire requis en interne par LanceDB pour le stockage.

## Génération (LLM)

- **ollama** — client Python qui pilote le serveur Ollama local et Qwen. Dans le déploiement Docker, Ollama est un conteneur privé de la même pile ; aucun service cloud n’est requis.

## Configuration / typage

- **pydantic** — validation et typage des données de configuration et des schémas d'API (requêtes/réponses FastAPI).
- **pyyaml** — lecture du fichier central `config.yaml` qui pilote tout le pipeline.

## Évaluation du pipeline RAG

- **ragas** — framework d'évaluation de systèmes RAG (fidélité, pertinence, rappel de contexte) utilisé pour scorer automatiquement les réponses générées.
- **datasets** — structure de données Hugging Face utilisée en interne par ragas pour manipuler les jeux d'évaluation.
- **langchain-ollama** — intégration LangChain pour utiliser Ollama comme LLM/juge dans les évaluations ragas.
- **langchain-community** — dépendance interne optionnelle requise par ragas pour certains connecteurs LangChain.

## Notebooks et utilitaires

- **jupyter** — environnement notebook utilisé pour explorer le pipeline pas à pas (`notebooks/`).
- **ipywidgets** — widgets interactifs dans les notebooks (barres de progression, contrôles).
- **tqdm** — barres de progression en ligne de commande et dans les notebooks, utilisées pendant les traitements longs (scraping, embeddings, évaluation).
- **pandas** — manipulation de données tabulaires (banque de questions, résultats d'évaluation, rapports).
- **numpy** — calcul numérique, utilisé notamment pour les opérations sur les vecteurs d'embeddings.

## Application web et déploiement serveur

- **fastapi** — framework web utilisé pour exposer l'API privée (`backend/main.py`) : authentification, documents, chat et administration.
- **uvicorn[standard]** — serveur ASGI qui fait tourner l'application FastAPI en production (`run_app.py`).
- **python-multipart** — gestion des formulaires et uploads PDF.
- **psycopg[binary]** — pilote PostgreSQL du profil serveur.
- **argon2-cffi** — dérivation Argon2id des mots de passe.
- **filelock** — verrou inter-processus protégeant le GPU et les index.
- **cryptography** — chiffrement des secrets éphémères transmis de l'API au worker.

## Dépendances de la bibliothèque standard (non listées dans requirements.txt)

Le projet utilise aussi largement la bibliothèque standard Python (`sqlite3`, `hashlib`, `hmac`, `secrets`, `pathlib`, `json`, `logging`, `asyncio`, etc.), notamment pour :
- la base applicative locale ou PostgreSQL (`backend/database.py`) ;
- les sessions opaques, jetons CSRF et de récupération (`backend/security.py`).

Aucune installation supplémentaire n'est nécessaire pour ces parties.

## Éléments d'environnement à prévoir pour un déploiement (hors requirements.txt)

- **Python 3.11** est la version de référence des images Docker. Python 3.10+ reste compatible pour les outils locaux.
- **Ollama local** est inclus par Docker. Hors Docker, il doit être installé et le modèle configuré doit être présent localement.
- **GPU facultatif** : le profil CPU fonctionne partout ; CUDA accélère les modèles sans changer les données ni les réponses attendues.
- **Variables d'environnement** : voir `.env.test.example` et `docs/deploiement_test.md` ; les secrets réels sont montés depuis `secrets/` et ne sont pas versionnés.
- **Espace disque** : `data/` conserve les documents et index, `models/` les modèles, et PostgreSQL les comptes/conversations du profil serveur. Ces trois emplacements doivent être persistants.

## Installation Python pour notebooks et développement

Cette installation ne lance ni PostgreSQL, ni Ollama, ni le frontend. Pour
l’application complète, utiliser Docker.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
```

### Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
```

Sélectionner ensuite le noyau `.venv` dans Jupyter. Les modèles Hugging Face
sont téléchargés au premier usage si leur cache local n’est pas déjà rempli ;
le mode conteneurisé prépare explicitement ce cache pour le fonctionnement hors
ligne.
