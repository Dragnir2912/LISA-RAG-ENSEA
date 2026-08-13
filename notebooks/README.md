# Notebooks du projet RAG ENSEA

Ce dossier contient deux ensembles complémentaires :

1. **Le parcours principal**, qui explique et exécute les étapes du RAG dans leur ordre réel.
2. **Les expériences historiques**, qui conservent les benchmarks ayant servi à choisir les meilleurs réglages.

Pour comprendre le projet, commencez par [`00_PARCOURS_RAG.ipynb`](parcours_rag/00_PARCOURS_RAG.ipynb). Il sert de sommaire et indique quoi exécuter, dans quel ordre, ainsi que les entrées et sorties de chaque étape.

## Parcours principal

| Ordre | Notebook | Rôle | Entrée principale | Sortie principale |
|---:|---|---|---|---|
| 1 | [`01_collecte_institutionnelle.ipynb`](parcours_rag/01_collecte_institutionnelle.ipynb) | Découvrir les URL et collecter pages HTML/PDF | Sitemaps ENSEA | Documents JSON bruts |
| 2 | [`02_collecte_facebook.ipynb`](parcours_rag/02_collecte_facebook.ipynb) | Collecter et contrôler les publications des clubs | Pages Facebook via Apify | Publications JSONL |
| 3 | [`03_import_pdf_eco_rci.ipynb`](parcours_rag/03_import_pdf_eco_rci.ipynb) | Importer les PDF économiques locaux | `data/sources/eco_rci` | Records PDF dans `data/raw` |
| 4 | [`04_nettoyage_et_qualite.ipynb`](parcours_rag/04_nettoyage_et_qualite.ipynb) | Retirer le bruit et produire un corpus lisible | Documents bruts | Documents Markdown nettoyés |
| 5 | [`05_chunking_parent_enfant.ipynb`](parcours_rag/05_chunking_parent_enfant.ipynb) | Découper et séparer les trois corpus | Markdown nettoyé | Chunks ENSEA, économiques et Facebook |
| 6 | [`06_embeddings_bge_m3.ipynb`](parcours_rag/06_embeddings_bge_m3.ipynb) | Transformer les chunks en vecteurs multilingues | Textes des chunks enfants | Matrices d'embeddings 1024D |
| 7 | [`07_indexation_lancedb_bm25.ipynb`](parcours_rag/07_indexation_lancedb_bm25.ipynb) | Construire les trois index ENSEA, économique et Facebook | Chunks + embeddings | Trois paires LanceDB + BM25 |
| 8 | [`08_retrieval_hybride_et_reranking.ipynb`](parcours_rag/08_retrieval_hybride_et_reranking.ipynb) | Expliquer la recherche hybride, MMR et le reranking | Question utilisateur + index | Sources pertinentes classées |
| 9 | [`09_generation_citations_et_conversation.ipynb`](parcours_rag/09_generation_citations_et_conversation.ipynb) | Construire le contexte, générer et citer les sources | Question + passages récupérés | Réponse conversationnelle citée |
| 10 | [`10_evaluation_retrieval_et_generation.ipynb`](parcours_rag/10_evaluation_retrieval_et_generation.ipynb) | Mesurer précision, couverture, qualité et latence | Golden set + pipeline | Rapport d'évaluation |
| 11 | [`11_observabilite.ipynb`](parcours_rag/11_observabilite.ipynb) | Inspecter traces, temps par étape et retours utilisateur | Journaux JSONL + base applicative | Diagnostic de fonctionnement |
| 12 | [`12_ajout_pdf_et_index_incremental.ipynb`](parcours_rag/12_ajout_pdf_et_index_incremental.ipynb) | Montrer l'ajout de PDF sans reconstruire tout l'index | PDF téléversé | Nouveaux chunks indexés |
| 13 | [`13_rag_ensea_end_to_end.ipynb`](parcours_rag/13_rag_ensea_end_to_end.ipynb) | Charger et tester le système complet | Index validés | Réponses finales du RAG |

## Règles d'exécution

- Ouvrez Jupyter depuis la racine du projet et sélectionnez le noyau Python de
  `.venv`. L’installation directe est décrite dans [`../DEPENDENCIES.md`](../DEPENDENCIES.md).
- Arrêtez l'application et fermez tous les autres kernels avant la reconstruction.
- L'application restera indisponible jusqu'à la création des nouveaux index par le notebook 07.
- Chaque opération modifiant des données est signalée par un drapeau `RUN_*`.
  Vérifiez sa valeur dans la cellule avant de l’exécuter.
- Les collectes réseau des notebooks 01 et 02 sont désactivées par défaut
  (`RUN_SITEMAP_DISCOVERY`, `RUN_INSTITUTIONAL_COLLECTION` et
  `RUN_FACEBOOK_COLLECTION` à `False`). Elles ne doivent être activées qu’avec
  Internet et, pour Facebook, un jeton Apify valide.
- La préparation des publications déjà collectées et la reconstruction locale
  des notebooks 02 à 07 sont actuellement activées. Ces étapes réécrivent les
  artefacts actifs de `data/raw`, `data/cleaned`, `data/chunks`,
  `data/embeddings`, `data/economic` et `data/facebook`.
- Les notebooks créent leurs sous-dossiers de sortie lorsqu’ils manquent, mais
  ils ne recréent pas les sources. Conservez donc `data/sources/`, les golden
  sets et les fichiers importés avant une reconstruction.
- Fermez les autres kernels avant de charger les modèles GPU, particulièrement sur une RTX 5060 Laptop de 8 Go.
- Le notebook racine [`../rag_ensea.ipynb`](../rag_ensea.ipynb) reste le notebook final d'utilisation quotidienne.

### Où reprendre après une interruption

- interruption avant la fin du notebook 04 : relancer le notebook interrompu ;
- interruption pendant 05 ou 06 : relancer ce notebook en entier pour éviter
  des chunks ou matrices incomplets ;
- interruption pendant 07 : relancer 07 avant d’utiliser le RAG ;
- après 07 validé : 08 à 13 ne reconstruisent pas les corpus sauf si un drapeau
  de test explicitement activé l’indique.

## Expériences historiques

Les notebooks du dossier [`test&comparisons`](test&comparisons) documentent les essais qui ont conduit au système actuel : variantes de nettoyage et de chunking, benchmarks de retrieval, élargissement du jeu d'évaluation, abstention, Facebook et génération routée.

Ils sont conservés pour la traçabilité scientifique. Ils ne constituent pas un tutoriel linéaire et ne doivent pas être exécutés au hasard pour reconstruire le RAG.

## Reproduire les notebooks du parcours

Le script [`../scripts/build_rag_workflow_notebooks.py`](../scripts/build_rag_workflow_notebooks.py) régénère la structure pédagogique. Les résultats d'exécution ne sont volontairement pas intégrés par le script : ils restent sous le contrôle de l'utilisateur dans Jupyter.
