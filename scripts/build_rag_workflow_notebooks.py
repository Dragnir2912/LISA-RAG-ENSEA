"""Construit le parcours pédagogique RAG sans modifier les notebooks d'expérimentation."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
WORKFLOW = NOTEBOOKS / "parcours_rag"


def _lines(text: str) -> list[str]:
    clean = textwrap.dedent(text).strip("\n")
    return [line + "\n" for line in clean.splitlines()]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def bootstrap() -> str:
    return """
    from pathlib import Path
    import sys

    def find_project_root(start=Path.cwd()):
        start = start.resolve()
        for candidate in (start, *start.parents):
            if (candidate / 'config.yaml').exists() and (candidate / 'src').is_dir():
                return candidate
        raise FileNotFoundError('Impossible de trouver config.yaml et src/.')

    PROJECT_ROOT = find_project_root()
    SRC_DIR = PROJECT_ROOT / 'src'
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    print('Projet :', PROJECT_ROOT)
    print('Python :', sys.executable)
    print('Données actives :', PROJECT_ROOT / 'data')
    """


def nb(title: str, purpose: str, cells: list[dict]) -> dict:
    return {
        "cells": [
            md(f"# {title}\n\n{purpose}\n\n> Vérifiez toujours les drapeaux `RUN_*` avant l'exécution. Pour cette reconstruction propre, les traitements locaux nécessaires des étapes 02 à 07 sont déjà activés ; les collectes réseau et évaluations GPU facultatives restent désactivées."),
            *cells,
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build() -> dict[Path, dict]:
    files: dict[Path, dict] = {}

    files[WORKFLOW / "00_PARCOURS_RAG.ipynb"] = nb(
        "Parcours complet du RAG ENSEA",
        "Point d'entrée du projet. Ce notebook présente toutes les étapes, leurs entrées, leurs sorties et l'ordre recommandé.",
        [
            md("""
            ## Architecture suivie

            `Sites ENSEA + Facebook + bibliothèque PDF économique → nettoyage → chunks parent-enfant → embeddings → LanceDB + BM25 → retrieval hybride → reranking → génération citée → évaluation → observabilité`

            Le dossier `parcours_rag/` contient le chemin principal. Les notebooks numérotés directement dans `notebooks/` restent les expériences historiques qui ont servi à sélectionner les paramètres.
            """),
            code(bootstrap()),
            code("""
            import pandas as pd
            from config import load_config

            config = load_config(PROJECT_ROOT / 'config.yaml')
            stages = [
                (1, 'Collecte institutionnelle', 'HTML, PDF', 'data/raw, data/pdf'),
                (2, 'Collecte Facebook', 'Pages clubs', 'data/facebook/raw'),
                (3, 'Import PDF économiques', 'data/sources/eco_rci', 'records PDF dans data/raw'),
                (4, 'Nettoyage', 'Documents bruts', 'Markdown + métadonnées'),
                (5, 'Chunking', 'Markdown', 'chunks ENSEA et ECO RCI séparés'),
                (6, 'Embeddings', 'Chunks enfants', 'trois matrices BGE-M3'),
                (7, 'Indexation', 'Chunks + vecteurs', 'trois couples LanceDB + BM25'),
                (8, 'Retrieval', 'Question', 'passages rerankés'),
                (9, 'Génération', 'Question + passages', 'réponse + citations'),
                (10, 'Évaluation', 'Golden set', 'Recall, MRR, fidélité, latence'),
                (11, 'Observabilité', 'Traces', 'cache, erreurs, P95, routes'),
                (12, 'Mises à jour', 'PDF ajouté', 'index incrémental'),
                (13, 'RAG complet', 'Tous les composants', 'assistant utilisable'),
            ]
            display(pd.DataFrame(stages, columns=['étape', 'composant', 'entrée', 'sortie']))
            """),
            code("""
            artifacts = {
                'documents_bruts': config.project.data_path / 'raw',
                'bibliothèque_eco_rci': config.local_documents.source_full_path,
                'documents_nettoyés': config.project.data_path / 'cleaned',
                'chunks': config.project.data_path / 'chunks' / 'chunks.jsonl',
                'lancedb': config.vectorstore.uri_path,
                'bm25': config.vectorstore.bm25_index_full_path,
                'chunks_économiques': config.local_documents.chunks_full_path,
                'lancedb_économique': PROJECT_ROOT / config.local_documents.vectorstore_uri,
                'bm25_économique': PROJECT_ROOT / config.local_documents.bm25_index_path,
                'golden_set': config.evaluation.expanded_golden_set_full_path,
                'traces': config.observability.traces_full_path,
            }
            display(pd.DataFrame([
                {'artefact': name, 'chemin': str(path), 'présent': path.exists()}
                for name, path in artifacts.items()
            ]))
            """),
            md("""
            ## Ordre d'exécution

            Arrêtez l'application FastAPI et fermez les autres kernels avant de commencer. N'essayez pas d'utiliser l'assistant avant la fin du notebook 07 : les anciens index ont été volontairement supprimés.

            1. Exécutez les notebooks 01 à 07 pour reconstruire les corpus et les trois index.
            2. Les notebooks 08 et 09 expliquent le chemin d'une question jusqu'à la réponse.
            3. Les notebooks 10 et 11 mesurent la qualité et le comportement du système.
            4. Les notebooks 12 et 13 couvrent l'exploitation quotidienne et le pipeline final.

            Les notebooks 02 à 07 contiennent les opérations actives de reconstruction. Laissez chaque opération se terminer et sauvegardez le notebook avant de passer au suivant. Le notebook 06 charge BGE-M3 sur le GPU : aucun autre kernel ne doit occuper la VRAM.
            """),
        ],
    )

    files[WORKFLOW / "01_collecte_institutionnelle.ipynb"] = nb(
        "01 — Collecte institutionnelle : site web et PDF",
        "Montre la découverte par sitemap, l'extraction HTML structurée, le téléchargement des PDF et la provenance enregistrée pour chaque document.",
        [
            code(bootstrap()),
            md("""
            ## Entrées et sorties

            - Entrées : sitemap et pages des domaines ENSEA.
            - Sorties HTML/PDF extraites : JSON dans `raw/` et fichiers originaux dans `pdf/`.
            - Métadonnées : URL, titre, titres de sections, liens, tableaux, dates et empreinte du contenu.
            """),
            code("""
            import json
            from table_facts import is_plausible_table_fact
            import pandas as pd
            from config import load_config

            config = load_config(PROJECT_ROOT / 'config.yaml')
            raw_files = sorted((config.project.data_path / 'raw').glob('*.json'))
            rows = []
            for path in raw_files[:200]:
                record = json.loads(path.read_text(encoding='utf-8'))
                rows.append({
                    'url': record.get('url'),
                    'title': record.get('title'),
                    'characters': len(record.get('text', '')),
                    'headings': len(record.get('headings', [])),
                    'type': record.get('metadata', {}).get('doc_type', 'page'),
                })
            print('Documents bruts présents :', len(raw_files))
            display(pd.DataFrame(rows).head(10))
            """),
            code("""
            from scraper import SitemapDiscoverer

            RUN_SITEMAP_DISCOVERY = False
            if RUN_SITEMAP_DISCOVERY:
                urls = SitemapDiscoverer(config.project.sitemap_url).discover()
                print('URLs découvertes :', len(urls))
                display(pd.Series(urls[:20], name='url').to_frame())
            else:
                print('Découverte réseau désactivée. Passez RUN_SITEMAP_DISCOVERY à True pour la lancer.')
            """),
            code("""
            from scraper import EnseaScraper

            RUN_INSTITUTIONAL_COLLECTION = False
            if RUN_INSTITUTIONAL_COLLECTION:
                await EnseaScraper(config).run()
                print('Collecte institutionnelle terminée dans', config.project.data_path)
            else:
                print('Collecte réseau désactivée ; les données brutes déjà présentes seront réutilisées.')
            """),
            md("""
            ## Contrôles attendus

            Vérifiez la présence du texte principal, des dates lorsqu'elles existent, des URL permanentes et des PDF. Une collecte réussie ne signifie pas encore que le texte est propre : le nettoyage intervient au notebook 04.
            """),
        ],
    )

    files[WORKFLOW / "02_collecte_facebook.ipynb"] = nb(
        "02 — Collecte des publications Facebook",
        "Explique la collecte Apify des pages ENSEA, clubs et associations, puis contrôle les dates, permaliens et doublons.",
        [
            code(bootstrap()),
            code("""
            import json
            import os
            import pandas as pd
            from config import load_config

            config = load_config(PROJECT_ROOT / 'config.yaml')
            facebook = config.facebook_scraping
            rows = []
            for path in sorted(facebook.raw_full_path.glob('*.jsonl')):
                for line in path.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
            posts = pd.DataFrame(rows)
            print('Publications normalisées :', len(posts))
            if not posts.empty:
                display(posts.groupby('page_key').size().rename('publications').to_frame())
                display(posts[['page_key', 'published_at', 'post_url', 'text']].head())
            """),
            md("""
            ## Pourquoi un index séparé ?

            Facebook contient des informations datées et moins institutionnelles. Les publications sont donc nettoyées, contrôlées et indexées séparément. Le routeur n'utilise cet index que lorsqu'une activité ou un club est explicitement nommé.
            """),
            code("""
            from getpass import getpass
            from dataclasses import asdict
            from facebook_scraper import FacebookScraper

            RUN_FACEBOOK_COLLECTION = False
            if RUN_FACEBOOK_COLLECTION:
                token = os.getenv('APIFY_TOKEN') or getpass('Jeton Apify : ')
                scraper = FacebookScraper(config, api_token=token)
                reports = scraper.run(max_posts_per_page=facebook.max_posts_per_page)
                display(pd.DataFrame([asdict(report) for report in reports]))
            else:
                print('Collecte Apify désactivée ; les fichiers existants sont seulement inspectés.')
            """),
            code("""
            from facebook_processor import FacebookPostProcessor

            RUN_FACEBOOK_PREPARATION = True
            if RUN_FACEBOOK_PREPARATION:
                report = FacebookPostProcessor(config).process_all()
                display(pd.Series(report).to_frame('valeur'))
            else:
                print('Préparation désactivée : aucun chunk Facebook ne sera reconstruit.')
            """),
            md("""
            ## Qualité minimale

            Chaque publication conservée doit avoir un texte substantiel, une page d'origine, une date, un permalien et une empreinte unique. Les republications identiques et les textes trop courts sont retirés.
            """),
        ],
    )

    files[WORKFLOW / "03_import_pdf_eco_rci.ipynb"] = nb(
        "03 — Import des documents économiques locaux",
        "Transforme les PDF et classeurs Excel de la bibliothèque ECO RCI en enregistrements bruts et en faits tabulaires atomiques.",
        [
            code(bootstrap()),
            code("""
            import pandas as pd
            from dataclasses import asdict
            from config import load_config

            config = load_config(PROJECT_ROOT / 'config.yaml')
            source_dir = config.local_documents.source_full_path
            files = sorted(path for path in source_dir.rglob('*') if path.is_file())
            pdfs = [path for path in files if path.suffix.casefold() == '.pdf']
            spreadsheets = [path for path in files if path.suffix.casefold() in {'.xlsx', '.xlsm'}]
            supported_documents = pdfs + spreadsheets
            display(pd.Series({
                'dossier_source': str(source_dir),
                'fichiers': len(files),
                'pdf': len(pdfs),
                'tableurs': len(spreadsheets),
                'autres_fichiers': len(files) - len(supported_documents),
                'taille_Mo': sum(path.stat().st_size for path in files) / 1024**2,
            }).to_frame('valeur'))
            display(pd.DataFrame({
                'source': [path.relative_to(source_dir).parts[0] for path in supported_documents],
                'catégorie': [path.relative_to(source_dir).parent.as_posix() for path in supported_documents],
                'format': [path.suffix.casefold() for path in supported_documents],
            }).groupby(['source', 'catégorie', 'format']).size().rename('documents').to_frame())
            """),
            code("""
            from local_documents import LocalPDFImporter

            RUN_LOCAL_PDF_IMPORT = True
            if RUN_LOCAL_PDF_IMPORT:
                report = LocalPDFImporter(config).import_all(show_progress=True)
                display(pd.Series(asdict(report)).to_frame('valeur'))
                if report.failures:
                    display(pd.DataFrame(report.failures))
                assert report.discovered_pdfs > 0
                success_ratio = (report.imported_pdfs + report.reused_pdfs) / report.discovered_pdfs
                assert success_ratio >= 0.90, 'Moins de 90 % des PDF ont pu être importés.'
                if report.failed_pdfs:
                    print(f'AVERTISSEMENT : {report.failed_pdfs} PDF nécessitent un OCR ou un nouveau téléchargement.')
                assert report.discovered_spreadsheets > 0, 'Aucun tableur local détecté.'
                assert (
                    report.imported_spreadsheets + report.reused_spreadsheets
                    == report.discovered_spreadsheets
                ), 'Certains tableurs n’ont pas été importés.'
            """),
            md("""
            ## Résultat transmis à la suite

            Deux barres de progression affichent le PDF ou le tableur en cours, les pages ou feuilles déjà lues, le moteur d'extraction et les compteurs d'import, de réutilisation, de doublons et d'erreurs.

            Les originaux restent dans `data/sources/eco_rci`. Les records régénérables `local_pdf_*.json` et `local_sheet_*.json` sont ajoutés à `data/raw`. Leur URL pointe vers l'API locale afin que les citations ouvrent le document original.

            Les feuilles Excel sont segmentées à chaque nouvel en-tête temporel, puis transformées dans le même contrat de faits atomiques que les tableaux PDF. Le rapport complet est écrit dans `data/import_reports/eco_rci.json`.
            """),
            md("""
            ## Extraction structurée des tableaux statistiques

            Chaque cellule exploitable devient un fait autonome : `indicateur + période + valeur + unité + territoire + type de mesure + page ou feuille`. Les tableaux PDF paysages sont remis dans leur repère logique avant extraction. Une ligne incohérente est rejetée avant le chunking et l'embedding.
            """),
            code("""
            import json

            local_records = [
                json.loads(path.read_text(encoding='utf-8'))
                for pattern in ('local_pdf_*.json', 'local_sheet_*.json')
                for path in sorted(config.local_documents.raw_full_path.glob(pattern))
            ]
            facts = [
                {'document': record.get('title', ''), 'url': record.get('url', ''), **fact}
                for record in local_records
                for fact in record.get('table_facts', [])
            ]
            table_facts_df = pd.DataFrame(facts)
            display(pd.Series({
                'documents_locaux_importés': len(local_records),
                'faits_tabulaires': len(table_facts_df),
                'pages_ou_feuilles': (
                    table_facts_df[['url', 'page']].drop_duplicates().shape[0]
                    if not table_facts_df.empty else 0
                ),
                'faits_depuis_tableurs': (
                    int(table_facts_df['source_method'].eq('spreadsheet_matrix').sum())
                    if not table_facts_df.empty else 0
                ),
            }).to_frame('valeur'))
            required = {'indicator', 'period', 'value', 'unit', 'territory', 'measure_type', 'page'}
            assert not table_facts_df.empty, 'Aucun fait tabulaire extrait.'
            assert required <= set(table_facts_df.columns), 'Facettes tabulaires incomplètes.'
            assert table_facts_df['source_method'].eq('spreadsheet_matrix').any(), (
                'Aucun fait issu des feuilles Excel n’a été extrait.'
            )
            invalid_facts = [
                fact
                for record in local_records
                for fact in record.get('table_facts', [])
                if not is_plausible_table_fact(fact)
            ]
            assert not invalid_facts, (
                f'{len(invalid_facts)} faits tabulaires mal formés ont franchi l’import.'
            )
            """),
        ],
    )

    files[WORKFLOW / "04_nettoyage_et_qualite.ipynb"] = nb(
        "04 — Nettoyage et contrôle qualité",
        "Transforme les JSON bruts en Markdown propre, retire navigation et boilerplate, puis mesure la qualité du corpus.",
        [
            code(bootstrap()),
            code("""
            import frontmatter
            import pandas as pd
            from config import load_config
            from cleaner import HTMLCleaner

            config = load_config(PROJECT_ROOT / 'config.yaml')
            CLEANED_DIR = config.project.data_path / 'cleaned'
            RUN_CLEANING = True
            if RUN_CLEANING:
                cleaner = HTMLCleaner(config)
                cleaner.raw_dir = config.project.data_path / 'raw'
                cleaner.cleaned_dir = CLEANED_DIR
                cleaned_documents = cleaner.clean_all(clear_output=True)
                print('Documents nettoyés :', len(cleaned_documents))
            print('Corpus analysé :', CLEANED_DIR)
            """),
            code("""
            records = []
            for path in sorted(CLEANED_DIR.glob('*.md')):
                document = frontmatter.load(path)
                records.append({
                    **dict(document.metadata),
                    'file': path.name,
                    'characters': len(document.content),
                    'lines': len(document.content.splitlines()),
                })
            cleaned_df = pd.DataFrame(records)
            summary = pd.Series({
                'documents': len(cleaned_df),
                'urls_uniques': cleaned_df['url'].nunique() if not cleaned_df.empty else 0,
                'caractères_médians': int(cleaned_df['characters'].median()) if not cleaned_df.empty else 0,
                'documents_trop_courts': int(cleaned_df['characters'].lt(config.cleaning.min_content_length).sum()) if not cleaned_df.empty else 0,
            })
            display(summary.to_frame('valeur'))
            """),
            code("""
            if not cleaned_df.empty:
                display(cleaned_df.groupby(['doc_type', 'extraction_method']).size().rename('documents').to_frame())
                display(cleaned_df.nlargest(10, 'characters')[['title', 'category', 'characters', 'url']])
                assert cleaned_df['url'].is_unique
                assert cleaned_df['characters'].ge(config.cleaning.min_content_length).all()
                print('Contrôles de structure validés.')
            """),
            code("""
            markdown_list_items = 0
            documents_with_lists = 0
            for path in sorted(CLEANED_DIR.glob('*.md')):
                content = frontmatter.load(path).content
                item_count = sum(
                    line.lstrip().startswith(('- ', '* '))
                    for line in content.splitlines()
                )
                markdown_list_items += item_count
                documents_with_lists += int(item_count > 0)
            display(pd.Series({
                'documents_avec_listes_markdown': documents_with_lists,
                'items_de_liste_preserves': markdown_list_items,
            }).to_frame('valeur'))
            assert markdown_list_items > 0, (
                "Aucune liste HTML n'a été conservée en Markdown."
            )
            """),
            md("""
            ## Résultat

            Un document nettoyé reste une unité documentaire complète. Il conserve ses métadonnées en front-matter YAML ; il n'est pas encore découpé ni vectorisé. Les listes HTML restent des listes Markdown séparées afin de préserver leur sens.
            """),
        ],
    )

    files[WORKFLOW / "05_chunking_parent_enfant.ipynb"] = nb(
        "05 — Chunking hiérarchique parent-enfant",
        "Montre comment les sections Markdown deviennent de petits chunks de recherche reliés à de grands parents transmis au générateur.",
        [
            code(bootstrap()),
            code("""
            import json
            import numpy as np
            import pandas as pd
            from chunker import Chunk, HierarchicalChunker
            from config import load_config

            config = load_config(PROJECT_ROOT / 'config.yaml')
            cleaned_dir = config.project.data_path / 'cleaned'
            output_path = config.project.data_path / 'chunks' / 'chunks.jsonl'
            economic_output_path = config.local_documents.chunks_full_path
            facebook_cleaned_dir = config.facebook_scraping.cleaned_candidate_full_path
            facebook_output_path = config.facebook_scraping.chunks_candidate_full_path
            combined_path = config.project.data_path / 'chunks' / '_combined_rebuild.jsonl'
            RUN_CHUNKING = True
            if RUN_CHUNKING:
                all_chunks = HierarchicalChunker(config).chunk_all(cleaned_dir, combined_path)
                economic_prefix = config.local_documents.public_route_prefix.rstrip('/') + '/'
                document_chunks = [chunk for chunk in all_chunks if not chunk.url.startswith(economic_prefix)]
                economic_chunks = [chunk for chunk in all_chunks if chunk.url.startswith(economic_prefix)]

                def write_chunks(path, items):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('w', encoding='utf-8') as handle:
                        for item in items:
                            handle.write(json.dumps(item.__dict__, ensure_ascii=False) + '\\n')

                write_chunks(output_path, document_chunks)
                write_chunks(economic_output_path, economic_chunks)
                combined_path.unlink(missing_ok=True)
                facebook_chunks = HierarchicalChunker(config).chunk_all(
                    facebook_cleaned_dir,
                    facebook_output_path,
                )
                chunks = all_chunks + facebook_chunks
                print(
                    'Chunks ENSEA :', len(document_chunks),
                    '| économiques :', len(economic_chunks),
                    '| Facebook :', len(facebook_chunks),
                )
                assert facebook_chunks, (
                    'Aucun chunk Facebook produit. Exécutez le notebook '
                    '02_collecte_facebook avant le notebook 05.'
                )
            else:
                chunks = [
                    Chunk(**json.loads(line))
                    for path in (output_path, economic_output_path, facebook_output_path)
                    for line in path.read_text(encoding='utf-8').splitlines()
                    if line.strip()
                ]
            print('Chunks analysés :', len(chunks))
            """),
            code("""
            chunk_df = pd.DataFrame([chunk.__dict__ for chunk in chunks])
            child_tokens_approx = chunk_df['text'].str.len() / 4
            parent_tokens_approx = chunk_df['parent_text'].str.len() / 4
            display(pd.Series({
                'chunks': len(chunk_df),
                'parents': chunk_df['parent_id'].nunique(),
                'documents': chunk_df['url'].nunique(),
                'enfant_p50_tokens_approx': int(child_tokens_approx.median()),
                'enfant_p95_tokens_approx': int(np.percentile(child_tokens_approx, 95)),
                'parent_p50_tokens_approx': int(parent_tokens_approx.median()),
                'parent_p95_tokens_approx': int(np.percentile(parent_tokens_approx, 95)),
            }).to_frame('valeur'))
            display(chunk_df.groupby(['doc_type', 'category']).size().rename('chunks').to_frame())
            assert chunk_df['chunk_id'].is_unique
            """),
            code("""
            sample_parent = chunk_df['parent_id'].value_counts().index[0]
            sample = chunk_df[chunk_df['parent_id'].eq(sample_parent)]
            display(sample[['title', 'section', 'chunk_id', 'parent_id', 'text']])
            print('\\nParent transmis au LLM :\\n', sample.iloc[0]['parent_text'][:1500])
            """),
            md("""
            ## Compromis retenu

            Le chunk enfant d'environ 400 tokens améliore la précision de la recherche. Le parent pouvant atteindre environ 1 800 tokens restitue le contexte complet au LLM. Le chevauchement de 15 % évite de perdre une information coupée à une frontière. Les chunks ENSEA, économiques et Facebook sont écrits dans trois fichiers séparés afin de préserver le routage des sources.
            """),
            md("""
            ## Trois formes complémentaires de chunks

            - `text` conserve les passages narratifs ;
            - `table_fact` représente une cellule validée avec indicateur, période, valeur, unité, territoire, type de mesure et page ou feuille ;
            - `derived_fact` représente un calcul déterministe traçable : total annuel de flux trimestriels compatibles ou variation entre trimestres successifs.

            Les cellules mal formées et les lignes contradictoires sont rejetées avant embedding. Les formules et cellules sources d'un calcul restent dans son parent.
            """),
            code("""
            content_type_summary = chunk_df.groupby(['category', 'content_type']).size().rename('chunks').to_frame()
            display(content_type_summary)
            structured_chunks = chunk_df[chunk_df['content_type'].eq('table_fact')]
            derived_chunks = chunk_df[chunk_df['content_type'].eq('derived_fact')]
            display(pd.Series({
                'chunks_narratifs': int(chunk_df['content_type'].eq('text').sum()),
                'chunks_faits_tabulaires': len(structured_chunks),
                'chunks_faits_calculés': len(derived_chunks),
                'lignes_tabulaires_parentes': structured_chunks['row_id'].nunique(),
                'pages_tabulaires': structured_chunks[['url', 'page']].drop_duplicates().shape[0],
            }).to_frame('valeur'))
            required = {'indicator', 'period', 'value', 'unit', 'territory', 'measure_type', 'page'}
            assert required <= set(chunk_df.columns), 'Facettes structurées absentes des chunks.'
            assert not structured_chunks.empty, 'Aucun table_fact produit.'
            assert not derived_chunks.empty, 'Aucun derived_fact produit.'
            assert {'derived_annual_sum', 'derived_quarter_change'} <= set(derived_chunks['source_method'])
            """),
        ],
    )

    files[WORKFLOW / "06_embeddings_bge_m3.ipynb"] = nb(
        "06 — Calcul des embeddings BGE-M3",
        "Encode les chunks enfants en vecteurs multilingues normalisés et vérifie dimensions, normes et temps d'encodage.",
        [
            code(bootstrap()),
            code("""
            import json
            import time
            import numpy as np
            import pandas as pd
            from chunker import Chunk
            from config import load_config

            config = load_config(PROJECT_ROOT / 'config.yaml')
            chunk_path = config.project.data_path / 'chunks' / 'chunks.jsonl'
            chunks = [Chunk(**json.loads(line)) for line in chunk_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            economic_path = config.local_documents.chunks_full_path
            economic_chunks = [Chunk(**json.loads(line)) for line in economic_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            facebook_path = config.facebook_scraping.chunks_candidate_full_path
            facebook_chunks = [
                Chunk(**json.loads(line))
                for line in facebook_path.read_text(encoding='utf-8').splitlines()
                if line.strip()
            ] if facebook_path.exists() else []
            print('Modèle :', config.embeddings.model_name)
            print('Chunks ENSEA :', len(chunks), '| économiques :', len(economic_chunks))
            print('Chunks Facebook :', len(facebook_chunks), '| dimension attendue : 1024 | précision :', config.embeddings.precision)
            """),
            code("""
            from embeddings import EmbeddingModel

            RUN_SAMPLE_EMBEDDING = False
            if RUN_SAMPLE_EMBEDDING:
                model = EmbeddingModel(config.embeddings)
                sample_texts = [chunk.text for chunk in chunks[:16]]
                started = time.perf_counter()
                sample_vectors = model.encode(sample_texts)
                elapsed = time.perf_counter() - started
                display(pd.Series({
                    'forme': str(sample_vectors.shape),
                    'secondes': elapsed,
                    'norme_moyenne': float(np.linalg.norm(sample_vectors, axis=1).mean()),
                    'mémoire_Mo': sample_vectors.nbytes / 1024**2,
                }).to_frame('valeur'))
            else:
                print('Encodage GPU désactivé. Passez RUN_SAMPLE_EMBEDDING à True pour tester 16 chunks.')
            """),
            code("""
            RUN_EMBED_ALL = True
            embedding_path = config.project.data_path / 'embeddings' / 'document_vectors.npy'
            economic_embedding_path = config.local_documents.vectors_full_path
            facebook_embedding_path = config.project.data_path / 'facebook' / 'vectors.npy'
            if RUN_EMBED_ALL:
                if 'model' not in globals():
                    model = EmbeddingModel(config.embeddings)
                def encode_chunks(items):
                    batches = []
                    for start in range(0, len(items), config.embeddings.batch_size):
                        texts = [chunk.text for chunk in items[start:start + config.embeddings.batch_size]]
                        batches.append(model.encode(texts))
                    return np.concatenate(batches, axis=0)

                started = time.perf_counter()
                embedding_path.parent.mkdir(parents=True, exist_ok=True)
                vectors = encode_chunks(chunks)
                np.save(embedding_path, vectors)
                print('Matrice ENSEA :', vectors.shape)
                economic_embedding_path.parent.mkdir(parents=True, exist_ok=True)
                economic_vectors = encode_chunks(economic_chunks)
                np.save(economic_embedding_path, economic_vectors)
                print('Matrice économique :', economic_vectors.shape)
                if facebook_chunks:
                    facebook_embedding_path.parent.mkdir(parents=True, exist_ok=True)
                    facebook_vectors = encode_chunks(facebook_chunks)
                    np.save(facebook_embedding_path, facebook_vectors)
                    print('Matrice Facebook :', facebook_vectors.shape)
                print('Temps total :', round(time.perf_counter() - started, 2), 's')
                model.unload()
            else:
                print('Encodage complet désactivé ; sortie prévue :', embedding_path)
            """),
            md("""
            ## Particularité de BGE-M3

            BGE-M3 est multilingue, comprend bien le français et produit des vecteurs de 1 024 dimensions. La normalisation permet d'utiliser directement la similarité cosinus. Les embeddings décrivent les chunks enfants, pas les parents.
            """),
        ],
    )

    files[WORKFLOW / "07_indexation_lancedb_bm25.ipynb"] = nb(
        "07 — Indexation dense LanceDB et lexicale BM25",
        "Construit depuis zéro les index ENSEA, économique et Facebook utilisés par l'application.",
        [
            code(bootstrap()),
            code("""
            import json
            import numpy as np
            import pandas as pd
            from chunker import Chunk
            from config import load_config
            from vectorstore import VectorStoreManager

            config = load_config(PROJECT_ROOT / 'config.yaml')
            chunk_path = config.project.data_path / 'chunks' / 'chunks.jsonl'
            chunks = [Chunk(**json.loads(line)) for line in chunk_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            economic_path = config.local_documents.chunks_full_path
            economic_chunks = [Chunk(**json.loads(line)) for line in economic_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            facebook_path = config.facebook_scraping.chunks_candidate_full_path
            facebook_chunks = [Chunk(**json.loads(line)) for line in facebook_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            print('Chunks ENSEA :', len(chunks), '| économiques :', len(economic_chunks), '| Facebook :', len(facebook_chunks))
            """),
            code("""
            RUN_INDEXING = True
            embedding_path = config.project.data_path / 'embeddings' / 'document_vectors.npy'
            economic_embedding_path = config.local_documents.vectors_full_path
            facebook_embedding_path = config.project.data_path / 'facebook' / 'vectors.npy'
            if RUN_INDEXING:
                if not all(path.exists() for path in (embedding_path, economic_embedding_path, facebook_embedding_path)):
                    raise FileNotFoundError('Exécutez le notebook 06_embeddings_bge_m3 avant celui-ci.')
                vectors = np.load(embedding_path)
                economic_vectors = np.load(economic_embedding_path)
                facebook_vectors = np.load(facebook_embedding_path)
                assert len(vectors) == len(chunks)
                assert len(economic_vectors) == len(economic_chunks)
                assert len(facebook_vectors) == len(facebook_chunks)

                production_store = VectorStoreManager(config)
                production_store.index_all(chunks, vectors)

                economic_config = config.model_copy(deep=True)
                economic_config.vectorstore.uri = config.local_documents.vectorstore_uri
                economic_config.vectorstore.table_name = config.local_documents.vectorstore_table
                economic_config.vectorstore.bm25_index_path = config.local_documents.bm25_index_path
                economic_store = VectorStoreManager(economic_config)
                economic_store.index_all(economic_chunks, economic_vectors)

                facebook_config = config.model_copy(deep=True)
                facebook_config.vectorstore.uri = config.facebook_scraping.vectorstore_candidate_uri
                facebook_config.vectorstore.table_name = config.facebook_scraping.vectorstore_candidate_table
                facebook_config.vectorstore.bm25_index_path = config.facebook_scraping.bm25_candidate_path
                facebook_store = VectorStoreManager(facebook_config)
                facebook_store.index_all(facebook_chunks, facebook_vectors)
            else:
                raise RuntimeError('L’indexation est requise pour poursuivre le parcours.')
            """),
            code("""
            production_store.load_sparse()
            economic_store.load_sparse()
            facebook_store.load_sparse()
            display(pd.Series({
                'vecteurs_ENSEA': production_store.dense.count(),
                'ENSEA_BM25': len(production_store.sparse.chunk_ids),
                'vecteurs_économiques': economic_store.dense.count(),
                'économie_BM25': len(economic_store.sparse.chunk_ids),
                'vecteurs_Facebook': facebook_store.dense.count(),
                'publications_BM25': len(facebook_store.sparse.chunk_ids),
            }).to_frame('valeur'))
            dense_example = production_store.dense.metadata_rows()[:3]
            display(pd.DataFrame(dense_example)[['chunk_id', 'title', 'section', 'url']])
            """),
            md("""
            ## Pourquoi trois index ?

            Chaque source possède son couple LanceDB + BM25. La séparation protège les réponses ENSEA du bruit potentiel des longs rapports économiques et des publications sociales. Le routeur choisit d'abord l'index pertinent ; à l'intérieur de cet index, LanceDB retrouve la proximité sémantique et BM25 privilégie les mots exacts, sigles et noms propres. Pour les tableaux, un troisième canal lexical regroupe les cellules par ligne statistique afin de sélectionner ensuite la période exacte.
            """),
            md("""
            ## Schéma d'index et contrôles de reconstruction

            LanceDB stocke le vecteur avec toutes les facettes structurées. BM25 indexe aussi l'indicateur, la période, la valeur, l'unité, le territoire et le type de mesure. Les faits calculés sont indexés comme des preuves traçables.
            """),
            code("""
            stores = {
                'institutional': production_store,
                'economic': economic_store,
                'facebook': facebook_store,
            }
            schema_rows = []
            for route, store in stores.items():
                metadata = store.dense.metadata_rows()
                content_counts = pd.Series([
                    row.get('content_type', 'text') for row in metadata
                ]).value_counts()
                schema_rows.append({
                    'route': route,
                    'schema_structuré': store.dense.schema_is_current(),
                    'dense': store.dense.count(),
                    'bm25': len(store.sparse.chunk_ids),
                    'text': int(content_counts.get('text', 0)),
                    'table_fact': int(content_counts.get('table_fact', 0)),
                    'derived_fact': int(content_counts.get('derived_fact', 0)),
                })
            schema_df = pd.DataFrame(schema_rows)
            display(schema_df)
            assert schema_df['schema_structuré'].all(), 'Un index LanceDB utilise l’ancien schéma.'
            assert schema_df['dense'].eq(schema_df['bm25']).all(), 'Dense et BM25 ne sont pas alignés.'
            economic_metadata = economic_store.dense.metadata_rows()
            derived_rows = [
                row for row in economic_metadata if row.get('content_type') == 'derived_fact'
            ]
            assert derived_rows, 'Les faits calculés ne sont pas présents dans LanceDB.'
            assert {'derived_annual_sum', 'derived_quarter_change'} <= {
                row.get('source_method') for row in derived_rows
            }
            """),
        ],
    )

    files[WORKFLOW / "08_retrieval_hybride_et_reranking.ipynb"] = nb(
        "08 — Retrieval hybride, MMR et reranking",
        "Suit une question à travers dense, BM25, fusion RRF, diversification MMR et cross-encoder BGE Reranker.",
        [
            code(bootstrap()),
            md("""
            ## Chaîne de recherche

            1. BGE-M3 encode la question.
            2. LanceDB, BM25 et le canal des lignes tabulaires proposent leurs candidats.
            3. Les meilleurs documents servent de semences à une recherche locale de la meilleure page ; une période, un indicateur ou un type de mesure incompatible est pénalisé.
            4. RRF fusionne les rangs.
            5. MMR sélectionne jusqu'à 15 passages diversifiés.
            6. BGE Reranker réordonne les candidats et retourne normalement 5 passages.
            7. Une demande explicitement exhaustive ou formulée comme une collection (« quels sont les X ») peut en conserver jusqu'à 10.

            En mode exhaustif seulement, les bassins dense/BM25 et le lot soumis au reranker sont élargis. Un rappel de couverture identifie aussi automatiquement la catégorie demandée dans des formulations générales comme « quels sont les X » ou « liste des X » : les passages contenant ce terme rejoignent le reranker, sans être forcés dans la réponse. MMR et la déduplication tolèrent davantage les documents proches, car deux passages très similaires peuvent décrire deux éléments distincts d'une même liste. Le budget du prompt est ensuite partagé entre tous les résultats retenus.
            """),
            code("""
            import pandas as pd
            from config import load_config
            from conversation import contextualize_question, is_exhaustive_request

            config = load_config(PROJECT_ROOT / 'config.yaml')
            history = [
                {'role': 'user', 'content': "Quels sont les finalistes du concours d'éloquence du CDR ?"},
                {'role': 'assistant', 'content': 'Plusieurs finalistes ont été retrouvés.'},
            ]
            followup = 'La liste est-elle exhaustive ? Je veux la liste complète.'
            retrieval_query, contextualized = contextualize_question(followup, history)
            print(retrieval_query)
            print('Question contextualisée :', contextualized)
            print('Recherche élargie :', is_exhaustive_request(followup))
            """),
            code("""
            from source_router import route_question

            route_examples = [
                "Quelles sont les filières de l'ENSEA ?",
                "Quel est le PIB de la Côte d'Ivoire selon l'ANSTAT ?",
                "Quels sont les finalistes du concours du CDR ?",
            ]
            display(pd.DataFrame([
                {
                    'question': question,
                    'route': route_question(question).route,
                    'raison': route_question(question).reason,
                }
                for question in route_examples
            ]))
            """),
            code("""
            RUN_RETRIEVAL = True
            if RUN_RETRIEVAL:
                from rag_pipeline import RAGPipeline
                pipeline = RAGPipeline(config)
                max_results = config.retrieval.exhaustive_max_results if is_exhaustive_request(followup) else None
                results = pipeline.retriever.retrieve(retrieval_query, max_results=max_results)
                display(pd.DataFrame([{
                    'rang': index,
                    'titre': chunk.title,
                    'section': chunk.section,
                    'route': pipeline.retriever.last_decision.route,
                    'dense': chunk.dense_score,
                    'bm25': chunk.bm25_score,
                    'rerank': chunk.rerank_score,
                    'url': chunk.url,
                } for index, chunk in enumerate(results, 1)]))
                display(pd.Series(pipeline.retriever.last_timings).to_frame('valeur'))
            else:
                print('Chargement GPU désactivé. Passez RUN_RETRIEVAL à True pour exécuter la recherche.')
            """),
            code("""
            if RUN_RETRIEVAL:
                debug = pd.DataFrame(pipeline.retriever.last_debug_rows)
                columns = [name for name in ['title', 'url', 'content_type', 'page', 'indicator', 'period', 'measure_type', 'dense_rank', 'bm25_rank', 'table_fact_rank', 'page_rescue_rank', 'page_rescue_bonus', 'rrf_rank', 'mmr_rank', 'rerank_rank', 'rerank_score', 'selection_score', 'anchor_candidate', 'anchor_title_candidate', 'coverage_reserved', 'final_rank'] if name in debug]
                display(debug[columns].sort_values(['final_rank', 'rerank_rank'], na_position='last').head(30))
            """),
            md("""
            ## Validation structurée générale

            Les facettes ne contiennent aucune réponse préprogrammée. Elles récompensent la couverture exacte des périodes et la compatibilité de l'indicateur, de l'unité, du territoire et du type de mesure. Une question nécessitant plusieurs valeurs recherche toutes ses composantes ; les sommes, variations ou parts sûres sont calculées par le code et gardent leurs opérandes citables.
            """),
            code("""
            if RUN_RETRIEVAL:
                checks = [
                    ("Combien d’hommes comptait la Côte d’Ivoire en 2025 ?", '17 103 895', 'table_fact'),
                    ('Quel était le rapport de masculinité en 2025 ?', '109,3', 'table_fact'),
                    ('Quel était le PIB en volume CVS au quatrième trimestre 2025 ?', '12 083,3', 'table_fact'),
                    ('De combien le PIB en volume CVS a-t-il varié entre le troisième et le quatrième trimestre 2025 ?', '-0,4', 'derived_fact'),
                    ('Quelle était la valeur nominale totale de l’agriculture vivrière sur l’ensemble de l’année 2025 ?', '3357,6', 'derived_fact'),
                    ('En rapportant le déficit budgétaire 2025 au PIB nominal annuel 2025, quelle part du PIB représentait-il ?', '2,9', 'derived_fact'),
                ]
                validation_rows = []
                for question, expected_value, expected_type in checks:
                    retrieved = pipeline.retriever.retrieve(question)
                    matches = [
                        (rank, chunk)
                        for rank, chunk in enumerate(retrieved, start=1)
                        if chunk.content_type == expected_type and chunk.value == expected_value
                    ]
                    validation_rows.append({
                        'question': question,
                        'valeur_attendue': expected_value,
                        'type_attendu': expected_type,
                        'trouvé': bool(matches),
                        'meilleur_rang': matches[0][0] if matches else None,
                        'page': matches[0][1].page if matches else None,
                        'page_rescue_candidats': pipeline.retriever.last_timings.get('page_rescue_candidates', 0),
                    })
                validation_df = pd.DataFrame(validation_rows)
                display(validation_df)
                assert validation_df['trouvé'].all(), 'Un fait direct ou calculé manque dans le top final.'
            """),
        ],
    )

    files[WORKFLOW / "09_generation_citations_et_conversation.ipynb"] = nb(
        "09 — Génération, citations et conversation",
        "Explique le budget du prompt, la réponse fidèle aux sources, le cache et le traitement des questions de suivi.",
        [
            code(bootstrap()),
            md("""
            ## Règles de génération

            Le générateur répond uniquement avec les passages récupérés. Chaque affirmation factuelle reçoit une référence `[n]`. L'historique sert seulement à comprendre une relance ; les anciennes réponses ne deviennent jamais des preuves et leurs anciens numéros de citation sont supprimés.
            """),
            code("""
            import pandas as pd
            from IPython.display import Markdown, display
            from config import load_config

            config = load_config(PROJECT_ROOT / 'config.yaml')
            RUN_GENERATION = False
            if RUN_GENERATION:
                from rag_pipeline import RAGPipeline
                pipeline = RAGPipeline(config)
                first_question = "Quels sont les finalistes du concours d'éloquence du CDR de cette année ?"
                first = pipeline.ask(first_question)
                display(Markdown(first.answer))
                history = [
                    {'role': 'user', 'content': first_question},
                    {'role': 'assistant', 'content': first.answer},
                ]
                followup = pipeline.ask(
                    'La liste est-elle exhaustive ? Je veux la liste exhaustive.',
                    conversation_history=history,
                )
                display(Markdown(followup.answer))
            else:
                print('Génération désactivée. Libérez la VRAM puis passez RUN_GENERATION à True.')
            """),
            code("""
            if RUN_GENERATION:
                display(pd.Series({
                    **pipeline.generator.last_prompt_stats,
                    **pipeline.generator.last_generation_stats,
                    'route': pipeline.retriever.last_decision.route,
                    'contextualized': pipeline.last_contextualized,
                    'expanded_retrieval': pipeline.last_expanded_retrieval,
                    'citations': len(followup.citations),
                }).to_frame('valeur'))
                display(pd.DataFrame([citation.__dict__ for citation in followup.citations]))
                display(pd.DataFrame([
                    {
                        'rang': index,
                        'titre': chunk.title,
                        'dans_le_prompt': index in pipeline.generator.prompt_builder.last_passages,
                        'couverture_obligatoire': chunk.coverage_reserved,
                        'rerank': chunk.rerank_score,
                        'url': chunk.url,
                    }
                    for index, chunk in enumerate(pipeline.last_chunks, start=1)
                ]))

                retrieval_debug = pd.DataFrame(pipeline.retriever.last_debug_rows)
                debug_columns = [name for name in [
                    'title', 'url', 'bm25_rank', 'mmr_rank', 'rerank_rank',
                    'rerank_score', 'selection_score', 'anchor_candidate',
                    'anchor_title_candidate', 'coverage_reserved', 'final_rank',
                ] if name in retrieval_debug]
                display(
                    retrieval_debug[debug_columns]
                    .sort_values(['final_rank', 'selection_score'], ascending=[True, False], na_position='last')
                    .head(50)
                )
            """),
            md("""
            ## Cache exact

            La clé du cache contient la question normalisée, les chunks, l'historique compact, le prompt et les paramètres du modèle. Une nouvelle politique de réponse ou une conversation différente ne peut donc pas recevoir une ancienne réponse incompatible.
            """),
        ],
    )

    files[WORKFLOW / "10_evaluation_retrieval_et_generation.ipynb"] = nb(
        "10 — Évaluation du retrieval et de la génération",
        "Regroupe l'audit du golden set, Recall/Precision/Hit Rate/MRR, latences et revue humaine des réponses.",
        [
            code(bootstrap()),
            code("""
            import re
            import pandas as pd
            from config import load_config
            from evaluation import load_golden_set, validate_golden_set

            config = load_config(PROJECT_ROOT / 'config.yaml')
            examples = load_golden_set(config.evaluation.expanded_golden_set_full_path)
            chunks_path = config.project.data_path / 'chunks' / 'chunks.jsonl'
            corpus_urls = []
            for line in chunks_path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    corpus_urls.append(__import__('json').loads(line).get('url', ''))
            report = validate_golden_set(examples, corpus_urls=corpus_urls)
            display(pd.Series(report).to_frame('valeur'))
            display(pd.DataFrame([example.__dict__ for example in examples]).groupby(['split', 'category']).size().rename('questions').to_frame())
            """),
            code("""
            import json

            local_records = []
            for path in sorted(config.local_documents.raw_full_path.glob('local_pdf_*.json')):
                local_records.append(json.loads(path.read_text(encoding='utf-8')))
            economic_chunks_path = config.local_documents.chunks_full_path
            economic_urls = []
            if economic_chunks_path.exists():
                for line in economic_chunks_path.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        economic_urls.append(json.loads(line).get('url', ''))
            indexed_urls = set(economic_urls)
            missing_local_urls = [record['url'] for record in local_records if record['url'] not in indexed_urls]
            display(pd.Series({
                'pdf_locaux_importés': len(local_records),
                'pdf_locaux_représentés_dans_les_chunks': len(local_records) - len(missing_local_urls),
                'pdf_locaux_absents_des_chunks': len(missing_local_urls),
            }).to_frame('valeur'))
            assert not missing_local_urls, missing_local_urls[:10]
            """),
            code("""
            from source_router import route_question

            routing_checks = {
                "Quel est le PIB de la Cote d'Ivoire selon l'ANSTAT ?": 'economic',
                "Quels sont les resultats du RGPH ?": 'economic',
                "Quelles sont les conditions d'admission a l'ENSEA ?": 'institutional',
                "Quels sont les finalistes du concours du CDR cette annee ?": 'facebook',
            }
            for question, expected_route in routing_checks.items():
                decision = route_question(question)
                print(question, '->', decision.route, '|', decision.reason)
                assert decision.route == expected_route, (question, decision)
            """),
            md("""
            ## Métriques

            - **Recall@k** : part des sources pertinentes retrouvées.
            - **Precision@k** : part de résultats pertinents dans le top k.
            - **Hit Rate@k** : présence d'au moins une source pertinente.
            - **MRR** : récompense une première bonne source placée très haut.
            - **P50/P95** : latence typique et latence haute.
            - **Fidélité, correction, complétude** : revue humaine de la génération.
            """),
            code("""
            RUN_RETRIEVAL_EVALUATION = False
            if RUN_RETRIEVAL_EVALUATION:
                from rag_pipeline import RAGPipeline
                from evaluation import RetrievalEvaluator
                pipeline = RAGPipeline(config)
                development = [example for example in examples if example.split == 'development']
                evaluator = RetrievalEvaluator(config.evaluation)
                metrics = evaluator.evaluate(pipeline, development, run_id='workflow-development')
                display(pd.Series(metrics).to_frame('valeur'))
                details = pd.DataFrame(evaluator.last_details)
                display(details[details['first_relevant_rank'].isna()][['question', 'top_title', 'top_url', 'relevant_urls']])
            else:
                print('Benchmark GPU désactivé. Le notebook benchmark_retrieval conserve les runs historiques.')
            """),
            code("""
            RUN_GENERATION_REVIEW = False
            if RUN_GENERATION_REVIEW:
                if 'pipeline' not in globals():
                    from rag_pipeline import RAGPipeline
                    pipeline = RAGPipeline(config)
                rows = []
                for example in [item for item in examples if item.should_answer][:8]:
                    response = pipeline.ask(example.question)
                    rows.append({
                        'question': example.question,
                        'answer': response.answer,
                        'answer_words': len(response.answer.split()),
                        'answer_mode': pipeline.generator.last_prompt_stats.get('answer_mode'),
                        'citations': len(response.citations),
                        'mechanical_style': bool(re.search(r'(?im)^\\s*(synthèse|réponse|citation)\\s*:', response.answer)),
                        'human_correct': None,
                        'human_faithful': None,
                        'human_complete': None,
                    })
                display(pd.DataFrame(rows))
            """),
        ],
    )

    files[WORKFLOW / "11_observabilite.ipynb"] = nb(
        "11 — Observabilité et diagnostic des performances",
        "Analyse les traces produites par le pipeline : latence, cache, erreurs, routes, conversation et recherche exhaustive.",
        [
            code(bootstrap()),
            code("""
            import pandas as pd
            from config import load_config
            from observability import load_traces, summarize_traces

            config = load_config(PROJECT_ROOT / 'config.yaml')
            events = load_traces(config.observability.traces_full_path)
            display(pd.Series(summarize_traces(events)).to_frame('valeur'))
            traces = pd.json_normalize(events, sep='.') if events else pd.DataFrame()
            print('Événements :', len(traces))
            """),
            code("""
            if not traces.empty:
                wanted = [column for column in [
                    'timestamp_utc', 'question', 'status', 'metadata.route',
                    'metadata.contextualized', 'metadata.expanded_retrieval',
                    'retrieval_stats.total_seconds', 'generation_stats.total_seconds',
                    'generation_stats.tokens_per_second', 'generation_stats.cache_hit',
                    'citation_count', 'total_seconds', 'error',
                ] if column in traces]
                display(traces[wanted].tail(20))
            """),
            code("""
            if not traces.empty and 'total_seconds' in traces:
                display(traces.nlargest(10, 'total_seconds')[wanted])
                if 'metadata.route' in traces:
                    display(traces.groupby('metadata.route')['total_seconds'].agg(['count', 'mean', 'median', 'max']))
            """),
            md("""
            ## Lecture recommandée

            Utilisez P50 pour l'expérience habituelle et P95 pour les lenteurs rares. Séparez retrieval et génération avant d'optimiser. Une vitesse de génération anormalement basse peut signaler une contention VRAM ou un déport partiel en RAM.
            """),
        ],
    )

    files[WORKFLOW / "12_ajout_pdf_et_index_incremental.ipynb"] = nb(
        "12 — Ajout de PDF et mise à jour incrémentale",
        "Décrit le chemin d'un PDF ajouté dans l'application et vérifie les documents déjà enregistrés sans reconstruire tout l'index.",
        [
            code(bootstrap()),
            md("""
            ## Chemin d'un document ajouté

            `Upload PDF → validation et empreinte → extraction du texte → nettoyage → chunks parent-enfant → embeddings des nouveaux chunks → mise à jour LanceDB → reconstruction BM25 → document interrogeable`

            L'index dense ne recalcule pas les milliers de chunks inchangés. BM25, beaucoup plus léger, est reconstruit pour rester cohérent.
            """),
            code("""
            import sqlite3
            import pandas as pd
            from config import load_config

            config = load_config(PROJECT_ROOT / 'config.yaml')
            database_path = config.project.data_path / 'app' / 'rag_ensea.sqlite3'
            uploads_path = config.project.data_path / 'uploads' / 'pdfs'
            print('Base application :', database_path)
            print('Dossier PDF :', uploads_path)
            if database_path.exists():
                with sqlite3.connect(database_path) as connection:
                    documents = pd.read_sql_query('SELECT id, filename, title, status, page_count, chunk_count, created_at FROM documents ORDER BY created_at DESC', connection)
                display(documents)
            """),
            code("""
            import requests

            RUN_UPLOAD_TEST = False
            PDF_TO_UPLOAD = PROJECT_ROOT / 'chemin' / 'vers' / 'document.pdf'
            if RUN_UPLOAD_TEST:
                if not PDF_TO_UPLOAD.exists():
                    raise FileNotFoundError(PDF_TO_UPLOAD)
                with PDF_TO_UPLOAD.open('rb') as handle:
                    response = requests.post(
                        'http://127.0.0.1:8000/api/documents',
                        files={'file': (PDF_TO_UPLOAD.name, handle, 'application/pdf')},
                        data={'title': PDF_TO_UPLOAD.stem, 'category': 'rapport'},
                        timeout=120,
                    )
                response.raise_for_status()
                display(response.json())
            else:
                print('Upload de test désactivé. Utilisez normalement la bibliothèque de l’application.')
            """),
            md("""
            ## Contrôles

            Vérifiez le nombre de pages, le nombre de chunks, le statut `ready`, puis posez une question dont la réponse figure uniquement dans le nouveau PDF. Une empreinte empêche l'ajout accidentel du même fichier deux fois.
            """),
        ],
    )

    files[WORKFLOW / "13_rag_ensea_end_to_end.ipynb"] = nb(
        "13 — RAG ENSEA complet de bout en bout",
        "Charge les index validés et expose une fonction unique pour les questions simples, Facebook, exhaustives et conversationnelles.",
        [
            code(bootstrap()),
            code("""
            import time
            import pandas as pd
            from IPython.display import Markdown, display
            from config import load_config

            config = load_config(PROJECT_ROOT / 'config.yaml')
            print('Embedding :', config.embeddings.model_name)
            print('Reranker :', config.reranker.model_name)
            print('Générateur :', config.generation.model_name)
            print('Résultats normaux/exhaustifs :', config.retrieval.rerank_top_k, '/', config.retrieval.exhaustive_max_results)
            """),
            code("""
            RUN_PIPELINE = False
            if RUN_PIPELINE:
                from rag_pipeline import RAGPipeline
                started = time.perf_counter()
                pipeline = RAGPipeline(config)
                print('Pipeline chargé en', round(time.perf_counter() - started, 2), 's')
                print('Index institutionnel :', pipeline.store.dense.count())
                print('Index economique :', pipeline.economic_store.dense.count() if pipeline.economic_store else 0)
                print('Index Facebook :', pipeline.facebook_store.dense.count() if pipeline.facebook_store else 0)
            else:
                print('Chargement désactivé. Fermez les autres kernels puis passez RUN_PIPELINE à True.')
            """),
            code("""
            conversation_history = []

            def ask_ensea(question, *, remember=True):
                started = time.perf_counter()
                response = pipeline.ask(question, conversation_history=conversation_history)
                display(Markdown(response.answer))
                display(pd.Series({
                    'route': pipeline.retriever.last_decision.route,
                    'contextualized': pipeline.last_contextualized,
                    'expanded_retrieval': pipeline.last_expanded_retrieval,
                    'retrieval_seconds': pipeline.retriever.last_timings.get('total_seconds', 0),
                    'generation_seconds': pipeline.generator.last_generation_stats.get('total_seconds', 0),
                    'total_seconds': time.perf_counter() - started,
                    'cache_hit': pipeline.generator.last_generation_stats.get('cache_hit', False),
                    'citations': len(response.citations),
                }).to_frame('valeur'))
                if remember:
                    conversation_history.extend([
                        {'role': 'user', 'content': question},
                        {'role': 'assistant', 'content': response.answer},
                    ])
                    del conversation_history[:-10]
                return response
            """),
            code("""
            RUN_EXAMPLES = False
            if RUN_EXAMPLES:
                ask_ensea("Quels sont les finalistes du concours d'éloquence du CDR de cette année ?")
                ask_ensea('La liste est-elle exhaustive ? Je veux la liste complète.')
            else:
                print('Exemples désactivés. Activez RUN_PIPELINE puis RUN_EXAMPLES.')
            """),
            md("""
            ## Notebook final officiel

            Ce notebook termine le parcours pédagogique. Le fichier `rag_ensea.ipynb` à la racine reste le notebook final d'usage courant, tandis que celui-ci montre explicitement comment toutes les étapes précédentes s'assemblent.
            """),
        ],
    )

    return files


def main() -> None:
    WORKFLOW.mkdir(parents=True, exist_ok=True)
    files = build()
    expected_paths = {path.resolve() for path in files}
    for old_notebook in WORKFLOW.glob("*.ipynb"):
        if old_notebook.resolve() not in expected_paths:
            old_notebook.unlink()
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
