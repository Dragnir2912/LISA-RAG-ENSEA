"""Genere le notebook d'evaluation autonome de la Banque de 100 questions."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "test&comparisons" / "evaluation_banque_100_questions.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown("""# Evaluation supplementaire - Banque de 100 questions

Ce notebook reconstruit le golden set depuis `data/questions/Banque_100_questions_CORRIGEE.docx`, puis evalue les **100 questions** avec le pipeline RAG complet.

Il mesure notamment :

- la route choisie (economique ou Facebook) ;
- la presence du bon document et, pour les PDF, de la bonne page ;
- la couverture de la reponse de reference et la qualite de l'abstention ;
- les temps de retrieval, de generation et le temps total de chaque reponse ;
- les sources retrouvees et les sources effectivement citees.

Les resultats sont sauvegardes progressivement. A la fin, le notebook exporte un CSV, un JSONL, un resume JSON et un **rapport PDF complet** associant chaque question, sa reponse, ses sources et ses temps.
"""),
    code("""from pathlib import Path
import hashlib
import json
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd
from IPython.display import FileLink, display
from tqdm.auto import tqdm

HERE = Path.cwd().resolve()
ROOT = next(
    (candidate for candidate in (HERE, *HERE.parents) if (candidate / "config.yaml").exists()),
    None,
)
if ROOT is None:
    raise FileNotFoundError("Impossible de localiser config.yaml depuis le dossier courant.")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config
from evaluation import canonical_url, load_golden_set, validate_golden_set
from evaluation_metrics import (
    answerable_automated_pass,
    is_abstention,
    reference_fact_metrics,
    route_compatible,
)
from evaluation_report import export_evaluation_pdf
from golden_set_word import build_banque_100_golden_set
from rag_pipeline import RAGPipeline

CONFIG_PATH = ROOT / "config.yaml"
cfg = load_config(CONFIG_PATH)
print(f"Projet : {ROOT}")
print(f"Modele de generation : {cfg.generation.model_name}")
"""),
    markdown("""## 1. Reconstruction et validation du golden set

Cette etape relit le Word corrige. Pour Facebook, elle ne se fie pas aveuglement au numero historique de publication : elle rattache la question au contenu reellement present dans l'index de la page concernee.
"""),
    code("""word_path = ROOT / "data/questions/Banque_100_questions_CORRIGEE.docx"
golden_path = cfg.evaluation.banque_100_golden_set_full_path

build_report = build_banque_100_golden_set(
    word_path,
    golden_path,
    facebook_raw_dir=ROOT / "data/facebook/raw",
    facebook_chunks_path=ROOT / "data/facebook/chunks.jsonl",
)
display(pd.Series(asdict(build_report), name="valeur").to_frame())
"""),
    code("""examples = load_golden_set(golden_path)
corpus_urls = []
for chunks_path in (
    ROOT / "data/chunks/chunks.jsonl",
    ROOT / "data/economic/chunks.jsonl",
    ROOT / "data/facebook/chunks.jsonl",
):
    if not chunks_path.exists():
        continue
    with chunks_path.open(encoding="utf-8") as handle:
        corpus_urls.extend(
            json.loads(line).get("url", "")
            for line in handle
            if line.strip()
        )

validation = validate_golden_set(examples, corpus_urls)
display(pd.Series(validation, name="valeur").to_frame())

catalog = pd.DataFrame([{
    "id": item.example_id,
    "numero": item.question_number,
    "domaine": item.domain_label,
    "statut": item.source_status,
    "route_attendue": item.expected_route,
    "type_source": item.source_kind,
    "page": item.reference_page,
    "question": item.question,
} for item in examples])
display(pd.crosstab(catalog["domaine"], catalog["statut"], margins=True))

assert validation["is_valid"], validation
assert len(examples) == 100
assert sum(item.should_answer for item in examples) == 75
assert sum(not item.should_answer for item in examples) == 25
print("Golden set valide : 100 questions, 75 IN-BASE et 25 OUT-OF-BASE.")
"""),
    markdown("""## 2. Parametres du run

- Laissez `MAX_QUESTIONS = None` pour le run complet de 100 questions.
- Pour un test rapide, mettez par exemple `MAX_QUESTIONS = 3`.
- Le mode reflexion est desactive par defaut afin de mesurer le mode normal de l'application.
- En cas d'interruption, copiez le chemin affiche dans `RESUME_RUN_DIR` et relancez : les questions deja terminees seront ignorees.
"""),
    code("""MAX_QUESTIONS = None       # None = les 100 questions
THINKING = False           # evaluation du mode normal de conversation
RESUME_RUN_DIR = None      # ex. ROOT / "data/evaluation_runs/20260812T120000Z_banque_100"
RETRY_ERRORS = True        # retente uniquement les lignes precedemment en erreur

selected_examples = examples if MAX_QUESTIONS is None else examples[:MAX_QUESTIONS]
print(f"Questions selectionnees : {len(selected_examples)}")
"""),
    markdown("""## 3. Initialisation du pipeline

Le pipeline est charge une seule fois. Un echauffement est execute hors mesure afin de ne pas penaliser artificiellement la premiere question.
"""),
    code("""config_hash = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
if RESUME_RUN_DIR is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = cfg.evaluation.runs_full_path / f"{stamp}_banque_100"
else:
    run_dir = Path(RESUME_RUN_DIR)
run_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = run_dir / "checkpoint_results.jsonl"

pipeline = RAGPipeline.from_config(CONFIG_PATH)
_ = pipeline.retriever.retrieve("Quelle etait la population de la Cote d'Ivoire en 2025 ?")

print(f"Dossier du run : {run_dir}")
print(f"Checkpoint : {checkpoint_path}")
print(f"Configuration : {config_hash[:16]}")
"""),
    markdown("""## 4. Fonctions de mesure

Les scores de contenu ci-dessous sont deterministes et transparents. Ils servent a reperer rapidement les echecs ; ils ne remplacent pas une revue humaine de la formulation.
"""),
    code("""def canonical_set(urls):
    return {canonical_url(url) for url in urls if url}

def url_hit(found_urls, expected_urls, k=5):
    if not expected_urls:
        return None
    return url_recall(found_urls, expected_urls, k) == 1.0

def url_recall(found_urls, expected_urls, k=5):
    expected = canonical_set(expected_urls)
    if not expected:
        return None
    return len(canonical_set(found_urls[:k]) & expected) / len(expected)

def page_hit(chunks, expected_urls, page, k=5):
    if page is None:
        return None
    expected = canonical_set(expected_urls)
    page_pattern = re.compile(rf"\\bpage\\s+0*{int(page)}\\b", re.IGNORECASE)
    for chunk in chunks[:k]:
        if canonical_url(chunk.url) not in expected:
            continue
        searchable = " ".join((chunk.section, chunk.text, chunk.parent_text))
        if page_pattern.search(searchable):
            return True
    return False

def citation_dict(citation):
    return {
        "index": citation.index,
        "title": citation.title,
        "section": citation.section,
        "url": citation.url,
        "rerank_score": float(citation.rerank_score),
        "passage": citation.passage,
    }

def retrieved_dict(chunk, rank):
    return {
        "rank": rank,
        "title": chunk.title,
        "section": chunk.section,
        "url": chunk.url,
        "rerank_score": float(chunk.rerank_score),
    }

def load_checkpoint(path):
    latest = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[row["id"]] = row
    return latest
"""),
    markdown("""## 5. Evaluation des 100 questions

Une ligne est ecrite dans le checkpoint apres chaque reponse. Le temps total est mesure autour de l'appel complet `pipeline.ask`, donc il inclut le routage, la recherche, la generation et le leger surcout d'orchestration.
"""),
    code("""existing = load_checkpoint(checkpoint_path)

for item in tqdm(selected_examples, desc="Evaluation Banque 100", unit="question"):
    previous = existing.get(item.example_id)
    if previous and (not previous.get("error") or not RETRY_ERRORS):
        continue

    started = time.perf_counter()
    try:
        response = pipeline.ask(item.question, thinking=THINKING)
        total_seconds = time.perf_counter() - started
        chunks = list(pipeline.last_chunks)
        citations = [citation_dict(citation) for citation in response.citations]
        retrieved_sources = [
            retrieved_dict(chunk, rank)
            for rank, chunk in enumerate(chunks, start=1)
        ]
        retrieved_urls = [chunk.url for chunk in chunks]
        cited_urls = [citation["url"] for citation in citations]
        retrieval_stats = dict(pipeline.retriever.last_timings)
        generation_stats = dict(pipeline.generator.last_generation_stats)
        retrieval_seconds = float(retrieval_stats.get("total_seconds", 0.0) or 0.0)
        generation_seconds = float(generation_stats.get("total_seconds", 0.0) or 0.0)
        route = pipeline.retriever.last_decision.route
        fact_metrics = (
            reference_fact_metrics(item.reference_answer, response.answer)
            if item.should_answer else None
        )
        coverage = fact_metrics.score if fact_metrics is not None else None
        abstained = is_abstention(response.answer)
        citation_source_hit = url_hit(cited_urls, item.relevant_urls, len(cited_urls))
        citation_source_recall = url_recall(cited_urls, item.relevant_urls, len(cited_urls))
        document_hit_3 = url_hit(retrieved_urls, item.relevant_urls, 3)
        document_hit_5 = url_hit(retrieved_urls, item.relevant_urls, 5)
        document_hit_10 = url_hit(retrieved_urls, item.relevant_urls, 10)
        document_source_recall_5 = url_recall(retrieved_urls, item.relevant_urls, 5)
        expected_page_hit = page_hit(chunks, item.relevant_urls, item.reference_page, 5)
        automated_pass = (
            answerable_automated_pass(
                item.reference_answer,
                response.answer,
                citation_source_hit=bool(citation_source_hit),
            )
            if item.should_answer else abstained
        )
        row = {
            "id": item.example_id,
            "question_number": item.question_number,
            "question": item.question,
            "domain_label": item.domain_label,
            "category": item.category,
            "should_answer": item.should_answer,
            "source_status": item.source_status,
            "source_kind": item.source_kind,
            "source_locator": item.source_locator,
            "reference_page": item.reference_page,
            "expected_route": item.expected_route,
            "expected_relevant_urls": item.relevant_urls,
            "reference_answer": item.reference_answer,
            "generated_answer": response.answer,
            "route": route,
            "route_confidence": float(pipeline.retriever.last_decision.confidence),
            "route_reason": pipeline.retriever.last_decision.reason,
            "route_exact": route == item.expected_route,
            "route_compatible": route_compatible(route, item.expected_route),
            "route_correct": route_compatible(route, item.expected_route),
            "document_hit_at_3": document_hit_3,
            "document_hit_at_5": document_hit_5,
            "document_hit_at_10": document_hit_10,
            "page_hit_at_5": expected_page_hit,
            "citation_source_hit": citation_source_hit,
            "citation_source_recall": citation_source_recall,
            "document_source_recall_at_5": document_source_recall_5,
            "reference_fact_coverage": coverage,
            "reference_numeric_recall": (
                fact_metrics.numeric_recall if fact_metrics is not None else None
            ),
            "reference_unit_recall": (
                fact_metrics.unit_recall if fact_metrics is not None else None
            ),
            "reference_lexical_recall": (
                fact_metrics.lexical_recall if fact_metrics is not None else None
            ),
            "abstained": abstained,
            "automated_pass": automated_pass,
            "citations": citations,
            "retrieved_sources": retrieved_sources,
            "citation_count": len(citations),
            "retrieved_source_count": len(retrieved_sources),
            "answer_words": len(response.answer.split()),
            "total_seconds": total_seconds,
            "retrieval_seconds": retrieval_seconds,
            "generation_seconds": generation_seconds,
            "orchestration_seconds": max(0.0, total_seconds - retrieval_seconds - generation_seconds),
            "cache_hit": bool(generation_stats.get("cache_hit", False)),
            "prompt_tokens": int(generation_stats.get("ollama_prompt_tokens", 0) or 0),
            "generated_tokens": int(generation_stats.get("generated_tokens", 0) or 0),
            "tokens_per_second": float(generation_stats.get("tokens_per_second", 0.0) or 0.0),
            "error": "",
            "human_correct": None,
            "human_faithful": None,
            "human_complete": None,
            "review_notes": "",
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as error:
        row = {
            "id": item.example_id,
            "question_number": item.question_number,
            "question": item.question,
            "domain_label": item.domain_label,
            "category": item.category,
            "should_answer": item.should_answer,
            "source_status": item.source_status,
            "source_kind": item.source_kind,
            "source_locator": item.source_locator,
            "reference_page": item.reference_page,
            "expected_route": item.expected_route,
            "expected_relevant_urls": item.relevant_urls,
            "reference_answer": item.reference_answer,
            "generated_answer": "",
            "total_seconds": time.perf_counter() - started,
            "retrieval_seconds": 0.0,
            "generation_seconds": 0.0,
            "citations": [],
            "retrieved_sources": [],
            "automated_pass": False,
            "error": f"{type(error).__name__}: {error}",
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    with checkpoint_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\\n")
    existing[item.example_id] = row

print(f"Checkpoint termine : {checkpoint_path}")
"""),
    markdown("""## 6. Resultats, metriques et exports tabulaires

Les temps sont presentes en secondes. Le p95 met en evidence les questions les plus lentes sans etre domine par un seul maximum exceptionnel.
"""),
    code("""latest = load_checkpoint(checkpoint_path)
ordered_rows = [latest[item.example_id] for item in selected_examples if item.example_id in latest]
results_df = pd.DataFrame(ordered_rows).sort_values("question_number").reset_index(drop=True)

successful = results_df[results_df["error"].fillna("").eq("")].copy()
answerable = successful[successful["should_answer"] == True]
unanswerable = successful[successful["should_answer"] == False]
pdf_answerable = answerable[answerable["source_kind"] == "pdf"]

def mean_bool(series):
    clean = series.dropna()
    return float(clean.astype(bool).mean()) if len(clean) else 0.0

def mean_number(series):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else 0.0

latencies = pd.to_numeric(successful["total_seconds"], errors="coerce").dropna()
summary = {
    "questions_total": len(selected_examples),
    "questions_completed": len(successful),
    "errors": int(results_df["error"].fillna("").ne("").sum()),
    "answerable_questions": len(answerable),
    "unanswerable_questions": len(unanswerable),
    "automated_pass_rate": mean_bool(successful["automated_pass"]),
    "retrieval_hit_rate_at_5": mean_bool(answerable["document_hit_at_5"]),
    "page_hit_rate_at_5": mean_bool(pdf_answerable["page_hit_at_5"]),
    "citation_source_hit_rate": mean_bool(answerable["citation_source_hit"]),
    "citation_source_recall_mean": mean_number(answerable["citation_source_recall"]),
    "retrieval_source_recall_at_5_mean": mean_number(answerable["document_source_recall_at_5"]),
    "reference_fact_coverage_mean": mean_number(answerable["reference_fact_coverage"]),
    "abstention_accuracy": mean_bool(unanswerable["abstained"]),
    "route_compatibility_rate": mean_bool(successful["route_compatible"]),
    "route_exact_rate": mean_bool(successful["route_exact"]),
    "mean_total_seconds": float(latencies.mean()) if len(latencies) else 0.0,
    "p50_total_seconds": float(latencies.quantile(0.50)) if len(latencies) else 0.0,
    "p95_total_seconds": float(latencies.quantile(0.95)) if len(latencies) else 0.0,
    "max_total_seconds": float(latencies.max()) if len(latencies) else 0.0,
    "mean_retrieval_seconds": mean_number(successful["retrieval_seconds"]),
    "mean_generation_seconds": mean_number(successful["generation_seconds"]),
}
display(pd.Series(summary, name="valeur").to_frame())

display_columns = [
    "question_number", "domain_label", "route", "route_exact", "route_compatible",
    "document_hit_at_5", "page_hit_at_5", "citation_source_hit",
    "reference_fact_coverage", "abstained", "automated_pass",
    "retrieval_seconds", "generation_seconds", "total_seconds", "error",
]
display(results_df[display_columns])

jsonl_path = run_dir / "results_complete.jsonl"
jsonl_path.write_text(
    "\\n".join(json.dumps(row, ensure_ascii=False) for row in ordered_rows) + "\\n",
    encoding="utf-8",
)

csv_df = results_df.copy()
for column in ("expected_relevant_urls", "citations", "retrieved_sources"):
    if column in csv_df:
        csv_df[column] = csv_df[column].apply(lambda value: json.dumps(value, ensure_ascii=False))
csv_path = run_dir / "results_complete.csv"
csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

summary_path = run_dir / "summary.json"
summary_path.write_text(json.dumps({
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "config_sha256": config_hash,
    "model": cfg.generation.model_name,
    "thinking": THINKING,
    "golden_set": str(golden_path),
    "summary": summary,
}, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"JSONL : {jsonl_path}")
print(f"CSV   : {csv_path}")
print(f"Resume: {summary_path}")
"""),
    code("""failed_cases = results_df[
    results_df["error"].fillna("").ne("") | (results_df["automated_pass"] != True)
]
print(f"Cas a verifier : {len(failed_cases)}/{len(results_df)}")
display(failed_cases[[
    "question_number", "question", "reference_answer", "generated_answer",
    "route", "source_locator", "total_seconds", "error",
]])
"""),
    markdown("""## 7. Rapport PDF final

Le PDF contient les 100 questions dans l'ordre, la reponse de reference, la reponse produite, la route, les temps total/retrieval/generation, la source attendue et les sources trouvees.
"""),
    code("""pdf_path = run_dir / "rapport_banque_100_questions.pdf"
export_evaluation_pdf(
    ordered_rows,
    summary,
    pdf_path,
    title="Évaluation RAG - Banque de 100 questions",
    subtitle="Questions, réponses générées, sources retrouvées et temps de réponse",
)
print(f"Rapport PDF : {pdf_path}")
display(FileLink(str(pdf_path)))
"""),
    markdown("""## Lecture conseillee

1. Commencez par le resume global et le p95 du temps total.
2. Examinez les cas `A VERIFIER` dans le PDF.
3. Pour les questions PDF, privilegiez `page_hit_at_5` au simple `document_hit_at_5`.
4. Completez les colonnes de revue humaine du CSV si vous souhaitez etablir un score final valide manuellement.
"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
