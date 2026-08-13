"""Évaluation du système RAG.

Deux familles de métriques :
1. Retrieval (calcul maison, pas de dépendance externe) : Recall@k, Precision@k,
   MRR, Hit Rate — comparent les URLs retournées par le retriever aux URLs de
   référence du golden set.
2. Génération (Ragas) : Faithfulness, Context Precision, Answer Relevancy —
   utilisent un LLM juge. Le juge est le même modèle Ollama local que la
   génération (pas d'appel API payant, cohérent avec "100% local").
"""

from __future__ import annotations

import json
import logging
import math
import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from config import EvaluationConfig
from rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class GoldenExample:
    question: str
    reference_answer: str
    relevant_urls: list[str]
    example_id: str = ""
    category: str = "non_classe"
    difficulty: str = "medium"
    split: str = "development"
    should_answer: bool = True
    answer_keywords: list[str] | None = None
    question_number: int | None = None
    domain_label: str = ""
    source_locator: str = ""
    source_status: str = ""
    source_kind: str = ""
    source_document: str = ""
    reference_page: int | None = None
    expected_route: str = ""


def load_golden_set(path: Path) -> list[GoldenExample]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            examples.append(GoldenExample(
                question=d["question"],
                reference_answer=d.get("reference_answer", ""),
                relevant_urls=d.get("relevant_urls", []),
                example_id=d.get("id", ""),
                category=d.get("category", "non_classe"),
                difficulty=d.get("difficulty", "medium"),
                split=d.get("split", "development"),
                should_answer=d.get("should_answer", True),
                answer_keywords=d.get("answer_keywords", []),
                question_number=d.get("question_number"),
                domain_label=d.get("domain_label", ""),
                source_locator=d.get("source_locator", ""),
                source_status=d.get("source_status", ""),
                source_kind=d.get("source_kind", ""),
                source_document=d.get("source_document", ""),
                reference_page=d.get("reference_page"),
                expected_route=d.get("expected_route", ""),
            ))
    return examples


def validate_golden_set(
    examples: list[GoldenExample], corpus_urls: list[str] | None = None,
) -> dict[str, Any]:
    """Contrôle les doublons, champs obligatoires et URLs absentes du corpus."""
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    duplicate_ids: list[str] = []
    duplicate_questions: list[str] = []
    missing_corpus_urls: list[dict[str, str]] = []
    structural_errors: list[str] = []
    corpus_canonical = _canonical_set(corpus_urls or [])

    for index, example in enumerate(examples, start=1):
        normalized_question = " ".join(example.question.casefold().split())
        if not example.example_id:
            structural_errors.append(f"Ligne {index}: id manquant")
        elif example.example_id in seen_ids:
            duplicate_ids.append(example.example_id)
        seen_ids.add(example.example_id)
        if normalized_question in seen_questions:
            duplicate_questions.append(example.question)
        seen_questions.add(normalized_question)
        if example.should_answer and not example.relevant_urls:
            structural_errors.append(
                f"{example.example_id}: relevant_urls vide pour une question répondable"
            )
        if not example.should_answer and example.relevant_urls:
            structural_errors.append(
                f"{example.example_id}: une question hors corpus ne doit pas avoir d'URL attendue"
            )
        if corpus_canonical:
            for url in example.relevant_urls:
                if canonical_url(url) not in corpus_canonical:
                    missing_corpus_urls.append({"id": example.example_id, "url": url})

    return {
        "examples": len(examples),
        "answerable": sum(example.should_answer for example in examples),
        "unanswerable": sum(not example.should_answer for example in examples),
        "duplicate_ids": duplicate_ids,
        "duplicate_questions": duplicate_questions,
        "missing_corpus_urls": missing_corpus_urls,
        "structural_errors": structural_errors,
        "is_valid": not (
            duplicate_ids or duplicate_questions or missing_corpus_urls or structural_errors
        ),
    }


# ---------------------------------------------------------------------------
# Métriques de retrieval (implémentation maison)
# ---------------------------------------------------------------------------

_ENSEA_MIRROR_HOSTS = {"ensea.ed.ci", "ensea360.ensea.ed.ci", "www.ensea.ed.ci"}


def canonical_url(url: str) -> str:
    """Normalise les miroirs officiels, fragments, requêtes et slash finaux."""
    parsed = urlsplit(url.strip())
    host = parsed.netloc.casefold()
    if host in _ENSEA_MIRROR_HOSTS:
        host = "ensea.ed.ci"
    path = "/" + "/".join(segment for segment in parsed.path.split("/") if segment)
    if path == "/":
        return f"{host}/"
    return f"{host}{path}"


def _canonical_set(urls: list[str]) -> set[str]:
    return {canonical_url(url) for url in urls if url}

def recall_at_k(retrieved_urls: list[str], relevant_urls: list[str], k: int) -> float:
    if not relevant_urls:
        return 0.0
    top_k = _canonical_set(retrieved_urls[:k])
    relevant = _canonical_set(relevant_urls)
    hits = len(top_k & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved_urls: list[str], relevant_urls: list[str], k: int) -> float:
    top_k = retrieved_urls[:k]
    if not top_k:
        return 0.0
    canonical_top_k = _canonical_set(top_k)
    if not canonical_top_k:
        return 0.0
    hits = len(canonical_top_k & _canonical_set(relevant_urls))
    return hits / len(canonical_top_k)


def reciprocal_rank(retrieved_urls: list[str], relevant_urls: list[str]) -> float:
    relevant_set = _canonical_set(relevant_urls)
    for rank, url in enumerate(retrieved_urls, start=1):
        if canonical_url(url) in relevant_set:
            return 1.0 / rank
    return 0.0


def hit_rate(retrieved_urls: list[str], relevant_urls: list[str], k: int) -> float:
    return 1.0 if _canonical_set(retrieved_urls[:k]) & _canonical_set(relevant_urls) else 0.0


def first_relevant_rank(
    retrieved_urls: list[str], relevant_urls: list[str],
) -> int | None:
    """Rang 1-indexé de la première URL attendue, ou ``None`` si absente."""
    relevant_set = _canonical_set(relevant_urls)
    return next(
        (
            rank for rank, url in enumerate(retrieved_urls, start=1)
            if canonical_url(url) in relevant_set
        ),
        None,
    )


def percentile(values: list[float], proportion: float) -> float:
    """Percentile par rang le plus proche, sans dépendance scientifique externe."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(proportion * len(ordered)) - 1)
    return ordered[index]


class RetrievalEvaluator:
    def __init__(self, cfg: EvaluationConfig) -> None:
        self.cfg = cfg
        # Rapport détaillé du dernier run, consultable dans le notebook sans
        # casser l'API historique de ``evaluate`` qui retourne un dict plat.
        self.last_details: list[dict] = []
        self.last_group_metrics: dict[str, dict[str, dict[str, float]]] = {}

    def evaluate(
        self,
        pipeline: RAGPipeline,
        examples: list[GoldenExample],
        *,
        run_id: str | None = None,
    ) -> dict:
        per_k_metrics = {k: {"recall": [], "precision": [], "hit_rate": []} for k in self.cfg.k_values}
        mrr_scores = []
        latencies = []
        self.last_details = []

        for example in examples:
            chunks = pipeline.retriever.retrieve(example.question)
            retrieved_urls = [c.url for c in chunks]
            latency = float(pipeline.retriever.last_timings.get("total_seconds", 0.0))
            latencies.append(latency)

            question_mrr = (
                reciprocal_rank(retrieved_urls, example.relevant_urls)
                if example.should_answer else None
            )
            if question_mrr is not None:
                mrr_scores.append(question_mrr)
            rank = (
                first_relevant_rank(retrieved_urls, example.relevant_urls)
                if example.should_answer else None
            )
            detail = {
                "id": example.example_id,
                "question": example.question,
                "category": example.category,
                "difficulty": example.difficulty,
                "split": example.split,
                "should_answer": example.should_answer,
                "first_relevant_rank": rank,
                "mrr": question_mrr,
                "latency_seconds": latency,
                "returned_chunks": len(chunks),
                "top_title": chunks[0].title if chunks else "",
                "top_url": retrieved_urls[0] if retrieved_urls else "",
                "retrieved_urls": retrieved_urls,
                "relevant_urls": example.relevant_urls,
            }
            for k in self.cfg.k_values:
                if example.should_answer:
                    recall = recall_at_k(retrieved_urls, example.relevant_urls, k)
                    precision = precision_at_k(retrieved_urls, example.relevant_urls, k)
                    hit = hit_rate(retrieved_urls, example.relevant_urls, k)
                    per_k_metrics[k]["recall"].append(recall)
                    per_k_metrics[k]["precision"].append(precision)
                    per_k_metrics[k]["hit_rate"].append(hit)
                    detail[f"hit@{k}"] = bool(hit)
                else:
                    detail[f"hit@{k}"] = None
            self.last_details.append(detail)
            if hasattr(pipeline, "observability"):
                pipeline.observability.record(
                    event_type="retrieval",
                    question=example.question,
                    source="evaluation.retrieval",
                    status="ok",
                    chunks=chunks,
                    retrieval_stats=pipeline.retriever.last_timings,
                    total_seconds=latency,
                    run_id=run_id,
                    metadata={
                        "example_id": example.example_id,
                        "category": example.category,
                        "difficulty": example.difficulty,
                        "split": example.split,
                        "should_answer": example.should_answer,
                        "first_relevant_rank": rank,
                    },
                )

        results = {
            "questions": len(examples),
            "answerable_questions": sum(example.should_answer for example in examples),
            "unanswerable_questions": sum(not example.should_answer for example in examples),
            "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
            "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50_latency_seconds": percentile(latencies, 0.50),
            "p95_latency_seconds": percentile(latencies, 0.95),
            "max_latency_seconds": max(latencies, default=0.0),
        }
        for k, metrics in per_k_metrics.items():
            results[f"recall@{k}"] = sum(metrics["recall"]) / len(metrics["recall"]) if metrics["recall"] else 0.0
            results[f"precision@{k}"] = sum(metrics["precision"]) / len(metrics["precision"]) if metrics["precision"] else 0.0
            results[f"hit_rate@{k}"] = sum(metrics["hit_rate"]) / len(metrics["hit_rate"]) if metrics["hit_rate"] else 0.0
        self.last_group_metrics = {
            "category": self._group_metrics("category"),
            "difficulty": self._group_metrics("difficulty"),
            "split": self._group_metrics("split"),
        }
        return results

    def _group_metrics(self, field: str) -> dict[str, dict[str, float]]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for detail in self.last_details:
            if detail["should_answer"]:
                groups[str(detail[field])].append(detail)
        output: dict[str, dict[str, float]] = {}
        for name, rows in sorted(groups.items()):
            group = {
                "questions": len(rows),
                "mrr": sum(float(row["mrr"]) for row in rows) / len(rows),
                "mean_latency_seconds": sum(row["latency_seconds"] for row in rows) / len(rows),
            }
            for k in self.cfg.k_values:
                group[f"hit_rate@{k}"] = sum(bool(row[f"hit@{k}"]) for row in rows) / len(rows)
            output[name] = group
        return output


def save_evaluation_run(
    runs_dir: Path,
    summary: dict[str, Any],
    details: list[dict[str, Any]],
    group_metrics: dict[str, Any],
    *,
    run_name: str = "retrieval",
) -> Path:
    """Sauvegarde un benchmark versionné (résumé JSON + détails CSV)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir / f"{stamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "summary": summary,
        "groups": group_metrics,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    if details:
        with open(run_dir / "details.csv", "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(details[0]))
            writer.writeheader()
            for detail in details:
                writer.writerow({
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                    for key, value in detail.items()
                })
    return run_dir


def _normalized_words(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return set(re.findall(r"[a-z0-9]+", normalized))


_RELATION_STOPWORDS = {
    "a", "au", "aux", "d", "de", "des", "du", "en", "et", "l", "la",
    "le", "les", "un", "une",
}


def keyword_coverage(answer: str, expected_keywords: list[str] | None) -> float:
    """Couverture déterministe minimale, complémentaire à une revue humaine."""
    if not expected_keywords:
        return 1.0
    normalized_answer = " ".join(sorted(_normalized_words(answer)))
    hits = 0
    for keyword in expected_keywords:
        normalized_keyword = " ".join(sorted(_normalized_words(keyword)))
        if normalized_keyword and all(word in normalized_answer.split() for word in normalized_keyword.split()):
            hits += 1
    return hits / len(expected_keywords)


_NUMBER_MENTION_RE = re.compile(
    r"(?<![\w])[-+−]?(?:\d{1,3}(?:[\s\u00a0\u202f]\d{3})+|\d+)"
    r"(?:[.,]\d+)?"
)


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text).casefold())
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _canonical_number(value: str) -> str:
    normalized = (
        value.replace("−", "-")
        .replace("\u00a0", "")
        .replace("\u202f", "")
        .replace(" ", "")
        .replace(",", ".")
    )
    try:
        number = Decimal(normalized)
    except InvalidOperation:
        return normalized
    formatted = format(number, "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _number_mentions(text: str) -> list[tuple[str, str]]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;])\s+|[\r\n]+", str(text))
        if sentence.strip()
    ]
    mentions = []
    for sentence in sentences:
        for match in _NUMBER_MENTION_RE.finditer(sentence):
            mentions.append((_canonical_number(match.group(0)), sentence))
    return mentions


def is_documentary_abstention(answer: str) -> bool:
    """Detecte les formulations d'abstention sans imposer une phrase unique."""
    normalized = _normalized_text(answer)
    patterns = (
        r"\bje ne (?:trouve|dispose|peux determiner) pas\b",
        r"\bn[' ]?est pas (?:mentionne|mentionnee|indique|indiquee|disponible|presente|precise)\b",
        r"\bne (?:permet|permettent|mentionne|mentionnent|contient|contiennent|fournit|fournissent) pas\b",
        r"\baucun(?:e)? (?:source|document|information|donnee|indication)\b",
        r"\baucun(?:e)? (?:chiffre|valeur|resultat) (?:specifique|exact|disponible)\b",
        r"\bne (?:mentionne|mentionnent|fournit|fournissent|donne|donnent) "
        r"aucun(?:e)? (?:chiffre|valeur|donnee|information)\b",
        r"\binformation (?:est )?(?:absente|indisponible|non disponible)\b",
        r"\bimpossible (?:de|a) determiner\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def reference_relation_score(question: str, reference: str, answer: str) -> float:
    """Verifie qu'une valeur est reliee a la bonne periode, pas seulement citee.

    Les nombres attendus sont compares apres normalisation des espaces et de la
    virgule decimale. Si une phrase associe la valeur a une annee differente de
    celle de la question, elle n'est pas comptee comme correcte.
    """
    question_years = set(re.findall(r"\b(?:19|20)\d{2}\b", _normalized_text(question)))
    reference_mentions = [
        value for value, _sentence in _number_mentions(reference)
        if not (value.isdigit() and 1900 <= int(value) <= 2099)
    ]
    if not reference_mentions:
        reference_tokens = sorted(
            _normalized_words(reference) - _RELATION_STOPWORDS
        )
        return keyword_coverage(answer, reference_tokens)

    answer_mentions = _number_mentions(answer)
    decrease_context = _normalized_text(" ".join((question, reference)))
    decrease_is_requested = bool(re.search(
        r"\b(?:baisse|recul|diminution|chute|contraction|repli|decru|diminue)\w*\b",
        decrease_context,
    ))
    relation_hits = 0
    for expected in dict.fromkeys(reference_mentions):
        matched = False
        for observed, sentence in answer_mentions:
            if observed != expected:
                # Une baisse peut s'ecrire « baisse de 10 % » ou « -10 % ».
                # L'equivalence par valeur absolue n'est autorisee que lorsque
                # la direction negative est explicite dans la question/reference
                # ou dans la phrase evaluee ; elle ne masque donc pas une erreur
                # de signe sur une mesure neutre.
                try:
                    same_magnitude = abs(Decimal(observed)) == abs(Decimal(expected))
                except InvalidOperation:
                    same_magnitude = False
                sentence_decrease = bool(re.search(
                    r"\b(?:baisse|recul|diminution|chute|contraction|repli|decru|diminue)\w*\b",
                    _normalized_text(sentence),
                ))
                if not (same_magnitude and (decrease_is_requested or sentence_decrease)):
                    continue
            sentence_years = set(
                re.findall(r"\b(?:19|20)\d{2}\b", _normalized_text(sentence))
            )
            if question_years and sentence_years and not (question_years & sentence_years):
                continue
            matched = True
            break
        relation_hits += int(matched)
    numeric_score = relation_hits / len(dict.fromkeys(reference_mentions))

    lexical_tokens = [
        token for token in _normalized_words(reference)
        if (
            not token.isdigit()
            and len(token) > 2
            and token not in _RELATION_STOPWORDS
        )
    ]
    lexical_score = keyword_coverage(answer, lexical_tokens) if lexical_tokens else 1.0
    return 0.85 * numeric_score + 0.15 * lexical_score


def is_compact_numeric_reference(reference: str) -> bool:
    """Indique qu'une référence attend principalement une valeur scalaire."""
    non_year_numbers = [
        value for value, _sentence in _number_mentions(reference)
        if not (value.isdigit() and 1900 <= int(value) <= 2099)
    ]
    return bool(non_year_numbers) and len(str(reference).split()) <= 18


def lead_answer_text(answer: str) -> str:
    """Isole la première affirmation, celle qui porte la réponse principale."""
    first_block = next(
        (line.strip() for line in str(answer).splitlines() if line.strip()),
        "",
    )
    if not first_block:
        return ""
    match = re.search(
        r"[!?](?=\s|$)|\.(?=\s+[A-ZÀ-ÖØ-Þ]|$)",
        first_block,
    )
    return first_block[:match.end()].strip() if match else first_block


def lead_reference_relation_score(
    question: str,
    reference: str,
    answer: str,
) -> float:
    """Exige la valeur attendue dans l'affirmation principale si elle est scalaire."""
    if not is_compact_numeric_reference(reference):
        return reference_relation_score(question, reference, answer)
    return reference_relation_score(
        question,
        reference,
        lead_answer_text(answer),
    )


# ---------------------------------------------------------------------------
# Métriques de génération via Ragas (Faithfulness, Context Precision, Answer Relevancy)
# ---------------------------------------------------------------------------

def run_ragas_evaluation(pipeline: RAGPipeline, examples: list[GoldenExample]) -> dict:
    """Évalue faithfulness / context_precision / answer_relevancy avec Ragas.

    Le LLM juge est le modèle Ollama local configuré (evaluation.ragas_judge_model),
    exposé à Ragas via l'intégration langchain-ollama. Si ces dépendances ne sont
    pas installées, la fonction lève une ImportError explicite plutôt que d'échouer
    silencieusement — l'appelant (notebook) peut choisir de l'ignorer.
    """
    from datasets import Dataset
    from langchain_ollama import ChatOllama
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for example in examples:
        chunks = pipeline.retriever.retrieve(example.question)
        response = pipeline.generator.generate(example.question, chunks)
        rows["question"].append(example.question)
        rows["answer"].append(response.answer)
        rows["contexts"].append([c.parent_text for c in chunks] or [""])
        rows["ground_truth"].append(example.reference_answer)

    dataset = Dataset.from_dict(rows)

    judge_llm = LangchainLLMWrapper(ChatOllama(model=pipeline.config.evaluation.ragas_judge_model))

    class _STEmbeddingsAdapter:
        """Adapte EmbeddingModel (déjà chargé en GPU) à l'interface attendue par Ragas."""

        def __init__(self, embedding_model):
            self._model = embedding_model

        def embed_query(self, text: str) -> list[float]:
            return self._model.encode([text])[0].tolist()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._model.encode(texts).tolist()

    judge_embeddings = LangchainEmbeddingsWrapper(_STEmbeddingsAdapter(pipeline.embedding_model))

    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    return result.to_pandas().mean(numeric_only=True).to_dict()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    pipeline = RAGPipeline.from_config()
    examples = load_golden_set(pipeline.config.evaluation.golden_set_full_path)
    evaluator = RetrievalEvaluator(pipeline.config.evaluation)
    results = evaluator.evaluate(pipeline, examples)
    logger.info("Résultats retrieval : %s", results)


if __name__ == "__main__":
    main()
