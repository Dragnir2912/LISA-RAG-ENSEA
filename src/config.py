"""Chargement et validation typée de config.yaml.

Toute la configuration du pipeline (modèles, tailles de chunks, top-k, chemins)
passe par ce module afin qu'aucun paramètre important ne soit codé en dur ailleurs
dans src/.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ProjectConfig(BaseModel):
    base_url: str
    sitemap_url: str
    data_dir: str

    @property
    def data_path(self) -> Path:
        return PROJECT_ROOT / self.data_dir


class ScrapingConfig(BaseModel):
    max_pages: int = 0
    max_concurrency: int = 4
    request_delay_seconds: float = 0.5
    user_agent: str
    download_pdfs: bool = True


class FacebookScrapingConfig(BaseModel):
    """Collecte Facebook via Apify, isolée du corpus institutionnel."""

    enabled: bool = False
    pages: dict[str, str] = Field(default_factory=dict)
    page_aliases: dict[str, list[str]] = Field(default_factory=dict)
    max_posts_per_page: int = 30
    actor_id: str = "apify/facebook-posts-scraper"
    raw_dir: str = "data/facebook/raw"
    cleaned_candidate_dir: str = "data/facebook/cleaned"
    chunks_candidate_path: str = "data/facebook/chunks.jsonl"
    vectorstore_candidate_uri: str = "data/facebook/lancedb"
    vectorstore_candidate_table: str = "facebook_chunks"
    bm25_candidate_path: str = "data/facebook/bm25_index.pkl"
    min_post_characters: int = 20
    min_rag_characters: int = 80
    normalize_unicode: bool = True
    deduplicate_content: bool = True
    retrieval_enabled: bool = True
    max_combined_results: int = 5
    institutional_bias_both: float = 0.02

    @property
    def raw_full_path(self) -> Path:
        return PROJECT_ROOT / self.raw_dir

    @property
    def cleaned_candidate_full_path(self) -> Path:
        return PROJECT_ROOT / self.cleaned_candidate_dir

    @property
    def chunks_candidate_full_path(self) -> Path:
        return PROJECT_ROOT / self.chunks_candidate_path

    @property
    def vectorstore_candidate_full_path(self) -> Path:
        return PROJECT_ROOT / self.vectorstore_candidate_uri

    @property
    def bm25_candidate_full_path(self) -> Path:
        return PROJECT_ROOT / self.bm25_candidate_path


class LocalDocumentsConfig(BaseModel):
    """Documents originaux déposés localement avant la construction du corpus."""

    enabled: bool = True
    source_dir: str = "data/sources/eco_rci"
    raw_dir: str = "data/raw"
    report_path: str = "data/import_reports/eco_rci.json"
    public_route_prefix: str = "/api/source-files"
    chunks_path: str = "data/economic/chunks.jsonl"
    vectors_path: str = "data/economic/vectors.npy"
    vectorstore_uri: str = "data/economic/lancedb"
    vectorstore_table: str = "economic_chunks"
    bm25_index_path: str = "data/economic/bm25_index.pkl"
    retrieval_enabled: bool = True
    extract_table_facts: bool = True
    table_min_periods: int = 2
    table_max_facts_per_page: int = 500

    @property
    def source_full_path(self) -> Path:
        return PROJECT_ROOT / self.source_dir

    @property
    def raw_full_path(self) -> Path:
        return PROJECT_ROOT / self.raw_dir

    @property
    def report_full_path(self) -> Path:
        return PROJECT_ROOT / self.report_path

    @property
    def chunks_full_path(self) -> Path:
        return PROJECT_ROOT / self.chunks_path

    @property
    def vectors_full_path(self) -> Path:
        return PROJECT_ROOT / self.vectors_path


class CleaningConfig(BaseModel):
    min_content_length: int = 80
    boilerplate_dedup_threshold: float = 0.30


class ChunkingConfig(BaseModel):
    strategy: str
    child_chunk_size_tokens: int = 400
    parent_chunk_size_tokens: int = 1800
    chunk_overlap_ratio: float = 0.15
    headers_to_split_on: list[list[str]] = Field(default_factory=list)


class EmbeddingsConfig(BaseModel):
    model_name: str
    device: str = "cuda"
    precision: str = "fp16"
    batch_size: int = 16
    normalize: bool = True
    alternatives: list[str] = Field(default_factory=list)


class VectorstoreConfig(BaseModel):
    backend: str = "lancedb"
    uri: str
    table_name: str
    bm25_index_path: str

    @property
    def uri_path(self) -> Path:
        return PROJECT_ROOT / self.uri

    @property
    def bm25_index_full_path(self) -> Path:
        return PROJECT_ROOT / self.bm25_index_path


class RetrievalConfig(BaseModel):
    top_k_dense: int = 30
    top_k_bm25: int = 30
    rrf_k: int = 60
    mmr_lambda: float = 0.5
    mmr_top_k: int = 15
    max_candidates_per_parent: int = 1
    max_candidates_per_url: int = 2
    max_results_per_url: int = 1
    max_table_fact_results_per_url: int = 2
    max_result_vector_similarity: float = 0.92
    rerank_top_k: int = 5
    exhaustive_max_results: int = 10
    exhaustive_top_k_dense: int = 80
    exhaustive_top_k_bm25: int = 120
    exhaustive_mmr_top_k: int = 40
    exhaustive_mmr_lambda: float = 0.85
    exhaustive_max_result_vector_similarity: float = 0.995
    exhaustive_anchor_max_document_frequency: int = 30
    exhaustive_anchor_max_candidates: int = 20
    exhaustive_anchor_rerank_bonus: float = 0.15
    exhaustive_anchor_title_min_rerank_score: float = 0.30
    exhaustive_noncoverage_score_floor: float = 0.55
    exhaustive_noncoverage_relative_margin: float = 0.12
    min_similarity_score: float = 0.25
    lexical_rescue_max_bm25_rank: int = 1
    lexical_rescue_min_rerank_score: float = 0.85
    table_fact_base_boost: float = 0.005
    table_fact_period_boost: float = 0.030
    table_fact_period_mismatch_penalty: float = 0.080
    table_fact_indicator_boost: float = 0.020
    table_fact_min_indicator_coverage: float = 0.45
    table_fact_focus_boost: float = 0.100
    table_fact_measure_alignment_weight: float = 0.100
    table_fact_channel_rank_weight: float = 0.180
    table_fact_preferred_period_boost: float = 0.060
    table_fact_secondary_period_penalty: float = 0.040
    table_fact_candidate_top_k: int = 8
    table_fact_rrf_weight: float = 1.0
    page_rescue_enabled: bool = True
    page_rescue_top_documents: int = 5
    page_rescue_per_document: int = 2
    page_rescue_rrf_weight: float = 0.8
    page_rescue_rank_weight: float = 0.12
    title_candidate_enabled: bool = False
    title_candidate_top_k: int = 1
    title_candidate_min_coverage: float = 0.60


class RerankerConfig(BaseModel):
    model_name: str
    device: str = "cuda"
    precision: str = "fp16"


class GenerationConfig(BaseModel):
    provider: str = "ollama"
    model_name: str
    alternatives: list[str] = Field(default_factory=list)
    temperature: float = 0.1
    thinking: bool = False
    min_num_ctx: int = 6144
    num_ctx: int = 6144
    context_budget_tokens: int = 6000
    max_passage_tokens: int = 1000
    reserved_output_tokens: int = 1536
    max_output_tokens: int = 768
    thinking_output_tokens: int = 1536
    direct_output_tokens: int = 320
    numeric_output_tokens: int = 192
    explanation_output_tokens: int = 520
    enumeration_output_tokens: int = 640
    repeat_penalty: float = 1.15
    repeat_last_n: int = 256
    max_consecutive_duplicate_list_items: int = 2
    response_cache_size: int = 128
    keep_alive: str = "30m"
    num_batch: int = 256
    # ``None`` laisse Ollama choisir le matériel disponible. ``0`` force le
    # CPU, ce qui rend le même paquet déployable sur un serveur sans GPU.
    num_gpu: int | None = None
    stream: bool = True


class RuntimeConfig(BaseModel):
    """Profil du serveur quotidien, distinct de l'indexation hors ligne."""

    performance_profile: str = "generation_priority_8gb"
    online_embeddings_device: str = "cpu"
    online_reranker_device: str = "cpu"
    torch_cpu_threads: int = 8


class EvaluationConfig(BaseModel):
    golden_set_path: str
    expanded_golden_set_path: str = "data/golden_set_expanded.jsonl"
    banque_100_golden_set_path: str = "data/golden_sets/banque_100_questions_corrigee.jsonl"
    runs_dir: str = "data/evaluation_runs"
    k_values: list[int] = Field(default_factory=lambda: [3, 5, 10])
    ragas_judge_model: str

    @property
    def golden_set_full_path(self) -> Path:
        return PROJECT_ROOT / self.golden_set_path

    @property
    def expanded_golden_set_full_path(self) -> Path:
        return PROJECT_ROOT / self.expanded_golden_set_path

    @property
    def banque_100_golden_set_full_path(self) -> Path:
        return PROJECT_ROOT / self.banque_100_golden_set_path

    @property
    def runs_full_path(self) -> Path:
        return PROJECT_ROOT / self.runs_dir


class ObservabilityConfig(BaseModel):
    enabled: bool = True
    traces_path: str = "data/observability/traces.jsonl"
    log_questions: bool = True
    log_answers: bool = False
    max_file_mb: int = 50
    keep_files: int = 3

    @property
    def traces_full_path(self) -> Path:
        return PROJECT_ROOT / self.traces_path


class Config(BaseModel):
    project: ProjectConfig
    scraping: ScrapingConfig
    facebook_scraping: FacebookScrapingConfig = Field(default_factory=FacebookScrapingConfig)
    local_documents: LocalDocumentsConfig = Field(default_factory=LocalDocumentsConfig)
    cleaning: CleaningConfig
    chunking: ChunkingConfig
    embeddings: EmbeddingsConfig
    vectorstore: VectorstoreConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    generation: GenerationConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Charge et valide config.yaml. Lève une erreur claire si un champ est invalide."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
