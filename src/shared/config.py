"""Pydantic Settings based configuration manager.

Replaces the old YAML-based config_handler.py with type-safe,
environment-variable-aware configuration.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from .env and defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- LLM / DashScope ---
    dashscope_api_key: str = ""
    dashscope_base_url: str | None = None

    # --- Model selection ---
    primary_model: str = "qwen-plus"
    fallback_model_strong: str = "qwen-max"
    fallback_model_weak: str = "qwen-turbo"

    # --- Embedding ---
    embedding_model: str = "text-embedding-v4"

    # --- RAG ---
    chunk_size: int = 500
    chunk_overlap: int = 75  # ~15%
    retrieval_top_k: int = 20  # before reranking
    rerank_top_n: int = 5  # after reranking
    rerank_model: str = "text-rerank-v2"

    # --- Paths ---
    project_root: str = str(Path(__file__).parents[2])
    knowledge_dir: str = "data/knowledge"
    kg_dir: str = "data/kg"
    eval_dir: str = "data/eval"
    prompts_dir: str = "prompts/v1"
    vector_store_dir: str = "chroma_db"
    md5_store: str = "md5.text"
    log_dir: str = "logs"

    # --- Routing / Circuit Breaker ---
    cb_failure_threshold: int = 5
    cb_timeout_seconds: int = 30
    cb_half_open_max: int = 3

    # --- Conversation memory ---
    conversation_window_size: int = 8  # verbatim turns to keep
    summary_trigger_turns: int = 16  # summarize after this many turns

    # --- Eval ---
    eval_golden_set: str = "data/eval/qa_golden_set.jsonl"
    eval_report_dir: str = "eval_reports"

    # --- Database ---
    db_url: str = "sqlite:///./medical_ai.db"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def knowledge_full_path(self) -> str:
        return str(Path(self.project_root) / self.knowledge_dir)

    @property
    def kg_full_path(self) -> str:
        return str(Path(self.project_root) / self.kg_dir)

    @property
    def prompts_full_path(self) -> str:
        return str(Path(self.project_root) / self.prompts_dir)

    @property
    def vector_store_full_path(self) -> str:
        return str(Path(self.project_root) / self.vector_store_dir)

    @property
    def md5_full_path(self) -> str:
        return str(Path(self.project_root) / self.md5_store)

    @property
    def log_full_path(self) -> str:
        return str(Path(self.project_root) / self.log_dir)


settings = Settings()
