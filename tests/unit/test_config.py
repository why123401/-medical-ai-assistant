"""Unit tests for shared/config — Pydantic Settings loading."""

import pytest
from src.shared.config import Settings


class TestSettings:
    """Test that Settings loads with defaults."""

    def test_default_model(self):
        s = Settings()
        assert s.primary_model == "qwen-plus"
        assert s.embedding_model == "text-embedding-v4"

    def test_default_paths(self):
        s = Settings()
        assert s.knowledge_dir == "data/knowledge"
        assert s.vector_store_dir == "chroma_db"

    def test_knowledge_full_path_exists(self):
        """knowledge_full_path should be a valid path string."""
        s = Settings()
        # On Windows it starts with drive letter, on Unix with /
        assert len(s.knowledge_full_path) > 0
        assert "data/knowledge" in s.knowledge_full_path or "data\\knowledge" in s.knowledge_full_path

    def test_circuit_breaker_defaults(self):
        s = Settings()
        assert s.cb_failure_threshold == 5
        assert s.cb_timeout_seconds == 30
