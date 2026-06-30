"""Integration tests for the RAG pipeline.

These tests verify the full retrieval chain works end-to-end.
They require a populated ChromaDB vector store.
"""

import pytest
from src.ai.rag.pipeline import RAGPipeline


class TestRAGPipeline:
    """Test the RAG pipeline with a minimal setup."""

    def test_pipeline_initialization(self):
        """Pipeline should initialize without errors."""
        pipeline = RAGPipeline()
        assert pipeline.retriever is not None
        assert pipeline.prompt_template is not None

    def test_invoke_returns_answer_and_sources(self):
        """invoke() should return a dict with 'answer' and 'sources'."""
        pipeline = RAGPipeline()
        result = pipeline.invoke("呼吸机 MED-VENT-X200 报警 E104")
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)
