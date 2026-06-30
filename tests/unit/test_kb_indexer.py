"""Unit tests for kb/indexer — document ingestion pipeline."""

import tempfile
from pathlib import Path

import pytest

from src.kb.indexer import _compute_md5, _load_txt


class TestMD5Helpers:
    """Test MD5-based deduplication helpers."""

    def test_compute_md5(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            path = f.name

        try:
            md5 = _compute_md5(path)
            assert len(md5) == 32  # MD5 hex digest length
            # Same content should produce same hash
            assert md5 == _compute_md5(path)
        finally:
            Path(path).unlink()

    def test_different_content_different_md5(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write("content A")
            path_a = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write("content B")
            path_b = f2.name

        try:
            assert _compute_md5(path_a) != _compute_md5(path_b)
        finally:
            Path(path_a).unlink()
            Path(path_b).unlink()


class TestLoadTxt:
    """Test text file loading."""

    def test_load_txt_returns_documents(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("测试内容")
            path = f.name

        try:
            docs = _load_txt(path)
            assert len(docs) == 1
            assert docs[0].page_content == "测试内容"
            assert docs[0].metadata["type"] == "txt"
        finally:
            Path(path).unlink()
