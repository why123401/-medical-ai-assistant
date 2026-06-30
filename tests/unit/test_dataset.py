"""Unit tests for eval/dataset — golden QA dataset loading."""

import json
import tempfile
from pathlib import Path

import pytest

from src.eval.dataset import load_golden_set, save_golden_set, filter_by_category, filter_by_device


class TestDatasetLoading:
    """Test QA dataset I/O operations."""

    def test_save_and_load_jsonl(self):
        pairs = [
            {"question": "Q1", "expected_answer": "A1", "device_code": "D1", "category": "spec"},
            {"question": "Q2", "expected_answer": "A2", "device_code": "D2", "category": "fault"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            path = f.name

        try:
            save_golden_set(pairs, path)
            loaded = load_golden_set(path)
            assert len(loaded) == 2
            assert loaded[0]["question"] == "Q1"
            assert loaded[1]["device_code"] == "D2"
        finally:
            Path(path).unlink()

    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            path = f.name

        try:
            loaded = load_golden_set(path)
            assert loaded == []
        finally:
            Path(path).unlink()

    def test_load_chinese_content(self):
        pairs = [{"question": "呼吸机 E104 报警", "expected_answer": "检查管路"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            path = f.name

        try:
            save_golden_set(pairs, path)
            loaded = load_golden_set(path)
            assert loaded[0]["question"] == "呼吸机 E104 报警"
        finally:
            Path(path).unlink()


class TestFiltering:
    """Test QA pair filtering."""

    def test_filter_by_category(self):
        pairs = [
            {"category": "spec", "device_code": "D1"},
            {"category": "fault", "device_code": "D1"},
            {"category": "spec", "device_code": "D2"},
        ]
        specs = filter_by_category(pairs, "spec")
        assert len(specs) == 2
        assert all(p["category"] == "spec" for p in specs)

    def test_filter_by_device(self):
        pairs = [
            {"category": "spec", "device_code": "D1"},
            {"category": "fault", "device_code": "D1"},
            {"category": "spec", "device_code": "D2"},
        ]
        d1 = filter_by_device(pairs, "D1")
        assert len(d1) == 2
        assert all(p["device_code"] == "D1" for p in d1)

    def test_filter_no_match(self):
        pairs = [{"category": "spec", "device_code": "D1"}]
        result = filter_by_category(pairs, "nonexistent")
        assert result == []
