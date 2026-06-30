"""Unit tests for kb/chunking — semantic-aware splitting."""

import pytest

from src.kb.chunking import create_splitter, chunk_by_device


class TestCreateSplitter:
    """Test the chunker factory produces correct parameters."""

    def test_returns_recursive_splitter(self):
        splitter = create_splitter()
        # LangChain stores these as private attributes
        assert splitter._chunk_size == 500
        assert splitter._chunk_overlap == 75

    def test_has_domain_separators(self):
        splitter = create_splitter()
        # Separators are stored as _separators
        assert "\n## " in splitter._separators


class TestChunkByDevice:
    """Test device-aware splitting."""

    def test_splits_on_device_codes(self):
        text = "MED-VENT-X200 呼吸机电压范围100-240V\nMED-CT-3200 CT扫描仪层数128"
        result = chunk_by_device(text)
        assert len(result) == 2
        assert result[0]["device_code"] == "MED-VENT-X200"
        assert result[1]["device_code"] == "MED-CT-3200"

    def test_empty_input(self):
        assert chunk_by_device("") == []

    def test_no_device_codes(self):
        text = "这是一段没有设备代码的普通文本"
        result = chunk_by_device(text)
        assert len(result) == 0
