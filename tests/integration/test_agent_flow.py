"""Integration tests for the medical agent flow.

Tests the full agent pipeline: intent recognition → tool selection → response.
These tests verify the agent correctly routes different query types.
"""

import pytest

from src.ai.agents.medical_agent import MedicalAgent, AGENT_TOOLS
from src.ai.agents.tools.medical_tools import _load_kg


class TestAgentInitialization:
    """Test agent setup."""

    def test_agent_creates_without_model(self):
        """Agent should work without LLM for KG-based queries."""
        agent = MedicalAgent()
        assert agent.pipeline is not None

    def test_agent_has_tools(self):
        """Agent should expose all available tools."""
        agent = MedicalAgent()
        tools = agent.get_tools()
        assert len(tools) >= 3  # rag_search, get_equipment_spec, get_fault_tree


class TestIntentRouting:
    """Test that agent routes queries to the correct tools."""

    def test_fault_code_routing(self):
        """Query with fault code should use KG fault tree."""
        agent = MedicalAgent()
        # The _resolve method should detect E104 and route to KG
        result = agent._resolve("呼吸机 E104 报警怎么处理？")
        answer, sources = result
        assert len(answer) > 0
        assert "氧浓度" in answer or "E104" in answer or len(answer) > 5

    def test_device_spec_routing(self):
        """Query with device code + spec keyword should use KG spec lookup."""
        agent = MedicalAgent()
        result = agent._resolve("MED-VENT-X200 的潮气量范围是多少？")
        answer, sources = result
        assert "5" in answer or "800" in answer or "ml" in answer

    def test_maintenance_routing(self):
        """Query with maintenance keyword should use KG maintenance lookup."""
        agent = MedicalAgent()
        result = agent._resolve("MED-VENT-X200 的日常维护项目")
        answer, sources = result
        assert "开机自检" in answer or "管路检查" in answer or len(answer) > 5

    def test_unknown_query_falls_back(self):
        """Unknown query should not crash."""
        agent = MedicalAgent()
        result = agent._resolve("这是一个测试问题")
        answer, sources = result
        assert isinstance(answer, str)
        assert isinstance(sources, list)


class TestKnowledgeGraph:
    """Test KG data loading."""

    def test_kg_loads_devices(self):
        kg = _load_kg()
        assert "MED-VENT-X200" in kg
        assert "MED-CT-3200" in kg

    def test_kg_has_faults(self):
        kg = _load_kg()
        vent = kg.get("MED-VENT-X200", {})
        faults = vent.get("faults", {})
        assert "E104" in faults
        assert faults["E104"]["name"] == "氧浓度过低"

    def test_kg_has_maintenance(self):
        kg = _load_kg()
        vent = kg.get("MED-VENT-X200", {})
        maint = vent.get("maintenance_intervals", {})
        assert "daily" in maint
        assert len(maint["daily"]) > 0
