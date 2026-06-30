"""Medical device domain tools for the AI agent.

Tools integrate with the knowledge graph (YAML-based) and RAG pipeline:
  - rag_search: RAG retrieval from vector store
  - get_equipment_spec: Fast lookup from KG YAML
  - get_fault_tree: Fault code lookup from KG YAML
  - get_maintenance_schedule: Maintenance plan lookup from KG YAML

The KG YAML file (data/kg/fault_trees.yaml) is the single source of truth
for device-fault-maintenance mappings. RAG handles unstructured document queries.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import tool

from src.ai.rag.pipeline import RAGPipeline
from src.shared.config import settings
from src.shared.logging import get_logger

logger = get_logger("ai.agents.tools")

# Lazy-loaded KG data
_kg_data: dict[str, Any] | None = None


def _load_kg() -> dict[str, Any]:
    """Load knowledge graph from YAML file (lazy-loaded, cached)."""
    global _kg_data
    if _kg_data is not None:
        return _kg_data

    kg_path = os.path.join(settings.kg_full_path, "fault_trees.yaml")
    if not os.path.exists(kg_path):
        logger.warning(f"KG file not found: {kg_path}")
        _kg_data = {}
        return _kg_data

    try:
        with open(kg_path, "r", encoding="utf-8") as f:
            _kg_data = yaml.safe_load(f) or {}
        logger.info(f"KG loaded: {len(_kg_data)} devices")
    except Exception as e:
        logger.error(f"Failed to load KG: {e}")
        _kg_data = {}

    return _kg_data


def _get_pipeline() -> RAGPipeline:
    """Lazy-load RAG pipeline."""
    try:
        return RAGPipeline()
    except Exception as e:
        logger.warning(f"RAG pipeline initialization failed: {e}")
        return None


# ============================================================
# Tool definitions
# ============================================================


@tool(description="从医疗设备知识库中检索参考资料。当需要查询设备参数、故障诊断、维护步骤时使用。")
def rag_search(query: str) -> str:
    """Search the medical device knowledge base via RAG.

    Args:
        query: Search term (device code, fault code, maintenance topic)

    Returns:
        Relevant knowledge base passages with citations
    """
    pipeline = _get_pipeline()
    if pipeline is None:
        return "知识库检索服务暂不可用"
    result = pipeline.invoke(query)
    return result["answer"]


@tool(description="查询设备规格参数。输入设备代码（如 MED-VENT-X200）获取详细规格。")
def get_equipment_spec(device_code: str) -> str:
    """Get equipment specification from the knowledge graph.

    Args:
        device_code: Medical device code (e.g., "MED-VENT-X200")

    Returns:
        Formatted specification string
    """
    kg = _load_kg()
    device = kg.get(device_code)
    if not device:
        return f"未找到设备 {device_code} 的信息"

    lines = [f"设备代码: {device_code}", f"设备名称: {device.get('name', '未知')}"]
    lines.append(f"类别: {device.get('category', '未知')}")
    lines.append("规格参数:")

    for key, value in device.get("specs", {}).items():
        lines.append(f"  - {key}: {value}")

    return "\n".join(lines)


@tool(description="查询故障诊断树。输入故障码（如 E104）和设备代码获取可能的原因和处理步骤。")
def get_fault_tree(fault_code: str, device_code: str | None = None) -> str:
    """Look up fault tree from the knowledge graph.

    Args:
        fault_code: Fault code (e.g., "E104")
        device_code: Optional device code to narrow scope

    Returns:
        Fault tree string with causes and procedures
    """
    kg = _load_kg()

    # If device_code specified, look in that device's faults
    if device_code:
        device = kg.get(device_code)
        if device:
            faults = device.get("faults", {})
            fault = faults.get(fault_code)
            if fault:
                return _format_fault(device_code, fault_code, fault)

    # Search all devices for this fault code
    for dev_code, device in kg.items():
        faults = device.get("faults", {})
        if fault_code in faults:
            fault = faults[fault_code]
            return _format_fault(dev_code, fault_code, fault)

    return f"未找到故障码 {fault_code} 的诊断信息"


def _format_fault(device_code: str, fault_code: str, fault: dict) -> str:
    """Format a fault entry into a readable string."""
    lines = [
        f"设备: {device_code} ({fault.get('name', '')})",
        f"故障码: {fault_code}",
        f"名称: {fault.get('name', '未知')}",
        f"严重程度: {fault.get('severity', 'unknown')}",
        "可能原因:",
    ]
    for cause in fault.get("causes", []):
        lines.append(f"  - {cause}")

    lines.append("处理步骤:")
    for i, proc in enumerate(fault.get("procedures", []), 1):
        lines.append(f"  {i}. {proc}")

    return "\n".join(lines)


@tool(description="查询设备维护计划。输入设备代码和维护周期（daily/weekly/monthly/yearly）获取维护项目。")
def get_maintenance_schedule(device_code: str, interval: str = "daily") -> str:
    """Get maintenance schedule from the knowledge graph.

    Args:
        device_code: Medical device code
        interval: Maintenance interval (daily/weekly/monthly/yearly)

    Returns:
        Formatted maintenance schedule
    """
    kg = _load_kg()
    device = kg.get(device_code)
    if not device:
        return f"未找到设备 {device_code} 的信息"

    schedules = device.get("maintenance_intervals", {})
    tasks = schedules.get(interval)

    if not tasks:
        available = ", ".join(schedules.keys()) if schedules else "无"
        return f"设备 {device_code} 的 {interval} 维护计划不存在。可用的周期: {available}"

    device_name = device.get("name", device_code)
    lines = [f"{device_name}（{device_code}）{interval} 维护计划:"]
    for i, task in enumerate(tasks, 1):
        lines.append(f"  {i}. {task}")

    return "\n".join(lines)


@tool(description="列出所有可用的设备代码和名称。")
def list_devices() -> str:
    """List all devices in the knowledge graph."""
    kg = _load_kg()
    if not kg:
        return "知识库中暂无设备信息"

    lines = ["医疗设备列表:"]
    for code, device in kg.items():
        lines.append(f"  {code}: {device.get('name', '未知')} ({device.get('category', '未知')})")

    return "\n".join(lines)
