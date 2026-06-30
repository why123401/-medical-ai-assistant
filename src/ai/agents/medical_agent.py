"""Medical device Agent implementation.

Wraps the RAG pipeline and KG tools into a unified agent interface.
Supports both single-turn invoke and streaming responses.

This is the bridge between the old LangChain ReAct agent and the new
modular architecture. It can be swapped with a LangGraph-based agent
in future iterations.
"""

import re
from typing import Any, Generator

from src.ai.agents.base import BaseAgent
from src.ai.agents.tools.medical_tools import (
    rag_search,
    get_equipment_spec,
    get_fault_tree,
    get_maintenance_schedule,
    list_devices,
)
from src.ai.rag.pipeline import RAGPipeline
from src.memory.manager import memory_manager
from src.shared.logging import get_logger

logger = get_logger("ai.agents.medical")

# All tools available to the agent
AGENT_TOOLS = [
    rag_search,
    get_equipment_spec,
    get_fault_tree,
    get_maintenance_schedule,
    list_devices,
]


class MedicalAgent(BaseAgent):
    """Medical device Q&A agent using RAG + Knowledge Graph tools.

    The agent follows a simple but effective pattern:
      1. Parse user intent (spec query / fault diagnosis / maintenance / general)
      2. Select appropriate tools (KG for structured, RAG for unstructured)
      3. Combine results into a coherent answer
      4. Track conversation history in memory

    Usage:
        agent = MedicalAgent()
        result = agent.invoke("E104 报警怎么处理")
        print(result["reply"])
    """

    def __init__(self, pipeline: RAGPipeline | None = None):
        self.pipeline = pipeline or RAGPipeline()
        self._memory = memory_manager

    def invoke(self, query: str, conversation_id: str | None = None) -> dict[str, Any]:
        """Execute a single-turn agent interaction."""
        memory = self._memory.get_or_create(conversation_id)
        memory.add_message("user", query)

        # Build conversation context for multi-turn awareness
        context_messages = memory.get_context_messages()

        # Intent-based tool selection (pass context for multi-turn awareness)
        answer, sources = self._resolve(query, context=context_messages)

        memory.add_message("assistant", answer)
        context = memory.get_context_messages()

        result = {
            "reply": answer,
            "sources": sources,
            "conversation_id": memory.conversation_id,
            "context": context,
        }

        logger.info(f"Agent response: query_len={len(query)}, answer_len={len(answer)}")
        return result

    def stream(self, query: str, conversation_id: str | None = None) -> Generator[str, None, None]:
        """Streaming version — yields the answer character by character."""
        result = self.invoke(query, conversation_id)
        answer = result["reply"]
        for char in answer:
            yield char
        yield "\n"

    def get_tools(self) -> list[Any]:
        return AGENT_TOOLS.copy()

    def _resolve(self, query: str, context: list[dict[str, str]] | None = None) -> tuple[str, list[dict]]:
        r"""Resolve user query using appropriate tool(s).

        Strategy:
          1. Detect chit-chat / greeting → reply directly, skip RAG
          2. Check for fault code pattern (E\d+, F\d+, A\d+, M\d+, U\d+)
          3. Check for device code pattern (MED-XXX)
          4. Check for maintenance keywords
          5. Fall back to RAG search (with conversation context)
        """
        # --- Chit-chat / greeting detection (skip expensive RAG) ---
        chitchat_patterns = [
            (r'^(你好|您好|嗨|hello|hi|hey|哈喽)\s*$',
             '您好！我是医疗设备知识助手，专注于为您解答呼吸机、监护仪等设备的技术问题。请问有什么可以帮您？'),
            (r'^(谢谢|感谢|多谢|thx|thanks)\s*$', '不客气，随时为您服务！'),
            (r'^(再见|拜拜|bye|goodbye)\s*$', '再见，祝您工作顺利！'),
            (r'^(嗯|哦|啊|好的|收到)\s*$', '好的，有其他问题随时问我。'),
            (r'^(你是谁|你叫什么|介绍一下自己)\s*$',
             '我是医疗设备知识助手，基于 RAG 技术为您提供呼吸机、监护仪、CT 扫描仪等设备的故障诊断、规格参数和维护计划查询服务。'),
        ]
        for pattern, reply in chitchat_patterns:
            if re.search(pattern, query.strip(), re.IGNORECASE):
                return reply, []

        # Pattern 1: Fault code lookup
        fault_match = re.search(r'([A-F]\d{3})', query.upper())
        if fault_match:
            fault_code = fault_match.group(1)
            device_match = re.search(r'(MED-\w+-\w+)', query)
            device_code = device_match.group(1) if device_match else None

            try:
                from src.ai.agents.tools.medical_tools import _load_kg, _format_fault
                kg = _load_kg()
                device = kg.get(device_code) if device_code else None
                if device and fault_code in device.get("faults", {}):
                    fault = device["faults"][fault_code]
                    answer = _format_fault(device_code, fault_code, fault)
                    return answer, [{"index": 1, "source": "knowledge_graph", "content": answer[:200]}]
                for dev_code, dev in kg.items():
                    faults = dev.get("faults", {})
                    if fault_code in faults:
                        answer = _format_fault(dev_code, fault_code, faults[fault_code])
                        return answer, [{"index": 1, "source": "knowledge_graph", "content": answer[:200]}]
                return f"未找到故障码 {fault_code} 的诊断信息", []
            except Exception as e:
                logger.warning(f"Fault tree lookup failed: {e}, falling back to RAG")

        # Pattern 2: Device spec lookup (any query with device code)
        device_match = re.search(r'([A-Z]+-\w+(?:-\w+)?)', query)
        if device_match:
            device_code = device_match.group(1)
            try:
                from src.ai.agents.tools.medical_tools import _load_kg
                kg = _load_kg()
                device = kg.get(device_code)
                if device:
                    lines = [f"设备代码: {device_code}", f"设备名称: {device.get('name', '未知')}"]
                    lines.append(f"类别: {device.get('category', '未知')}")
                    lines.append("规格参数:")
                    for key, value in device.get("specs", {}).items():
                        lines.append(f"  - {key}: {value}")
                    answer = "\n".join(lines)
                    return answer, [{"index": 1, "source": "knowledge_graph", "content": answer[:200]}]
                return f"未找到设备 {device_code} 的信息", []
            except Exception as e:
                logger.warning(f"Equipment spec lookup failed: {e}, falling back to RAG")

        # Pattern 3: Maintenance query
        if any(kw in query for kw in ["维护", "保养", "校准", "更换", "检查", "日常", "定期"]):
            device_match = re.search(r'([A-Z]+-\w+(?:-\w+)?)', query)
            device_code = device_match.group(1) if device_match else None
            interval = "daily"
            if "月" in query or "monthly" in query:
                interval = "monthly"
            elif "年" in query or "yearly" in query:
                interval = "yearly"
            elif "周" in query or "weekly" in query:
                interval = "weekly"

            if device_code:
                try:
                    from src.ai.agents.tools.medical_tools import _load_kg
                    kg = _load_kg()
                    device = kg.get(device_code)
                    if device:
                        schedules = device.get("maintenance_intervals", {})
                        tasks = schedules.get(interval)
                        if tasks:
                            device_name = device.get("name", device_code)
                            lines = [f"{device_name}（{device_code}）{interval} 维护计划:"]
                            for i, task in enumerate(tasks, 1):
                                lines.append(f"  {i}. {task}")
                            answer = "\n".join(lines)
                            return answer, [{"index": 1, "source": "knowledge_graph", "content": answer[:200]}]
                        return f"设备 {device_code} 的 {interval} 维护计划不存在", []
                    return f"未找到设备 {device_code} 的信息", []
                except Exception as e:
                    logger.warning(f"Maintenance lookup failed: {e}, falling back to RAG")

        # Pattern 4: RAG search (catch-all, with conversation context)
        try:
            result = self.pipeline.invoke(query, context=context)
            return result["answer"], result["sources"]
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return "抱歉，AI 服务暂时不可用，请稍后再试。", []
