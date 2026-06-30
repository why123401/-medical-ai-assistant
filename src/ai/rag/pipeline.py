"""RAG pipeline orchestrator: retrieval → rerank → generate.

This is the core inference pipeline that ties together hybrid retrieval,
cross-encoder reranking, and LLM generation with citation support.

Pipeline flow:
    Query ──→ HybridRetriever ──→ Top-5 docs ──→ Prompt + LLM ──→ Answer
                                      │
                                      └──→ Sources for citation

Usage:
    pipeline = RAGPipeline()           # auto-loads model + retriever
    result = pipeline.invoke("E104 报警怎么处理")
    print(result["answer"])
    print(result["sources"])
"""

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.ai.rag.retriever import HybridRetriever
from src.shared.config import settings
from src.shared.logging import get_logger

logger = get_logger("ai.rag.pipeline")


class RAGPipeline:
    """End-to-end RAG pipeline for medical device Q&A.

    Attributes:
        retriever: Hybrid retriever (BM25 + Vector + RRF)
        model: LangChain chat model (loaded from DashScope)
        chain: LangChain LCEL chain (prompt | model | parser)
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        model: Any | None = None,
        prompt_template: str | None = None,
    ):
        # Lazy initialization: don't load retriever/model until first invoke()
        self._retriever = retriever
        self._model = model
        self._prompt_template = prompt_template or self._load_prompt()
        self._chain = None

    @property
    def retriever(self) -> Any:
        """Lazy-load retriever on first access."""
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    @property
    def model(self) -> Any:
        """Lazy-load model on first access."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _build_chain(self):
        """Build the LangChain Expression Language chain.

        Chain: PromptTemplate → ChatModel → StrOutputParser
        """
        prompt = PromptTemplate.from_template(self._prompt_template)
        return prompt | self._model | StrOutputParser()

    @property
    def chain(self) -> Any:
        """Lazy-build chain on first access."""
        if self._chain is None:
            if self._model is None:
                self._model = self._load_model()
            if self._model:
                self._chain = self._build_chain()
        return self._chain

    @chain.setter
    def chain(self, value: Any) -> None:
        self._chain = value

    @staticmethod
    def _load_model() -> Any:
        """Load the chat model from DashScope.

        Uses ChatTongyi (the current DashScope wrapper in langchain-community).
        Falls back gracefully if no API key is configured.
        """
        try:
            from langchain_community.chat_models import ChatTongyi

            model = ChatTongyi(
                model=settings.primary_model,
                dashscope_api_key=settings.dashscope_api_key,
                temperature=0.7,
                timeout=30,
                max_retries=1,
            )
            logger.info(f"Loaded model: {settings.primary_model}")
            return model
        except Exception as e:
            logger.warning(f"Could not load chat model: {e}")
            return None

    @staticmethod
    def _load_prompt() -> str:
        """Load the system prompt for RAG answering."""
        prompt_path = f"{settings.prompts_full_path}/rag_answer.txt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.debug(f"Prompt file not found: {prompt_path}, using default")
            return RAGPipeline._default_prompt_text()

    @staticmethod
    def _default_prompt_text() -> str:
        return """你是专业的医疗设备知识助手，也能进行日常交流。

用户问题：{input}

参考资料（带引用编号）：
{context}

要求：
1. 如果用户的问题与医疗设备相关，请严格基于参考资料回答，不编造、不添加未提及的内容，并在答案中标注参考来源编号 [ref-1]、[ref-2] 等
2. 如果参考资料中没有相关信息（即参考资料为空或与问题无关），请直接用自己的知识回答，不需要引用参考资料
3. 仅用中文回答，语气专业、友善、简洁
4. 如果是日常聊天（如问候、闲聊），自然回应即可，不必拘泥于医疗设备的角色
"""

    def invoke(self, query: str, context: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Execute the full RAG pipeline.

        Args:
            query: User question
            context: Optional conversation history from the agent layer.

        Returns:
            Dict with:
                - answer: str (the generated response)
                - sources: list[dict] (cited reference chunks)
        """
        # Step 1: Retrieve relevant chunks
        docs = self.retriever.invoke(query)
        if not docs:
            logger.warning(f"No documents retrieved for query: '{query[:80]}...' — will still attempt LLM fallback")

        # Step 2: Format with citation markers
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(f"[ref-{i + 1}] {source}: {doc.page_content}")

        kb_context = "\n\n".join(context_parts)

        # Step 3: Inject conversation history into the prompt context
        history_lines = []
        if context:
            for msg in context:
                role_label = "用户" if msg.get("role") == "user" else "助手"
                history_lines.append(f"    [{role_label}]: {msg.get('content', '')}")
        conversation_history = "\n".join(history_lines)

        # Step 4: Generate answer
        if self.chain:
            try:
                answer = self.chain.invoke({
                    "input": query,
                    "context": kb_context or "(无参考资料)",
                })
            except Exception as e:
                logger.error(f"LLM generation failed: query='{query[:80]}...', error={e}")
                answer = "抱歉，AI 服务暂时不可用，请稍后再试。"
        else:
            logger.error(f"Chain not built — model is None. Check dashscope_api_key and network.")
            answer = "模型未配置，请检查 API Key 和网络连接。"

        # Step 4: Build sources list for frontend display
        sources = [
            {
                "index": i + 1,
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content[:300],
            }
            for i, doc in enumerate(docs)
        ]

        logger.info(
            f"RAG pipeline: query='{query[:50]}...', "
            f"docs={len(docs)}, answer_len={len(answer)}"
        )

        return {
            "answer": answer.strip(),
            "sources": sources,
        }

    async def ainvoke(self, query: str) -> dict[str, Any]:
        """Async version of invoke (for FastAPI async endpoints)."""
        return self.invoke(query)


def create_rag_pipeline(**kwargs) -> RAGPipeline:
    """Factory function to create a RAG pipeline instance."""
    return RAGPipeline(**kwargs)
