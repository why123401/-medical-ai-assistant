"""Evaluation metrics for the medical AI assistant.

Metrics implemented:
  - Faithfulness: Is the answer grounded in the retrieved context? (0-1)
  - AnswerRelevancy: Does the answer directly address the question? (0-1)
  - ContextPrecision: Are relevant chunks ranked higher? (0-1)
  - ContextRecall: Were all correct documents retrieved? (0-1)

Scoring modes:
  1. Heuristic (default): Keyword overlap, fast and free
  2. LLM-as-Judge (optional): Uses LLM to evaluate, more accurate but costly

Usage:
    # Quick heuristic evaluation
    metrics = compute_all_metrics(question, context, answer)

    # LLM-as-Judge (requires model)
    checker = FaithfulnessChecker(use_llm_judge=True, model=my_model)
    result = checker.evaluate(question, context, answer)
"""

import re
from dataclasses import dataclass, field
from typing import Any

from src.shared.logging import get_logger

logger = get_logger("eval.metrics")


@dataclass
class MetricResult:
    """Single metric evaluation result."""
    name: str
    score: float  # 0.0 - 1.0
    details: dict[str, Any] = field(default_factory=dict)


class FaithfulnessChecker:
    """Check if an answer is faithful to the retrieved context.

    Faithfulness is the most important RAG metric — it measures whether
    the answer contains information NOT present in the context (hallucination).
    """

    def __init__(self, use_llm_judge: bool = False, model: Any = None):
        self.use_llm_judge = use_llm_judge
        self.model = model

    def evaluate(self, question: str, context: str, answer: str) -> MetricResult:
        if self.use_llm_judge and self.model:
            return self._llm_judge(question, context, answer)
        return self._heuristic(context, answer)

    def _heuristic(self, context: str, answer: str) -> MetricResult:
        """Keyword-based faithfulness heuristic.

        Measures overlap between context and answer terms.
        Penalizes answers that are much longer than the context coverage.
        """
        context_terms = set(re.findall(r'[一-鿿\w]+', context))
        answer_terms = set(re.findall(r'[一-鿿\w]+', answer))

        if not context_terms:
            return MetricResult("faithfulness", 0.0, {"reason": "empty_context"})

        overlap = len(context_terms & answer_terms) / len(context_terms)
        # Penalize if answer introduces many new terms not in context
        new_term_ratio = len(answer_terms - context_terms) / max(len(answer_terms), 1)
        score = overlap * (1 - new_term_ratio * 0.5)  # 50% penalty for new terms

        return MetricResult(
            "faithfulness",
            round(max(0.0, min(1.0, score)), 4),
            {
                "context_terms": len(context_terms),
                "answer_terms": len(answer_terms),
                "overlap_ratio": round(overlap, 4),
                "new_term_ratio": round(new_term_ratio, 4),
            },
        )

    def _llm_judge(self, question: str, context: str, answer: str) -> MetricResult:
        """LLM-as-Judge for faithfulness scoring.

        Asks the LLM to check if the answer contains any information
        not present in the context. Returns 1.0 (fully faithful) to 0.0 (full hallucination).
        """
        prompt = f"""你是一个评测专家。请判断以下回答是否忠实于参考资料。

问题：{question}

参考资料：
{context}

回答：
{answer}

请判断：回答中是否有参考资料中没有的信息？
- 如果回答完全基于参考资料 → 评分 1.0
- 如果回答大部分基于参考资料但有少量添加 → 评分 0.7
- 如果回答有一半以上信息不在参考资料中 → 评分 0.3
- 如果回答完全是编造的 → 评分 0.0

只输出一个 0.0 到 1.0 之间的数字，不要输出其他内容。"""

        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            chain = ChatPromptTemplate.from_messages([("human", prompt)]) | self.model | StrOutputParser()
            raw = chain.invoke({}).strip()
            # Extract number from response
            score = float(re.search(r'[\d.]+', raw).group())
            return MetricResult("faithfulness", round(score, 4), {"method": "llm_judge", "raw_output": raw})
        except Exception as e:
            logger.error(f"LLM judge failed: {e}")
            return MetricResult("faithfulness", 0.5, {"method": "llm_judge_failed", "error": str(e)})


class AnswerRelevancyChecker:
    """Check if an answer directly addresses the question."""

    def evaluate(self, question: str, answer: str) -> MetricResult:
        return self._heuristic(question, answer)

    def _heuristic(self, question: str, answer: str) -> MetricResult:
        """Keyword overlap between question and answer.

        Direct answers tend to reuse key terms from the question.
        """
        q_terms = set(re.findall(r'[一-鿿\w]+', question))
        a_terms = set(re.findall(r'[一-鿿\w]+', answer))

        if not q_terms:
            return MetricResult("answer_relevancy", 0.0)

        overlap = len(q_terms & a_terms) / len(q_terms)
        # Scale up since direct answers reuse question terms
        score = min(1.0, overlap * 2)

        return MetricResult(
            "answer_relevancy",
            round(score, 4),
            {"question_terms": len(q_terms), "overlap_ratio": round(overlap, 4)},
        )


class ContextPrecisionChecker:
    """Check if the retrieved chunks contain relevant information."""

    def evaluate(self, question: str, contexts: list[str], expected_keywords: list[str] | None = None) -> MetricResult:
        if expected_keywords:
            return self._with_expectations(contexts, expected_keywords)
        return self._heuristic(question, contexts)

    def _with_expectations(self, contexts: list[str], expected_keywords: list[str]) -> MetricResult:
        """Check how many top chunks contain expected keywords."""
        relevant_count = 0
        for ctx in contexts:
            ctx_terms = set(re.findall(r'[一-鿿\w]+', ctx))
            if any(kw in ctx_terms for kw in expected_keywords):
                relevant_count += 1

        precision = relevant_count / max(len(contexts), 1)
        return MetricResult(
            "context_precision",
            round(precision, 4),
            {"relevant_chunks": relevant_count, "total_chunks": len(contexts)},
        )


class ContextRecallChecker:
    """Check if all correct documents were retrieved."""

    def evaluate(self, expected_sources: list[str], retrieved_sources: list[str]) -> MetricResult:
        if not expected_sources:
            return MetricResult("context_recall", 1.0)

        found = sum(1 for src in expected_sources if src in retrieved_sources)
        recall = found / len(expected_sources)

        return MetricResult(
            "context_recall",
            round(recall, 4),
            {"found": found, "expected": len(expected_sources)},
        )


def compute_all_metrics(
    question: str,
    context: str,
    answer: str,
    contexts: list[str] | None = None,
    expected_keywords: list[str] | None = None,
) -> dict[str, float]:
    """Compute all metrics for a single Q&A pair.

    Returns dict of metric_name -> score.
    """
    faithfulness = FaithfulnessChecker().evaluate(question, context, answer)
    relevancy = AnswerRelevancyChecker().evaluate(question, answer)

    result = {
        "faithfulness": faithfulness.score,
        "answer_relevancy": relevancy.score,
    }

    if contexts and expected_keywords:
        precision = ContextPrecisionChecker().evaluate(question, contexts, expected_keywords)
        result["context_precision"] = precision.score

    return result
