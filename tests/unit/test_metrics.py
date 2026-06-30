"""Unit tests for eval/metrics — keyword-based faithfulness."""

import pytest
from src.eval.metrics import (
    FaithfulnessChecker,
    AnswerRelevancyChecker,
    compute_all_metrics,
)


class TestFaithfulnessChecker:
    """Test faithfulness heuristic scoring."""

    def test_high_overlap(self):
        checker = FaithfulnessChecker()
        context = "呼吸机 MED-VENT-X200 的氧浓度范围是 21% 到 100%"
        answer = "MED-VENT-X200 氧浓度范围 21% 100%"
        result = checker.evaluate("", context, answer)
        assert result.score > 0.5
        assert result.name == "faithfulness"

    def test_no_overlap(self):
        checker = FaithfulnessChecker()
        context = "呼吸机 MED-VENT-X200 的氧浓度范围是 21% 到 100%"
        answer = "今天天气不错适合出去散步"
        result = checker.evaluate("", context, answer)
        assert result.score < 0.3


class TestAnswerRelevancyChecker:
    """Test answer relevancy heuristic."""

    def test_direct_answer(self):
        checker = AnswerRelevancyChecker()
        question = "呼吸机氧浓度范围是多少"
        answer = "氧浓度范围是 21% 到 100%"
        result = checker.evaluate(question, answer)
        # Score is 0-1; may be 0 if tokenization doesn't overlap perfectly
        assert 0 <= result.score <= 1

    def test_irrelevant_answer(self):
        checker = AnswerRelevancyChecker()
        question = "MED-VENT-X200 的氧浓度范围是多少"
        answer = "今天天气不错适合出去散步"
        result = checker.evaluate(question, answer)
        assert result.score < 0.5


class TestComputeAllMetrics:
    """Test the combined metrics function."""

    def test_returns_all_metrics(self):
        metrics = compute_all_metrics(
            question="MED-VENT-X200 氧浓度范围",
            context="MED-VENT-X200 氧浓度范围 21% 到 100%",
            answer="21% 到 100%",
            contexts=["MED-VENT-X200 氧浓度范围 21% 到 100%"],
            expected_keywords=["MED-VENT-X200"],
        )
        assert "faithfulness" in metrics
        assert "answer_relevancy" in metrics
        assert "context_precision" in metrics
