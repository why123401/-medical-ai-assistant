"""Batch evaluation runner for the medical AI assistant.

Usage:
    python -m src.eval.runner --golden-set data/eval/qa_golden_set.jsonl \
                              --output eval_reports/result_20260627.json

Runs all QA pairs through the RAG pipeline and computes metrics.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.eval.dataset import load_golden_set, save_golden_set
from src.eval.metrics import (
    compute_all_metrics,
    FaithfulnessChecker,
    AnswerRelevancyChecker,
    ContextPrecisionChecker,
    ContextRecallChecker,
    MetricResult,
)
from src.shared.config import settings
from src.shared.logging import get_logger

logger = get_logger("eval.runner")


class EvalRunner:
    """Executes batch evaluation on a golden QA dataset.

    Args:
        qa_engine: Callable that takes a question string and returns {"answer": str, "sources": [...]}
        golden_set: Path to golden QA dataset (JSONL)
    """

    def __init__(self, qa_engine: Callable[[str], dict[str, Any]], golden_set: str | None = None):
        self.qa_engine = qa_engine
        self.golden_set_path = golden_set or settings.eval_golden_set
        self.results: list[dict[str, Any]] = []

    def run(self, verbose: bool = True) -> dict[str, Any]:
        """Execute full evaluation pipeline.

        Returns:
            Evaluation summary with average scores per metric
        """
        qa_pairs = load_golden_set(self.golden_set_path)
        if verbose:
            logger.info(f"Evaluating {len(qa_pairs)} QA pairs from {self.golden_set_path}")

        results = []
        for i, pair in enumerate(qa_pairs):
            result = self._evaluate_one(i, pair)
            results.append(result)

            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{len(qa_pairs)}")

        self.results = results
        summary = self._compute_summary(results)

        if verbose:
            logger.info(f"Evaluation complete. Summary: {json.dumps(summary, ensure_ascii=False, indent=2)}")

        return summary

    def _evaluate_one(self, index: int, pair: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a single QA pair."""
        question = pair["question"]
        expected = pair["expected_answer"]
        relevant_ctx = pair.get("relevant_contexts", [])
        expected_kw = [ctx.split("#")[0] for ctx in relevant_ctx if "#" in ctx]

        # Get answer from QA engine
        try:
            qa_result = self.qa_engine(question)
            answer = qa_result.get("answer", "无回答")
            sources = qa_result.get("sources", [])
            source_texts = [s.get("content", "") for s in sources]
        except Exception as e:
            logger.error(f"QA engine failed for question {index}: {e}")
            answer = "ERROR: 引擎调用失败"
            source_texts = []
            sources = []

        # Compute metrics
        context_combined = "\n".join(source_texts)
        metrics = compute_all_metrics(
            question=question,
            context=context_combined,
            answer=answer,
            contexts=source_texts,
            expected_keywords=expected_kw if expected_kw else None,
        )

        # Context recall
        recall_checker = ContextRecallChecker()
        recall_result = recall_checker.evaluate(relevant_ctx, [s.get("source", "") for s in sources])
        metrics["context_recall"] = recall_result.score

        return {
            "index": index,
            "question": question,
            "expected_answer": expected,
            "actual_answer": answer,
            "metrics": metrics,
            "sources_count": len(sources),
            "device_code": pair.get("device_code", ""),
            "category": pair.get("category", ""),
        }

    @staticmethod
    def _compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate metrics from all results."""
        if not results:
            return {"error": "no_results"}

        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        averages = {}
        for metric in metric_names:
            scores = [r["metrics"].get(metric, 0) for r in results]
            averages[metric] = round(sum(scores) / len(scores), 4) if scores else 0

        return {
            "total_questions": len(results),
            "averages": averages,
            "timestamp": datetime.now().isoformat(),
            "per_question": [
                {
                    "question": r["question"],
                    "category": r.get("category", ""),
                    "metrics": r["metrics"],
                    "sources_count": r["sources_count"],
                }
                for r in results
            ],
        }

    def save_report(self, output_dir: str | None = None) -> str:
        """Save evaluation report to JSON file.

        Returns:
            Path to the saved report
        """
        output_dir = output_dir or settings.eval_report_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(output_dir) / f"eval_report_{timestamp}.json"

        summary = self._compute_summary(self.results)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Report saved to {report_path}")
        return str(report_path)


def create_qa_engine_wrapper(pipeline) -> Callable[[str], dict[str, Any]]:
    """Wrap a RAG pipeline as a QA engine callable for the evaluator."""
    def engine(question: str) -> dict[str, Any]:
        return pipeline.invoke(question)
    return engine


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Medical AI Assistant Evaluation Runner")
    parser.add_argument("--golden-set", default=None, help="Path to golden QA dataset (JSONL)")
    parser.add_argument("--output-dir", default=None, help="Output directory for reports")
    parser.add_argument("--no-verbose", action="store_true", help="Suppress console output")
    parser.add_argument("--mode", choices=["placeholder", "rag"], default="rag",
                        help="Evaluation mode: 'placeholder' for quick test, 'rag' for full pipeline")
    args = parser.parse_args()

    if args.mode == "rag":
        # Full RAG pipeline evaluation
        from src.ai.rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        qa_engine = create_qa_engine_wrapper(pipeline)
    else:
        # Placeholder for quick testing without model
        def qa_engine(question: str) -> dict[str, Any]:
            return {"answer": f"占位回复: {question}", "sources": []}

    runner = EvalRunner(
        qa_engine=qa_engine,
        golden_set=args.golden_set,
    )

    summary = runner.run(verbose=not args.no_verbose)
    report_path = runner.save_report(args.output_dir)
    print(f"\nReport saved to: {report_path}")
    print(f"Averages: {json.dumps(summary.get('averages', {}), ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
