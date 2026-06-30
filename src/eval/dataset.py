"""Gold standard QA dataset for medical device evaluation.

Loads from JSONL file (data/eval/qa_golden_set.jsonl) by default.
Contains 200+ QA pairs covering:
  - Equipment specifications (61 pairs)
  - Fault diagnosis (54 pairs)
  - Maintenance schedules (26 pairs)
  - Comparisons (8 pairs)
  - Scenarios (51 pairs)

Devices covered:
  - MED-VENT-X200 (呼吸机) — 51 pairs
  - MED-CT-3200 (CT 扫描仪) — 46 pairs
  - MED-MRI-1.5T (磁共振) — 22 pairs
  - MED-MON-5000 (多参数监护仪) — 26 pairs
  - MED-US-PRO (超声诊断仪) — 21 pairs
  - MED-INF-500 (高压灭菌器) — 25 pairs
  - MULTIPLE (跨设备对比) — 9 pairs
"""

import json
from pathlib import Path
from typing import Any


def load_golden_set(path: str | None = None) -> list[dict[str, Any]]:
    """Load the golden QA dataset from JSONL file.

    Args:
        path: Path to JSONL file. Default: settings.eval_golden_set

    Returns:
        List of QA pair dicts
    """
    if path is None:
        from src.shared.config import settings
        path = settings.eval_golden_set

    return _load_jsonl(path)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load QA pairs from a JSONL file."""
    qa_pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_pairs.append(json.loads(line))
    return qa_pairs


def save_golden_set(qa_pairs: list[dict[str, Any]], path: str) -> None:
    """Save QA pairs to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def get_categories(qa_pairs: list[dict[str, Any]] | None = None) -> list[str]:
    """Get unique categories from QA pairs."""
    if qa_pairs is None:
        from src.shared.config import settings
        qa_pairs = load_golden_set(settings.eval_golden_set)
    return sorted(set(p.get("category", "unknown") for p in qa_pairs))


def get_devices(qa_pairs: list[dict[str, Any]] | None = None) -> list[str]:
    """Get unique devices from QA pairs."""
    if qa_pairs is None:
        from src.shared.config import settings
        qa_pairs = load_golden_set(settings.eval_golden_set)
    return sorted(set(p.get("device_code", "unknown") for p in qa_pairs))


def filter_by_category(qa_pairs: list[dict[str, Any]], category: str) -> list[dict]:
    """Filter QA pairs by category."""
    return [p for p in qa_pairs if p.get("category") == category]


def filter_by_device(qa_pairs: list[dict[str, Any]], device_code: str) -> list[dict]:
    """Filter QA pairs by device code."""
    return [p for p in qa_pairs if p.get("device_code") == device_code]
