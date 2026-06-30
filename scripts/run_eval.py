#!/usr/bin/env python
"""Convenience wrapper for running evaluation.

Usage:
    python scripts/run_eval.py                  # Full RAG evaluation
    python scripts/run_eval.py --mode placeholder  # Quick test without model
    python scripts/run_eval.py --no-verbose      # Silent mode
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.runner import main

if __name__ == "__main__":
    main()
