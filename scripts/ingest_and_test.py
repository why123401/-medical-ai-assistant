#!/usr/bin/env python
"""Quick test script: ingest knowledge docs → run RAG pipeline → print results.

Usage:
    python scripts/ingest_and_test.py

This script:
    1. Ingests all .txt files from data/knowledge/
    2. Runs a sample query through the RAG pipeline
    3. Prints the answer and sources
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kb.indexer import ingest_document, ingest_batch
from src.ai.rag.pipeline import RAGPipeline
from src.shared.logging import get_logger

logger = get_logger("scripts.ingest_test")


def main():
    knowledge_dir = Path(__file__).parent.parent / "data" / "knowledge"

    # Step 1: Ingest all documents
    print("=" * 60)
    print("Step 1: Ingesting knowledge documents")
    print("=" * 60)

    txt_files = list(knowledge_dir.rglob("*.txt"))
    if not txt_files:
        print("No .txt files found in data/knowledge/")
        print("Please add medical device documentation first.")
        return

    for f in txt_files:
        print(f"  → {f.relative_to(knowledge_dir)}")

    results = ingest_batch([str(f) for f in txt_files])
    total_chunks = sum(results.values())
    print(f"\n  Total: {len(results)} files, {total_chunks} chunks ingested")

    # Step 2: Run sample queries
    print("\n" + "=" * 60)
    print("Step 2: Running RAG queries")
    print("=" * 60)

    pipeline = RAGPipeline()

    queries = [
        "呼吸机 MED-VENT-X200 报警 E104 怎么处理？",
        "CT 扫描仪探测器温度异常 F201 什么原因？",
        "MED-CT-3200 的辐射剂量特点是什么？",
        "高压灭菌器 MED-INF-500 的灭菌温度是多少？",
    ]

    for query in queries:
        print(f"\nQ: {query}")
        print("-" * 40)
        try:
            result = pipeline.invoke(query)
            print(f"A: {result['answer'][:200]}...")
            print(f"Sources: {len(result['sources'])} chunks retrieved")
            for s in result["sources"][:2]:
                print(f"  [{s['index']}] {s['source'][:50]}")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
