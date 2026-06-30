"""Cross-encoder reranker for medical device RAG.

Supports three reranking strategies, tried in order:
  1. DashScope text-rerank-v2 API (production, highest quality)
  2. Local BGE-reranker (offline, no API key needed)
  3. Keyword overlap fallback (graceful degradation)

Why reranking matters:
    RRF fusion gives us Top-20 candidates, but they're still roughly ordered.
    A Cross-Encoder can examine each query-chunk pair individually with
    full attention, catching subtle relevance signals that Bi-Encoders miss.
    This typically improves NDCG@5 by 10-20 percentage points.
"""

import re
from typing import Any

from langchain_core.documents import Document

from src.shared.config import settings
from src.shared.logging import get_logger

logger = get_logger("ai.rag.reranker")


class Reranker:
    """Rerank retrieved documents using a cross-encoder model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.rerank_model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """Rerank documents for the given query.

        Tries DashScope API first, falls back to local model,
        then to keyword overlap scoring.

        Args:
            query: User question
            documents: Candidate documents from hybrid retrieval
            top_n: Number of documents to return

        Returns:
            Re-ranked document list (top_n most relevant first)
        """
        if not documents:
            return []

        top_n = top_n or settings.rerank_top_n

        # Try DashScope API
        try:
            return self._dashscope_rerank(query, documents, top_n)
        except ImportError as e:
            logger.debug(f"DashScope not available ({e}), trying local reranker")
        except Exception as e:
            logger.warning(f"DashScope reranker failed ({e}), falling back")

        # Try local BGE-reranker
        try:
            return self._local_rerank(query, documents, top_n)
        except Exception as e:
            logger.debug(f"Local reranker failed ({e}), falling back to keyword overlap")

        # Final fallback: keyword overlap
        return self._keyword_overlap(query, documents, top_n)

    def _dashscope_rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> list[Document]:
        """Use DashScope text-rerank-v2 API for cross-encoder reranking."""
        from dashscope import TextRerank

        result = TextRerank.run(
            model=self.model_name,
            query=query,
            documents=[doc.page_content for doc in documents],
            top_n=top_n,
            return_documents=True,
        )

        if result.status_code == 200 and result.output.get("results"):
            reranked = []
            for item in result.output["results"][:top_n]:
                doc_text = item["document"].get("text", "")
                # Find matching document by content identity
                for doc in documents:
                    if doc.page_content == doc_text:
                        doc.metadata["rerank_score"] = item.get("relevance_score", 0)
                        reranked.append(doc)
                        break
            return reranked

        return documents[:top_n]

    def _local_rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> list[Document]:
        """Use a local Cross-Encoder model (BGE-reranker) for reranking."""
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            raise ImportError(
                "Install FlagEmbedding for local reranking: pip install FlagEmbedding"
            )

        # FlagReranker expects pairs: [(query, doc1), (query, doc2), ...]
        pairs = [(query, doc.page_content) for doc in documents]
        scorer = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        scores = scorer.compute_score(pairs, task_type="rerank")

        # Pair scores with documents and sort
        scored = list(zip(scores, documents))
        scored.sort(key=lambda x: x[0], reverse=True)

        for i, (score, doc) in enumerate(scored[:top_n]):
            doc.metadata["rerank_score"] = float(score)

        return [doc for _, doc in scored[:top_n]]

    @staticmethod
    def _keyword_overlap(
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> list[Document]:
        """Fallback: simple keyword overlap scoring.

        Counts shared Chinese characters and alphanumeric tokens between
        query and document. Works without any model dependency.
        """
        # Extract meaningful tokens (Chinese chars + alphanumeric words)
        query_tokens = set(re.findall(r"[一-鿿]+|\w+", query.lower()))
        scored = []
        for doc in documents:
            doc_tokens = set(re.findall(r"[一-鿿]+|\w+", doc.page_content.lower()))
            overlap = len(query_tokens & doc_tokens)
            # Normalize by query length to penalize documents with low coverage
            normalized = overlap / max(len(query_tokens), 1)
            scored.append((normalized, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]


def create_reranker(model_name: str | None = None) -> Reranker:
    """Factory function to create a reranker instance."""
    return Reranker(model_name=model_name)
