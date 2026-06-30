"""Hybrid retriever combining BM25 keyword search with dense vector retrieval.

Uses Reciprocal Rank Fusion (RRF) to combine results from both retrieval methods.
RRF is parameter-free (k=60), making it robust and easy to tune.

Architecture:
    Query ──→ [BM25] ──┐
                        ├──→ RRF Fusion ──→ Top-20 ──→ Reranker ──→ Top-5
    Query ──→ [Vector] ─┘

Why this works for medical devices:
    - BM25 excels at exact-match terms (device codes like MED-VENT-X200,
      fault codes like E104) that have no semantic meaning in embedding space.
    - Vector retrieval captures semantic similarity ("呼吸机坏了怎么办" ≈
      "呼吸机故障处理流程") that BM25 misses due to vocabulary mismatch.
    - RRF combines both without requiring weight tuning.
"""

from collections import defaultdict
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.ai.rag.reranker import Reranker
from src.shared.config import settings
from src.shared.logging import get_logger

logger = get_logger("ai.rag.retriever")


class HybridRetriever:
    """Combines BM25 + vector retrieval with RRF fusion.

    Two construction patterns:
        1. Automatic: HybridRetriever() — loads from ChromaDB on first query
        2. Pre-configured: HybridRetriever(bm25=..., vector=...)

    Usage:
        retriever = HybridRetriever()
        docs = retriever.invoke("呼吸机 MED-VENT-X200 报警 E104")
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        vector_retriever: BaseRetriever | None = None,
        embedding_fn: Any | None = None,
        k: int = settings.retrieval_top_k,
    ):
        self.k = k
        self._bm25 = bm25_retriever
        self._vector = vector_retriever
        self._embedding_fn = embedding_fn
        self._reranker = Reranker()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-initialize both retrievers from ChromaDB if not already set."""
        if self._initialized:
            return

        if self._bm25 is None or self._vector is None:
            try:
                from langchain_chroma import Chroma

                vector_store = Chroma(
                    collection_name="medical_devices",
                    persist_directory=settings.vector_store_full_path,
                    embedding_function=self._embedding_fn,
                )

                if self._vector is None:
                    self._vector = vector_store.as_retriever(
                        search_kwargs={"k": self.k * 2},
                    )

                if self._bm25 is None:
                    collection = vector_store.get(include=["documents", "metadatas"])
                    docs_data = collection.get("documents", [])
                    metas_data = collection.get("metadatas", [])

                    if docs_data:
                        langchain_docs = [
                            Document(
                                page_content=str(d),
                                metadata=m or {},
                            )
                            for d, m in zip(docs_data, metas_data)
                        ]
                        self._bm25 = BM25Retriever.from_documents(langchain_docs)
                        self._bm25.k = self.k * 2
                        logger.info(f"BM25 index built from {len(langchain_docs)} documents")
                    else:
                        logger.warning("ChromaDB collection is empty — BM25 index skipped")
            except Exception as e:
                logger.warning(f"Failed to auto-initialize retrievers: {e}")

        self._initialized = True

    @classmethod
    def from_vector_store(cls, vector_store: Any, **kwargs) -> "HybridRetriever":
        """Create hybrid retriever from an existing Chroma vector store.

        Args:
            vector_store: LangChain-compatible vector store (Chroma, etc.)
            **kwargs: Extra args passed to constructor

        Returns:
            Configured HybridRetriever instance.
        """
        vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": kwargs.get("k", settings.retrieval_top_k) * 2},
        )

        collection = vector_store.get(include=["documents", "metadatas"])
        docs_data = collection.get("documents", [])
        metas_data = collection.get("metadatas", [])

        if not docs_data:
            logger.warning("No documents available for BM25 index")
            return cls(vector_retriever=vector_retriever, **kwargs)

        langchain_docs = [
            Document(page_content=str(d), metadata=m or {})
            for d, m in zip(docs_data, metas_data)
        ]
        bm25 = BM25Retriever.from_documents(langchain_docs)
        bm25.k = kwargs.get("k", settings.retrieval_top_k) * 2

        return cls(bm25_retriever=bm25, vector_retriever=vector_retriever, **kwargs)

    def invoke(self, query: str) -> list[Document]:
        """Execute hybrid retrieval with RRF fusion and optional reranking.

        Flow:
          1. Ensure both retrievers are initialized (lazy load from ChromaDB)
          2. BM25 retrieves top-K keyword matches
          3. Vector retrieves top-K semantic matches
          4. RRF fusion ranks combined results
          5. Reranker reorders top candidates with cross-attention
        """
        self._ensure_initialized()

        # Step 1: Get results from both retrievers
        bm25_docs = self._bm25.invoke(query) if self._bm25 else []
        vector_docs = self._vector.invoke(query) if self._vector else []

        if not bm25_docs and not vector_docs:
            return []

        # Step 2: RRF fusion
        fused = self._rrf_fusion(bm25_docs, vector_docs, k=60)

        # Step 3: Take top-K*2 before reranking (give reranker more candidates)
        pre_rerank = fused[: self.k * 2]

        # Step 4: Rerank to final top-N
        final = self._reranker.rerank(query, pre_rerank, top_n=settings.rerank_top_n)

        logger.debug(
            f"Hybrid retrieval: {len(bm25_docs)} BM25 + {len(vector_docs)} vector "
            f"→ {len(fused)} fused → {len(final)} after rerank"
        )
        return final

    @staticmethod
    def _rrf_fusion(
        bm25_docs: list[Document],
        vector_docs: list[Document],
        k: int = 60,
    ) -> list[Document]:
        """Reciprocal Rank Fusion to combine two ranked lists.

        Documents appearing in both lists get boosted scores,
        reflecting higher confidence in relevance.

        Formula: score(doc) = Σ 1 / (k + rank_i)
        where k=60 is the standard constant (proven effective in practice).
        """
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for idx, doc in enumerate(bm25_docs):
            doc_id = _make_doc_id("bm25", idx, doc)
            scores[doc_id] += 1.0 / (k + idx)
            doc_map[doc_id] = doc

        for idx, doc in enumerate(vector_docs):
            doc_id = _make_doc_id("vec", idx, doc)
            scores[doc_id] += 1.0 / (k + idx)
            doc_map[doc_id] = doc

        # Sort by fused score descending
        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[rid] for rid in ranked_ids if rid in doc_map]


def _make_doc_id(prefix: str, idx: int, doc: Document) -> str:
    """Generate a stable document identifier for RRF fusion.

    Uses metadata doc_id if available, otherwise falls back to
    content hash for deduplication across retrieval sources.
    """
    meta_id = doc.metadata.get("doc_id")
    if meta_id:
        return f"{prefix}_{meta_id}"
    # Hash content for dedup (fast enough for typical chunk sizes)
    import hashlib

    content_hash = hashlib.md5(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{content_hash}"
