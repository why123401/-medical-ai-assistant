"""Document ingestion pipeline for the knowledge base.

Handles: load → chunk → embed → store, with MD5 dedup.

Pipeline:
    Upload file → MD5 check → Text extraction → Semantic chunking
    → Embedding → ChromaDB storage → MD5 record

This is the write path; the read path is handled by src/ai/rag/retriever.py.
"""

import hashlib
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.kb.chunking import create_splitter
from src.shared.config import settings
from src.shared.logging import get_logger

logger = get_logger("kb.indexer")


def _get_embedding_function():
    """Load the embedding function, trying multiple backends."""
    # Try DashScope embedding API
    try:
        from langchain_community.embeddings import DashScopeEmbeddings

        return DashScopeEmbeddings(
            model_name=settings.embedding_model,
            dashscope_api_key=settings.dashscope_api_key,
        )
    except Exception as e:
        logger.debug(f"DashScope embeddings unavailable: {e}")

    # Try local embedding model
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    except Exception as e:
        logger.debug(f"HuggingFace embeddings unavailable: {e}")

    logger.warning("No embedding function available — chunks will be stored without vectors")
    return None


def _get_vector_store(embedding_fn=None) -> Chroma:
    """Get or create the ChromaDB vector store."""
    return Chroma(
        collection_name="medical_devices",
        persist_directory=settings.vector_store_full_path,
        embedding_function=embedding_fn,
    )


def _compute_md5(file_path: str) -> str:
    """Compute MD5 hash of a file for deduplication."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _check_md5(md5_hash: str) -> bool:
    """Check if this file has already been ingested."""
    md5_path = settings.md5_full_path
    if not os.path.exists(md5_path):
        return False
    with open(md5_path, "r", encoding="utf-8") as f:
        return md5_hash in (line.strip() for line in f)


def _save_md5(md5_hash: str) -> None:
    """Record an MD5 hash as ingested."""
    with open(settings.md5_full_path, "a", encoding="utf-8") as f:
        f.write(md5_hash + "\n")


def ingest_document(file_path: str, md5_hash: str | None = None) -> int:
    """Ingest a single document into the vector store.

    This is the main entry point called from the API upload endpoint.

    Args:
        file_path: Absolute path to the document file
        md5_hash: Pre-computed MD5 (computed automatically if None)

    Returns:
        Number of chunks created and stored
    """
    if md5_hash is None:
        md5_hash = _compute_md5(file_path)

    if _check_md5(md5_hash):
        logger.info(f"[ingest] Skipping already-ingested file: {file_path}")
        return 0

    # Load text content
    ext = Path(file_path).suffix.lower()
    docs = _load_file(file_path, ext)

    if not docs:
        logger.warning(f"[ingest] No content extracted from: {file_path}")
        return 0

    # Chunk using semantic-aware splitter
    splitter = create_splitter()
    chunks = splitter.split_documents(docs)

    if not chunks:
        logger.warning(f"[ingest] No chunks produced from: {file_path}")
        return 0

    # Enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_index": i,
            "total_chunks": len(chunks),
            "original_file": file_path,
        })

    # Store in vector DB
    embedding_fn = _get_embedding_function()
    store = _get_vector_store(embedding_fn)
    store.add_documents(chunks)

    _save_md5(md5_hash)
    logger.info(f"[ingest] {file_path} → {len(chunks)} chunks stored")
    return len(chunks)


def ingest_batch(file_paths: list[str]) -> dict[str, int]:
    """Ingest multiple documents, returning per-file chunk counts.

    Args:
        file_paths: List of absolute file paths

    Returns:
        Dict mapping filename → chunk count
    """
    results = {}
    for path in file_paths:
        count = ingest_document(path)
        results[Path(path).name] = count
    return results


def _load_file(file_path: str, ext: str) -> list[Document]:
    """Extract text content from a file based on its extension."""
    loaders = {
        ".txt": _load_txt,
        ".pdf": _load_pdf,
    }
    loader = loaders.get(ext)
    if not loader:
        logger.warning(f"[ingest] Unsupported file type: {ext}")
        return []
    return loader(file_path)


def _load_txt(file_path: str) -> list[Document]:
    """Load plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(
        page_content=content,
        metadata={"source": file_path, "type": "txt"},
    )]


def _load_pdf(file_path: str) -> list[Document]:
    """Load PDF file using PyMuPDF (preferred) or pdfminer.six (fallback)."""
    # Try PyMuPDF first (faster, supports images)
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None

    if fitz:
        docs = []
        with fitz.open(file_path) as pdf:
            for page_num, page in enumerate(pdf):
                text = page.get_text()
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "type": "pdf",
                            "page": page_num + 1,
                        },
                    ))
        return docs

    # Fallback: pdfminer
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(file_path)
        return [Document(
            page_content=text,
            metadata={"source": file_path, "type": "pdf"},
        )]
    except ImportError:
        logger.warning(
            "[ingest] No PDF reader available. "
            "Install PyMuPDF (pip install pymupdf) or pdfminer.six"
        )
        return []


def remove_document(md5_hash: str) -> bool:
    """Remove a document's chunks from the vector store.

    Note: ChromaDB doesn't support deleting by metadata filter in all versions.
    This is a placeholder for future implementation.

    Args:
        md5_hash: MD5 of the document to remove

    Returns:
        True if removal was attempted
    """
    logger.warning(f"[ingest] Document removal not yet implemented for MD5: {md5_hash}")
    # TODO: Implement deletion using ChromaDB's delete_documents API
    return False
