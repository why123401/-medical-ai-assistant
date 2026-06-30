"""Knowledge base upload and management routes.

Endpoints:
  GET    /api/kb/documents          — list all uploaded documents
  POST   /api/kb/upload             — upload a document
  DELETE /api/kb/documents/{filename} — delete a document
"""

import hashlib
import uuid
from pathlib import Path
from typing import BinaryIO

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.api.schemas import KBUploadResponse, KBListResponse, KBDeleteRequest
from src.infra.db import get_db
from src.infra.models import KnowledgeDoc
from src.kb.indexer import ingest_document  # plugs in Step 2 RAG
from src.shared.config import settings
from src.shared.logging import get_logger

logger = get_logger("api.kb")

router = APIRouter()


def _compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# --- Upload document ---

@router.post("/upload", response_model=KBUploadResponse)
async def upload_document(
    file: UploadFile,
    db: Session = Depends(get_db),
):
    allowed_types = {".txt", ".pdf"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_types}",
        )

    # Read file content
    content = await file.read()
    md5_hash = _compute_md5(content)

    # Check for duplicate
    existing = (
        db.query(KnowledgeDoc)
        .filter(KnowledgeDoc.md5_hash == md5_hash)
        .first()
    )
    if existing:
        return KBUploadResponse(
            filename=file.filename,
            status="skipped",
            chunks_created=existing.chunks_count,
            message="文件已存在于知识库中，跳过重复上传",
        )

    # Save to disk
    upload_dir = Path(settings.knowledge_full_path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as f:
        f.write(content)

    # Ingest into vector store (plugs in Step 2)
    chunks_count = 0
    try:
        chunks_count = ingest_document(str(file_path), md5_hash)
    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    # Persist metadata
    doc = KnowledgeDoc(
        id=str(uuid.uuid4()),
        filename=file.filename,
        file_type=ext.lstrip("."),
        md5_hash=md5_hash,
        chunks_count=chunks_count,
    )
    db.add(doc)
    db.commit()

    logger.info(f"Uploaded {file.filename} → {chunks_count} chunks")
    return KBUploadResponse(
        filename=file.filename,
        status="uploaded",
        chunks_created=chunks_count,
        message=f"成功上传并分片为 {chunks_count} 个 chunk",
    )


# --- List documents ---

@router.get("/documents", response_model=KBListResponse)
async def list_documents(db: Session = Depends(get_db)):
    docs = db.query(KnowledgeDoc).order_by(KnowledgeDoc.uploaded_at.desc()).all()
    return KBListResponse(
        documents=[
            {
                "filename": d.filename,
                "file_type": d.file_type,
                "chunks_count": d.chunks_count,
                "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else "",
            }
            for d in docs
        ],
        total=len(docs),
    )


# --- Delete document ---

@router.delete("/documents/{filename}", status_code=204)
async def delete_document(
    filename: str,
    db: Session = Depends(get_db),
):
    doc = db.query(KnowledgeDoc).filter(KnowledgeDoc.filename == filename).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from disk
    file_path = Path(settings.knowledge_full_path) / filename
    if file_path.exists():
        file_path.unlink()

    db.delete(doc)
    db.commit()
    logger.info(f"Deleted document {filename}")
