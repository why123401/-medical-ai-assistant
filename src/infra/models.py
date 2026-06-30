"""SQLAlchemy ORM models for conversations and knowledge base metadata."""

from datetime import datetime, timezone

from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import relationship

from src.infra.db import Base


class Conversation(Base):
    """Represents a user conversation session."""

    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True)
    title = Column(String(256), nullable=False, default="新对话")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Message(Base):
    """Individual message within a conversation."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON string of cited chunks
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="messages")


Conversation.messages = relationship("Message", order_by=Message.created_at, back_populates="conversation")


class KnowledgeDoc(Base):
    """Metadata for ingested knowledge base documents."""

    __tablename__ = "knowledge_docs"

    id = Column(String(36), primary_key=True)
    filename = Column(String(512), nullable=False, unique=True, index=True)
    file_type = Column(String(16), nullable=False)  # txt / pdf
    md5_hash = Column(String(32), nullable=False, unique=True)
    chunks_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
