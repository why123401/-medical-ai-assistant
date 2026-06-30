"""Conversation CRUD routes.

Endpoints:
  POST   /api/conversations          — create a new conversation
  GET    /api/conversations          — list conversations (paginated)
  GET    /api/conversations/{id}     — get conversation detail
  DELETE /api/conversations/{id}     — delete conversation + messages
  POST   /api/conversations/{id}/messages — send a message (chat)
"""

import json
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.schemas import (
    ConversationCreate,
    ConversationResponse,
    ConversationListResponse,
    ConversationMessagesResponse,
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
)
from src.infra.db import get_db
from src.infra.models import Conversation, Message
from src.shared.logging import get_logger

logger = get_logger("api.conversations")

# Module-level agent singleton — imported once, reused for all requests
_agent_singleton = None


def _get_agent():
    """Lazy-load the agent singleton."""
    global _agent_singleton
    if _agent_singleton is None:
        from src.ai.agents.medical_agent import MedicalAgent
        _agent_singleton = MedicalAgent()
    return _agent_singleton


router = APIRouter()


def _to_response(conv: Conversation) -> ConversationResponse:
    msg_count = len(conv.messages) if hasattr(conv, "messages") else 0
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at.isoformat() if conv.created_at else "",
        updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
        message_count=msg_count,
    )


# --- Create conversation ---

@router.post("/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    body: ConversationCreate,
    db: Session = Depends(get_db),
):
    conv = Conversation(id=str(uuid.uuid4()), title=body.title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    logger.info(f"Created conversation {conv.id}")
    return _to_response(conv)


# --- List conversations ---

@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
):
    offset = (page - 1) * page_size
    total = db.query(Conversation).count()
    rows = (
        db.query(Conversation)
        .order_by(Conversation.updated_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return ConversationListResponse(
        conversations=[_to_response(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


# --- Get conversation detail ---

@router.get("/conversations/{conv_id}", response_model=ConversationResponse)
async def get_conversation(
    conv_id: str,
    db: Session = Depends(get_db),
):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return _to_response(conv)


# --- Get conversation message history ---

@router.get("/conversations/{conv_id}/messages", response_model=ConversationMessagesResponse)
async def get_conversation_messages(
    conv_id: str,
    db: Session = Depends(get_db),
):
    """Return the full message history for a conversation."""
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conv_id)
        .order_by(Message.created_at)
        .all()
    )

    return ConversationMessagesResponse(
        id=conv.id,
        title=conv.title,
        messages=[
            ChatMessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                sources=m.sources or "",
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m in msgs
        ],
        created_at=conv.created_at.isoformat() if conv.created_at else "",
        updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
    )


# --- Delete conversation ---

@router.delete("/conversations/{conv_id}", status_code=204)
async def delete_conversation(
    conv_id: str,
    db: Session = Depends(get_db),
):
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.query(Message).filter(Message.conversation_id == conv_id).delete()
    db.delete(conv)
    db.commit()
    logger.info(f"Deleted conversation {conv_id}")


# --- Send message (calls RAG pipeline) ---

@router.post("/conversations/{conv_id}/messages", response_model=ChatResponse)
async def send_message(
    conv_id: str,
    body: ChatRequest,
    db: Session = Depends(get_db),
):
    """Send a user message, get AI reply via RAG pipeline."""
    conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Auto-generate conversation title from the first user message
    if conv.title == "新对话" and body.message.strip():
        stripped = body.message.strip()
        # Extract up to 12 Chinese/CJK characters for the title
        cn_chars = re.findall(r'[一-鿿㐀-䶿]', stripped)
        if cn_chars:
            conv.title = ''.join(cn_chars[:12])
        else:
            conv.title = stripped[:24]
        db.commit()

    # Store user message
    user_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv_id,
        role="user",
        content=body.message,
    )
    db.add(user_msg)
    db.commit()

    # Invoke agent (intent-based tool selection)
    try:
        agent = _get_agent()
        result = agent.invoke(body.message, conversation_id=conv_id)
        reply = result["reply"]
        sources = result.get("sources", [])
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        reply = "抱歉，AI 服务暂时不可用，请稍后再试。"
        sources = []

    # Store assistant reply
    assistant_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conv_id,
        role="assistant",
        content=reply,
        sources=json.dumps(sources, ensure_ascii=False),
    )
    db.add(assistant_msg)
    db.commit()

    return ChatResponse(
        conversation_id=conv_id,
        reply=reply,
        sources=sources,
    )
