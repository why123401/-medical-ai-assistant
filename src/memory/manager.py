"""Conversation memory manager.

Handles session state, message history, and summarization triggers.
Keeps the last N turns verbatim and summarizes older messages.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from src.shared.config import settings
from src.shared.logging import get_logger

logger = get_logger("memory.manager")


class ConversationMemory:
    """Manages conversation state for a single session."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.messages: list[dict[str, str]] = []  # [{"role": "user/assistant", "content": "..."}]
        self.summary: str | None = None
        self.turn_count = 0

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
        self.turn_count += 1

        # Trigger summarization
        if self.turn_count >= settings.summary_trigger_turns:
            self._trigger_summarization()

    def get_context_messages(self) -> list[dict[str, str]]:
        """Get the last N turns for the conversation context.

        Returns summarized older messages + verbatim recent ones.
        """
        window = settings.conversation_window_size
        recent = self.messages[-window:] if len(self.messages) > window else self.messages[:]

        if self.summary and recent != self.messages:
            return [{"role": "system", "content": f"对话摘要: {self.summary}"}] + recent

        return recent

    def get_full_history(self) -> list[dict[str, str]]:
        """Return the complete message history."""
        return self.messages.copy()

    def _trigger_summarization(self) -> None:
        """Mark that summarization is needed for older messages.

        Actual summarization uses the LLM — implemented in the agent layer.
        """
        logger.info(f"Summarization triggered for conversation {self.conversation_id} at turn {self.turn_count}")
        self.summary = "[待摘要]"  # Placeholder — actual summary requires LLM call

    def clear(self) -> None:
        """Reset the conversation memory."""
        self.messages.clear()
        self.summary = None
        self.turn_count = 0


class MemoryManager:
    """Manages multiple conversation memories.

    Loads historical messages from the database on first access per
    conversation_id so that restarts don't lose context.
    """

    def __init__(self):
        self._stores: dict[str, ConversationMemory] = {}

    def _load_history_from_db(self, conversation_id: str) -> list[dict[str, str]]:
        """Load message history from SQLite for a given conversation."""
        try:
            from src.infra.db import SessionLocal
            from src.infra.models import Message

            session = SessionLocal()
            msgs = (
                session.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
                .all()
            )
            history = [{"role": m.role, "content": m.content} for m in msgs]
            session.close()
            return history
        except Exception as e:
            logger.warning(f"Failed to load history for {conversation_id}: {e}")
            return []

    def get_or_create(self, conversation_id: str | None = None) -> ConversationMemory:
        """Get existing or create new conversation memory."""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        if conversation_id not in self._stores:
            mem = ConversationMemory(conversation_id)
            # Restore persisted history from DB
            history = self._load_history_from_db(conversation_id)
            if history:
                for msg in history:
                    mem.messages.append(msg)
                    mem.turn_count += 1
            self._stores[conversation_id] = mem

        return self._stores[conversation_id]

    def get(self, conversation_id: str) -> ConversationMemory | None:
        return self._stores.get(conversation_id)

    def delete(self, conversation_id: str) -> bool:
        if conversation_id in self._stores:
            del self._stores[conversation_id]
            return True
        return False

    def list_conversations(self) -> list[dict[str, Any]]:
        """List all active conversations with metadata."""
        return [
            {
                "conversation_id": cm.conversation_id,
                "message_count": len(cm.messages),
                "turn_count": cm.turn_count,
                "has_summary": cm.summary is not None,
            }
            for cm in self._stores.values()
        ]


# Module-level singleton
memory_manager = MemoryManager()
