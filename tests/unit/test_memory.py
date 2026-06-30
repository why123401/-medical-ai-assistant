"""Unit tests for memory/manager — conversation memory management."""

import pytest

from src.memory.manager import ConversationMemory, MemoryManager


class TestConversationMemory:
    """Test single conversation memory."""

    def test_add_message(self):
        mem = ConversationMemory("test-1")
        mem.add_message("user", "你好")
        mem.add_message("assistant", "您好")
        assert len(mem.messages) == 2
        assert mem.turn_count == 2

    def test_get_context_returns_recent(self):
        mem = ConversationMemory("test-2")
        # Add 20 messages
        for i in range(20):
            mem.add_message("user", f"msg {i}")
            mem.add_message("assistant", f"reply {i}")

        context = mem.get_context_messages()
        # Should return last 8 turns (window_size default)
        assert len(context) <= 16  # 8 turns * 2 roles

    def test_clear_resets_state(self):
        mem = ConversationMemory("test-3")
        mem.add_message("user", "hello")
        mem.clear()
        assert len(mem.messages) == 0
        assert mem.turn_count == 0
        assert mem.summary is None


class TestMemoryManager:
    """Test multi-conversation memory management."""

    def test_create_new_memory(self):
        mgr = MemoryManager()
        mem = mgr.get_or_create("conv-1")
        assert mem.conversation_id == "conv-1"

    def test_reuse_existing_memory(self):
        mgr = MemoryManager()
        mem1 = mgr.get_or_create("conv-1")
        mem2 = mgr.get_or_create("conv-1")
        assert mem1 is mem2  # Same instance

    def test_auto_generate_id(self):
        mgr = MemoryManager()
        mem = mgr.get_or_create()
        assert mem.conversation_id is not None
        assert len(mem.conversation_id) > 0

    def test_delete_memory(self):
        mgr = MemoryManager()
        mgr.get_or_create("conv-1")
        assert mgr.get("conv-1") is not None
        assert mgr.delete("conv-1") is True
        assert mgr.get("conv-1") is None
        assert mgr.delete("nonexistent") is False

    def test_list_conversations(self):
        mgr = MemoryManager()
        mgr.get_or_create("conv-1").add_message("user", "hi")
        mgr.get_or_create("conv-2").add_message("user", "hello")
        mgr.get_or_create("conv-3")

        conversations = mgr.list_conversations()
        assert len(conversations) == 3
        assert all("conversation_id" in c for c in conversations)
        assert all("message_count" in c for c in conversations)
