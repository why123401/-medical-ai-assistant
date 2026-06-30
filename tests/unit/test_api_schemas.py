"""Unit tests for API schemas — Pydantic validation."""

import pytest
from pydantic import ValidationError

from src.api.schemas import ChatRequest, ChatResponse, HealthResponse


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_chat_request_valid(self):
        req = ChatRequest(message="你好")
        assert req.message == "你好"
        assert req.conversation_id is None

    def test_chat_request_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_chat_request_none_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message=None)

    def test_chat_response_with_sources(self):
        resp = ChatResponse(
            conversation_id="abc-123",
            reply="这是回复",
            sources=[{"index": 1, "source": "test.txt", "content": "context"}],
        )
        assert resp.conversation_id == "abc-123"
        assert len(resp.sources) == 1

    def test_health_response_defaults(self):
        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.version == "1.0.0"
