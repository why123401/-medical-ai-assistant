"""Pydantic request/response schemas for the medical AI assistant API."""

from pydantic import BaseModel, Field


# --- Conversation schemas ---

class ConversationCreate(BaseModel):
    """Request to create a new conversation."""
    title: str = Field(default="新对话", description="Conversation title")


class ChatMessageResponse(BaseModel):
    """A single message in the conversation history."""
    id: str
    role: str
    content: str
    sources: str = ""
    created_at: str = ""


class ConversationMessagesResponse(BaseModel):
    """Full message history for a conversation."""
    id: str
    title: str
    messages: list[ChatMessageResponse]
    created_at: str
    updated_at: str


class ConversationResponse(BaseModel):
    """Response with conversation metadata."""
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class ChatRequest(BaseModel):
    """Incoming chat request from the client."""
    conversation_id: str | None = Field(default=None, description="Optional conversation ID")
    message: str = Field(min_length=1, description="User message text")


class ChatResponse(BaseModel):
    """Streaming/chat response."""
    conversation_id: str
    reply: str
    sources: list[dict] = Field(default_factory=list, description="Cited source chunks")


class ConversationListResponse(BaseModel):
    """Paginated conversation list."""
    conversations: list[ConversationResponse]
    total: int
    page: int
    page_size: int


# --- Knowledge base schemas ---

class KBUploadResponse(BaseModel):
    """Response after uploading a document."""
    filename: str
    status: str
    chunks_created: int
    message: str


class KBListResponse(BaseModel):
    """List of documents in the knowledge base."""
    documents: list[dict]
    total: int


class KBDeleteRequest(BaseModel):
    """Request to delete a document."""
    filename: str


# --- Health check ---

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    models_available: bool = True
