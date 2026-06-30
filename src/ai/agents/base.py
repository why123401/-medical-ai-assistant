"""Abstract Agent interface for the medical AI assistant.

Defines a unified contract that all agent implementations must follow.
This allows swapping between ReAct agent, pipeline agent, or future
implementations without changing the API layer.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class for AI agents."""

    @abstractmethod
    def invoke(self, query: str, conversation_id: str | None = None) -> dict[str, Any]:
        """Execute a single-turn agent interaction.

        Args:
            query: User question
            conversation_id: Optional conversation session ID

        Returns:
            Dict with at least 'reply' and 'sources' keys
        """

    @abstractmethod
    def stream(self, query: str, conversation_id: str | None = None):
        """Execute a streaming interaction (generator).

        Yields chunks of the response as they are generated.
        """

    @abstractmethod
    def get_tools(self) -> list[Any]:
        """Return the list of tools available to this agent."""
