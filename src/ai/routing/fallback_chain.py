"""Fallback chain builder for model routing.

Constructs the ordered list of models to try when the primary model fails,
integrating with the circuit breaker in router.py.
"""

from src.shared.config import settings
from src.shared.logging import get_logger

logger = get_logger("routing.fallback")

# Default fallback priority chain
DEFAULT_CHAIN = [
    settings.primary_model,
    settings.fallback_model_strong,
    settings.fallback_model_weak,
]


def build_fallback_chain(
    override_model: str | None = None,
) -> list[str]:
    """Build the model fallback chain.

    Args:
        override_model: If specified, return only that model (for testing).

    Returns:
        Ordered list of model names to try.
    """
    if override_model:
        logger.info(f"Using override model: {override_model}")
        return [override_model]

    chain = []
    for name in DEFAULT_CHAIN:
        if name:
            chain.append(name)

    logger.info(f"Fallback chain: {' → '.join(chain)}")
    return chain


def is_model_in_chain(model_name: str) -> bool:
    """Check if a model is registered in the fallback chain."""
    return model_name in DEFAULT_CHAIN
