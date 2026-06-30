"""Multi-model routing engine with circuit breaker pattern.

Routes LLM requests to the best available model with automatic
fallback on failure. Integrates cost tracking and health monitoring.
"""

import time
from enum import Enum
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from src.shared.config import settings
from src.shared.logging import get_logger
from src.shared.exceptions import ModelError

logger = get_logger("routing")


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Too many failures, stop calling
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for model health monitoring.

    After `threshold` consecutive failures, opens the circuit.
    After `timeout` seconds, transitions to half-open to test recovery.
    """

    def __init__(
        self,
        threshold: int = settings.cb_failure_threshold,
        timeout: int = settings.cb_timeout_seconds,
        half_open_max: int = settings.cb_half_open_max,
    ):
        self.threshold = threshold
        self.timeout = timeout
        self.half_open_max = half_open_max

        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time: float = 0
        self.half_open_attempts = 0

    def record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED (recovery confirmed)")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker OPEN (half-open test failed)")
        elif self.failure_count >= self.threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPEN (failures: {self.failure_count}/{self.threshold})"
            )

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
                self.success_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False

        # HALF_OPEN
        self.half_open_attempts += 1
        return self.half_open_attempts <= self.half_open_max

    @property
    def is_available(self) -> bool:
        return self.allow_request()


class CostTracker:
    """Track per-request token usage and approximate cost."""

    # Approximate costs per 1M tokens (USD)
    COST_PER_1M = {
        "qwen-plus": {"input": 0.04, "output": 0.12},
        "qwen-max": {"input": 0.80, "output": 2.00},
        "qwen-turbo": {"input": 0.003, "output": 0.006},
    }

    def __init__(self):
        self.total_cost = 0.0
        self.request_count = 0
        self.model_usage: dict[str, int] = {}

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        costs = self.COST_PER_1M.get(model, {"input": 0.04, "output": 0.12})
        cost = (input_tokens / 1_000_000) * costs["input"] + \
               (output_tokens / 1_000_000) * costs["output"]
        self.total_cost += cost
        self.request_count += 1
        self.model_usage[model] = self.model_usage.get(model, 0) + 1
        return cost

    @property
    def avg_cost_per_request(self) -> float:
        return self.total_cost / max(self.request_count, 1)


class ModelRouter:
    """Central routing engine that selects and invokes the best model.

    Priority chain:
      1. primary (qwen-plus)
      2. fallback strong (qwen-max)
      3. fallback weak (qwen-turbo)

    Each model has its own circuit breaker for independent health tracking.
    """

    def __init__(self):
        self.cost_tracker = CostTracker()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._models: dict[str, BaseChatModel] = {}

    def register_model(self, name: str, model: BaseChatModel) -> None:
        """Register a chat model and its circuit breaker."""
        self._models[name] = model
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker()

    def get_breaker(self, model_name: str) -> CircuitBreaker:
        if model_name not in self._breakers:
            self._breakers[model_name] = CircuitBreaker()
        return self._breakers[model_name]

    def invoke(
        self,
        messages: list[Any],
        model_name: str | None = None,
    ) -> Any:
        """Invoke the model routing chain with automatic fallback.

        Args:
            messages: Chat messages to send
            model_name: Override the primary model selection

        Returns:
            Model response

        Raises:
            ModelError: If all models in the chain fail
        """
        chain = self._build_fallback_chain(model_name)

        for name, model, breaker in chain:
            if not breaker.is_available:
                logger.debug(f"Skipping {name}: circuit breaker open")
                continue

            try:
                logger.info(f"Routing to model: {name}")
                response = model.invoke(messages)

                # Record success
                breaker.record_success()

                # Track cost (best effort)
                if hasattr(response, "usage_metadata"):
                    meta = response.usage_metadata
                    cost = self.cost_tracker.record(
                        name,
                        meta.get("input_tokens", 0),
                        meta.get("output_tokens", 0),
                    )
                    logger.debug(f"Model {name} cost: ${cost:.6f}")

                return response

            except Exception as e:
                breaker.record_failure()
                logger.error(f"Model {name} failed: {e}")
                continue

        raise ModelError(
            f"All models in routing chain failed. Tried: {[m[0] for m in chain]}",
            code="ROUTING_ALL_FAILED",
        )

    def _build_fallback_chain(
        self,
        override_model: str | None = None,
    ) -> list[tuple[str, BaseChatModel, CircuitBreaker]]:
        """Build the model fallback chain ordered by priority."""
        if override_model:
            return [(override_model, self._models[override_model], self.get_breaker(override_model))]

        chain = []
        for name in [settings.primary_model, settings.fallback_model_strong, settings.fallback_model_weak]:
            if name in self._models:
                chain.append((name, self._models[name], self.get_breaker(name)))
        return chain

    @property
    def cost_summary(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self.cost_tracker.total_cost, 6),
            "request_count": self.cost_tracker.request_count,
            "avg_cost": round(self.cost_tracker.avg_cost_per_request, 6),
            "model_distribution": dict(self.cost_tracker.model_usage),
        }


# Module-level singleton
router = ModelRouter()
