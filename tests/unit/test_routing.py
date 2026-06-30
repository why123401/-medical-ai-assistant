"""Unit tests for AI routing — circuit breaker and fallback chain."""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.ai.routing.router import CircuitBreaker, CircuitState, CostTracker
from src.ai.routing.fallback_chain import build_fallback_chain, is_model_in_chain


class TestCircuitBreaker:
    """Test circuit breaker state transitions."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(threshold=3, timeout=10)
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3, timeout=10)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # Not yet

        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_available is False

    def test_recovers_to_half_open_after_timeout(self):
        cb = CircuitBreaker(threshold=2, timeout=0)  # 0 timeout for fast test
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.01)  # Wait for timeout
        assert cb.allow_request() is True  # Transitions to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_successful_half_open(self):
        cb = CircuitBreaker(threshold=2, timeout=0, half_open_max=1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.01)

        cb.allow_request()  # Transition to HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_opens_again_if_half_open_fails(self):
        cb = CircuitBreaker(threshold=2, timeout=0, half_open_max=1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.01)

        cb.allow_request()  # Transition to HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestCostTracker:
    """Test cost tracking calculations."""

    def test_records_qwen_plus_cost(self):
        tracker = CostTracker()
        cost = tracker.record("qwen-plus", 1000, 500)
        # input: 0.04/1M tokens, output: 0.12/1M tokens
        expected = (1000 / 1_000_000) * 0.04 + (500 / 1_000_000) * 0.12
        assert abs(cost - expected) < 1e-9

    def test_tracks_total_cost(self):
        tracker = CostTracker()
        tracker.record("qwen-plus", 1000, 500)
        tracker.record("qwen-turbo", 500, 200)
        assert tracker.request_count == 2
        assert tracker.total_cost > 0

    def test_avg_cost(self):
        tracker = CostTracker()
        tracker.record("qwen-plus", 1000, 500)
        tracker.record("qwen-plus", 1000, 500)
        assert tracker.avg_cost_per_request == tracker.total_cost / 2


class TestFallbackChain:
    """Test model fallback chain construction."""

    def test_default_chain_order(self):
        chain = build_fallback_chain()
        assert chain[0] == "qwen-plus"
        assert chain[1] == "qwen-max"
        assert chain[2] == "qwen-turbo"

    def test_override_model(self):
        chain = build_fallback_chain(override_model="qwen-turbo")
        assert chain == ["qwen-turbo"]

    def test_model_in_chain(self):
        assert is_model_in_chain("qwen-plus") is True
        assert is_model_in_chain("qwen-turbo") is True
        assert is_model_in_chain("gpt-4") is False
