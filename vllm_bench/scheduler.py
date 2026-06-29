"""Selective speculation scheduler — routes requests to baseline or EAGLE3.

vLLM locks speculative_config at engine init, so per-request routing is done by
batching requests onto separate baseline vs spec engine runs (not simultaneous).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import config


@dataclass(frozen=True)
class RoutingDecision:
    use_speculation: bool
    reason: str


@dataclass
class SelectiveSpecScheduler:
    min_prompt_tokens: int = config.SCHED_MIN_PROMPT_TOKENS
    max_spec_temperature: float = config.SCHED_MAX_SPEC_TEMPERATURE
    min_rolling_accept_rate: float = config.SCHED_MIN_ROLLING_ACCEPT_RATE
    rolling_window: int = config.SCHED_ROLLING_WINDOW
    initial_accept_rate: float = config.SCHED_INITIAL_ACCEPT_RATE
    routing_log: list[RoutingDecision] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._accept_lengths: deque[float] = deque(maxlen=self.rolling_window)

    @property
    def has_acceptance_samples(self) -> bool:
        return len(self._accept_lengths) > 0

    @property
    def rolling_accept_rate(self) -> float:
        if not self._accept_lengths:
            return self.initial_accept_rate
        return sum(self._accept_lengths) / len(self._accept_lengths)

    def decide(self, prompt_tokens: int, temperature: float) -> RoutingDecision:
        if prompt_tokens < self.min_prompt_tokens:
            return RoutingDecision(False, "short_prompt")
        if temperature > self.max_spec_temperature:
            return RoutingDecision(False, "high_temperature")
        if self.rolling_accept_rate < self.min_rolling_accept_rate:
            return RoutingDecision(False, "low_rolling_accept_rate")
        return RoutingDecision(True, "spec_enabled")

    def record_acceptance(self, mean_acceptance_length: float) -> None:
        self._accept_lengths.append(mean_acceptance_length)

    def log_decision(self, decision: RoutingDecision) -> None:
        self.routing_log.append(decision)

    def reason_counts(self) -> Counter[str]:
        return Counter(d.reason for d in self.routing_log)


_tokenizer = None


def prompt_token_count(text: str) -> int:
    """Token count for scheduler policy (lazy-loads Llama tokenizer once)."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(config.TARGET_MODEL)
    return len(_tokenizer.encode(text))
