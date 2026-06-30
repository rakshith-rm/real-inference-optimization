"""Selective speculation scheduler — routes requests to baseline or EAGLE3."""

from __future__ import annotations

from collections import Counter, deque

import config

# (idx, prompt, temperature, max_tokens)
Request = tuple[int, str, float, int]


class SelectiveSpecScheduler:
    def __init__(self) -> None:
        self.min_prompt_tokens = config.SCHED_MIN_PROMPT_TOKENS
        self.max_spec_temperature = config.SCHED_MAX_SPEC_TEMPERATURE
        self.min_rolling_accept_rate = config.SCHED_MIN_ROLLING_ACCEPT_RATE
        self._accept_lengths: deque[float] = deque(maxlen=config.SCHED_ROLLING_WINDOW)
        self.routing_log: list[tuple[bool, str]] = []

    @property
    def rolling_accept_rate(self) -> float:
        if not self._accept_lengths:
            return config.SCHED_INITIAL_ACCEPT_RATE
        return sum(self._accept_lengths) / len(self._accept_lengths)

    @property
    def has_acceptance_samples(self) -> bool:
        return bool(self._accept_lengths)

    def decide(self, prompt_tokens: int, temperature: float) -> tuple[bool, str]:
        if prompt_tokens < self.min_prompt_tokens:
            reason = "short_prompt"
        elif temperature > self.max_spec_temperature:
            reason = "high_temperature"
        elif self._accept_lengths and self.rolling_accept_rate < self.min_rolling_accept_rate:
            reason = "low_rolling_accept_rate"
        else:
            reason = "spec_enabled"
        use_spec = reason == "spec_enabled"
        self.routing_log.append((use_spec, reason))
        return use_spec, reason

    def record_acceptance(self, mean_acceptance_length: float) -> None:
        self._accept_lengths.append(mean_acceptance_length)

    def reason_counts(self) -> dict[str, int]:
        return dict(Counter(r for _, r in self.routing_log))


_tokenizer = None


def prompt_token_count(text: str) -> int:
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(config.TARGET_MODEL)
    return len(_tokenizer.encode(text))
