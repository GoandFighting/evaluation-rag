"""Provider protocols for embedding-based and LLM-as-judge metrics.

These protocols define the interfaces that future embedding models and LLM
judges must satisfy.  Metrics that require these capabilities are declared with
``evaluator=None``; once a concrete provider is deployed, the evaluator can be
wired in without changing the metric contract or the front-end.
"""

from __future__ import annotations

from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    """Embedding model interface for semantic similarity metrics."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text."""
        ...

    def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Return similarity in [0, 1] between two vectors."""
        ...


class LLMJudge(Protocol):
    """LLM-as-judge interface for faithfulness, factual correctness, etc."""

    def judge(self, prompt: str, **kwargs: Any) -> str:
        """Return a free-form judgement string."""
        ...

    def judge_binary(self, prompt: str, **kwargs: Any) -> bool:
        """Return a boolean judgement (e.g. supported / not supported)."""
        ...
