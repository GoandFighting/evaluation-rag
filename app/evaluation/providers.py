"""Runtime contracts for model-backed evaluation metrics.

Concrete providers are intentionally kept outside the evaluation engine.  An
internal model integration only needs to implement these protocols and place
the instances in :class:`EvaluationContext`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


Capability = Literal["embedding", "llm_judge"]


class EmbeddingProvider(Protocol):
    """Asynchronous embedding interface used by semantic metrics."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text."""
        ...

    def similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Return a documented similarity score for two vectors."""
        ...


class LLMJudge(Protocol):
    """Asynchronous judge interface used by rubric and claim metrics."""

    async def judge(self, prompt: str, **kwargs: Any) -> str:
        """Return a free-form judgement when structured output is unnecessary."""
        ...

    async def judge_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Return validated JSON-compatible judgement data."""
        ...


@dataclass(slots=True)
class EvaluationContext:
    """Providers and execution limits supplied to model-backed metrics."""

    embedding_provider: EmbeddingProvider | None = None
    llm_judge: LLMJudge | None = None
    max_concurrency: int = 4
    timeout_seconds: float = 60.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_capability(self, capability: Capability) -> bool:
        if capability == "embedding":
            return self.embedding_provider is not None
        if capability == "llm_judge":
            return self.llm_judge is not None
        return False

    def missing_capabilities(
        self, capabilities: tuple[Capability, ...]
    ) -> list[Capability]:
        return [item for item in capabilities if not self.has_capability(item)]
