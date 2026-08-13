from __future__ import annotations

from collections.abc import Iterable

from app.rag_adapters.base import RAGAdapter


class RAGAdapterRegistry:
    def __init__(self, adapters: Iterable[RAGAdapter] = ()) -> None:
        self._adapters: dict[str, RAGAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: RAGAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"RAG Adapter 名称重复：{adapter.name}")
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> RAGAdapter | None:
        return self._adapters.get(name)

    def all(self) -> list[RAGAdapter]:
        return list(self._adapters.values())

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "name": adapter.name,
                "label": adapter.label,
                "description": adapter.description,
                "available": adapter.is_available(),
                "capabilities": adapter.capabilities.model_dump(),
            }
            for adapter in self.all()
        ]
