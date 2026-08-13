from app.rag_adapters.confluence_skill import ConfluenceSkillAdapter
from app.rag_adapters.registry import RAGAdapterRegistry
from app.rag_adapters.smallrag import SmallRAGAdapter


adapter_registry = RAGAdapterRegistry(
    [ConfluenceSkillAdapter(), SmallRAGAdapter()]
)


__all__ = ["adapter_registry"]
