from app.evaluation.base import MetricSpec


# Stable contracts for the retrieval developer. Add evaluators without changing the API/UI.
METRICS = [
    MetricSpec("hit_rate_at_k", "Hit Rate@K", "retrieval", "Top-K 是否命中相关文档。", ("chunks.document_id", "relevant_doc_ids")),
    MetricSpec("recall_at_k", "Recall@K", "retrieval", "黄金文档被召回的比例。", ("chunks.document_id", "relevant_doc_ids")),
    MetricSpec("precision_at_k", "Precision@K", "retrieval", "Top-K 中相关文档的比例。", ("chunks.document_id", "relevant_doc_ids")),
    MetricSpec("mrr", "MRR", "retrieval", "首个相关结果排名的倒数。", ("chunks.document_id", "chunks.rank", "relevant_doc_ids")),
    MetricSpec("context_relevance", "上下文相关性", "retrieval", "检索内容对问题是否有用。", ("query", "chunks.content")),
]
