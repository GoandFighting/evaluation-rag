from app.evaluation.base import MetricSpec


# Stable contracts for the generation developer. Add evaluators without changing the API/UI.
METRICS = [
    MetricSpec("faithfulness", "忠实度", "generation", "回答中的陈述是否受到上下文支持。", ("answer", "chunks.content")),
    MetricSpec("context_utilization", "上下文利用率", "generation", "回答是否使用了上下文中的关键证据。", ("answer", "chunks.content")),
    MetricSpec("citation_correctness", "引用正确性", "generation", "引用内容是否支持对应结论。", ("answer", "citations", "chunks.chunk_id")),
    MetricSpec("citation_completeness", "引用完整性", "generation", "重要事实是否提供引用。", ("answer", "citations")),
    MetricSpec("factual_consistency", "事实一致性", "generation", "回答与上下文是否存在事实冲突。", ("answer", "chunks.content")),
]
