from app.evaluation.engine import evaluate_dataset
from app.evaluation.modules.retrieval import (
    context_relevance,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.schemas import Chunk, EvaluationCase


def make_case(**overrides) -> EvaluationCase:
    values: dict = {
        "sample_id": "q-1::a",
        "question_id": "q-1",
        "query": "Linux IOMMU 模式",
        "chunks": [
            Chunk(chunk_id="c1", content="RHEL 默认 IOMMU 模式为 DMA Translation", document_id="d1", rank=1),
            Chunk(chunk_id="c2", content="Ubuntu 使用 Pass-through 模式", document_id="d2", rank=2),
        ],
        "relevant_chunk_ids": ["c1"],
    }
    values.update(overrides)
    return EvaluationCase(**values)


# --- chunk-level relevance (relevant_chunk_ids) ---------------------------------

def test_hit_rate_chunk_level_hit():
    outcome = hit_rate_at_k(make_case())
    assert outcome.score == 1.0
    assert outcome.evidence["relevance_level"] == "chunk"


def test_hit_rate_chunk_level_miss():
    case = make_case(relevant_chunk_ids=["c99"])
    assert hit_rate_at_k(case).score == 0.0


def test_recall_all_relevant_found():
    case = make_case(relevant_chunk_ids=["c1", "c2"])
    assert recall_at_k(case).score == 1.0


def test_recall_partial():
    case = make_case(relevant_chunk_ids=["c1", "c99"])
    assert recall_at_k(case).score == 0.5


def test_precision_at_k():
    # 2 chunks, 1 relevant → 1/2 = 0.5
    assert precision_at_k(make_case()).score == 0.5


def test_mrr_first_rank():
    assert mrr(make_case()).score == 1.0


def test_mrr_second_rank():
    case = make_case(relevant_chunk_ids=["c2"])
    assert mrr(case).score == 0.5


def test_mrr_no_hit():
    case = make_case(relevant_chunk_ids=["c99"])
    assert mrr(case).score == 0.0


def test_ndcg_perfect_ranking():
    assert ndcg_at_k(make_case()).score == 1.0


def test_ndcg_relevant_second():
    case = make_case(relevant_chunk_ids=["c2"])
    # DCG = 0/log2(2) + 1/log2(3) = 0 + 0.6309
    # IDCG = 1/log2(2) = 1.0
    assert 0.6 < ndcg_at_k(case).score < 0.7


def test_context_relevance_overlap():
    outcome = context_relevance(make_case(query="IOMMU 模式"))
    assert outcome.score > 0.0
    assert "iommu" in outcome.evidence["matched_tokens"]


# --- doc-level relevance fallback (relevant_doc_ids) ---------------------------

def test_doc_level_fallback_hit():
    case = make_case(relevant_chunk_ids=None, relevant_doc_ids=["d1"])
    outcome = hit_rate_at_k(case)
    assert outcome.score == 1.0
    assert outcome.evidence["relevance_level"] == "doc"


def test_doc_level_fallback_recall():
    case = make_case(relevant_chunk_ids=None, relevant_doc_ids=["d1", "d2"])
    assert recall_at_k(case).score == 1.0


def test_doc_level_fallback_mrr():
    case = make_case(relevant_chunk_ids=None, relevant_doc_ids=["d2"])
    assert mrr(case).score == 0.5


# --- any_of_fields: either relevance label makes metric runnable ---------------

def test_chunk_relevance_labels_make_metric_runnable():
    from app.evaluation.registry import registry

    desc = {m["name"]: m for m in registry.describe_for([make_case()])}
    assert desc["hit_rate_at_k"]["runnable"] is True
    assert desc["hit_rate_at_k"]["eligible_samples"] == 1


def test_doc_relevance_labels_make_metric_runnable():
    from app.evaluation.registry import registry

    case = make_case(relevant_chunk_ids=None, relevant_doc_ids=["d1"])
    desc = {m["name"]: m for m in registry.describe_for([case])}
    assert desc["hit_rate_at_k"]["runnable"] is True


def test_no_relevance_labels_not_applicable():
    result = evaluate_dataset([make_case(relevant_chunk_ids=None)], ["hit_rate_at_k"])
    assert result["results"][0]["status"] == "not_applicable"


# --- edge cases -----------------------------------------------------------------

def test_empty_chunks_not_applicable():
    case = make_case(chunks=[])
    result = evaluate_dataset([case], ["hit_rate_at_k"])
    assert result["results"][0]["status"] == "not_applicable"


def test_context_relevance_empty_chunks():
    case = make_case(chunks=[])
    outcome = context_relevance(case)
    assert outcome.score == 0.0


def test_embedding_and_llm_metrics_not_implemented():
    from app.evaluation.registry import registry

    desc = {m["name"]: m for m in registry.describe_for([make_case()])}
    assert desc["context_relevance_semantic"]["implemented"] is False
    assert desc["context_precision"]["implemented"] is False


def test_all_retrieval_metrics_run_on_golden_format():
    case = make_case()
    result = evaluate_dataset(
        [case],
        ["hit_rate_at_k", "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "context_relevance"],
    )
    assert all(r["status"] == "success" for r in result["results"])
    assert len(result["summary"]) == 6
