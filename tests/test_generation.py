from app.evaluation.engine import evaluate_dataset
from app.evaluation.modules.generation import (
    citation_completeness,
    citation_correctness,
    context_utilization,
    factual_consistency,
    faithfulness,
)
from app.schemas import Chunk, EvaluationCase


def make_case(**overrides) -> EvaluationCase:
    values: dict = {
        "sample_id": "gen-1::a",
        "question_id": "q-1",
        "query": "IOMMU 模式",
        "chunks": [
            Chunk(chunk_id="c1", content="RHEL 默认 IOMMU 模式为 DMA Translation", rank=1),
        ],
        "answer": "RHEL 默认 IOMMU 模式为 DMA Translation。",
    }
    values.update(overrides)
    return EvaluationCase(**values)


# --- faithfulness ---------------------------------------------------------------

def test_faithfulness_high_overlap():
    outcome = faithfulness(make_case())
    assert outcome.score == 1.0
    assert outcome.evidence["supported_claims"] == 1


def test_faithfulness_hallucinated():
    case = make_case(answer="Windows 默认使用 Pass-through 模式且需要特殊硬件配置。")
    outcome = faithfulness(case)
    assert outcome.score == 0.0
    assert outcome.evidence["unsupported_claims"] == 1


def test_faithfulness_empty_context():
    case = make_case(chunks=[])
    outcome = faithfulness(case)
    assert outcome.score == 0.0


# --- context_utilization --------------------------------------------------------

def test_context_utilization_chunk_used():
    outcome = context_utilization(make_case())
    assert outcome.score == 1.0
    assert outcome.evidence["utilized_chunks"] == 1


def test_context_utilization_chunk_not_used():
    case = make_case(answer="完全无关的回答内容。")
    outcome = context_utilization(case)
    assert outcome.score == 0.0


def test_context_utilization_multiple_chunks():
    case = make_case(
        chunks=[
            Chunk(chunk_id="c1", content="RHEL 默认 IOMMU 模式为 DMA Translation", rank=1),
            Chunk(chunk_id="c2", content="Ubuntu 使用完全不同的机制", rank=2),
        ],
        answer="RHEL 默认 IOMMU 模式为 DMA Translation。",
    )
    outcome = context_utilization(case)
    assert outcome.evidence["utilized_chunks"] == 1
    assert outcome.score == 0.5


# --- citation_correctness -------------------------------------------------------

def make_citation_case(**overrides) -> EvaluationCase:
    values: dict = {
        "sample_id": "gen-2::a",
        "question_id": "q-2",
        "query": "测试引用",
        "chunks": [
            Chunk(chunk_id="c1", content="片段一内容", rank=1),
            Chunk(chunk_id="c2", content="片段二内容", rank=2),
        ],
        "answer": "第一句[1]。第二句[2]。",
        "citations": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
    }
    values.update(overrides)
    return EvaluationCase(**values)


def test_citation_correctness_all_valid():
    outcome = citation_correctness(make_citation_case())
    assert outcome.score == 1.0
    assert outcome.evidence["valid_citations"] == 4  # 2 inline + 2 list


def test_citation_correctness_invalid_inline():
    case = make_citation_case(answer="第一句[1]。第二句[3]。")
    outcome = citation_correctness(case)
    # [1] valid, [3] invalid (3 > 2 chunks), c1 valid, c2 valid → 3/4
    assert outcome.score == 0.75
    assert outcome.evidence["valid_citations"] == 3


def test_citation_correctness_invalid_list():
    case = make_citation_case(citations=[{"chunk_id": "c1"}, {"chunk_id": "c99"}])
    outcome = citation_correctness(case)
    # [1] valid, [2] valid, c1 valid, c99 invalid → 3/4
    assert outcome.score == 0.75


def test_citation_correctness_no_citations():
    case = make_citation_case(answer="无引用的句子。", citations=[{"chunk_id": "c1"}])
    outcome = citation_correctness(case)
    # No inline markers, 1 valid list citation → 1/1
    assert outcome.score == 1.0


# --- citation_completeness ------------------------------------------------------

def test_citation_completeness_all_cited():
    outcome = citation_completeness(make_citation_case())
    assert outcome.score == 1.0
    assert outcome.evidence["cited_sentences"] == 2


def test_citation_completeness_partial():
    case = make_citation_case(answer="有引用[1]。无引用。")
    outcome = citation_completeness(case)
    assert outcome.score == 0.5


def test_citation_completeness_no_markers_with_list():
    case = make_citation_case(answer="无标记句子一。无标记句子二。", citations=[{"chunk_id": "c1"}])
    outcome = citation_completeness(case)
    # No inline markers, 1 citation → min(1, 2) / 2 = 0.5
    assert outcome.score == 0.5


# --- factual_consistency --------------------------------------------------------

def test_factual_consistency_high_alignment():
    outcome = factual_consistency(make_case())
    assert outcome.score > 0.5
    assert outcome.evidence["level"] == "lexical"


def test_factual_consistency_low_alignment():
    case = make_case(answer="完全无关的回答。")
    outcome = factual_consistency(case)
    assert outcome.score < 0.2


# --- edge cases & engine integration -------------------------------------------

def test_empty_context_not_applicable_for_faithfulness():
    case = make_case(chunks=[])
    result = evaluate_dataset([case], ["faithfulness"])
    assert result["results"][0]["status"] == "not_applicable"


def test_no_citations_not_applicable():
    case = make_case(citations=None)
    result = evaluate_dataset([case], ["citation_correctness"])
    assert result["results"][0]["status"] == "not_applicable"


def test_embedding_and_llm_metrics_not_implemented():
    from app.evaluation.registry import registry

    desc = {m["name"]: m for m in registry.describe_for([make_case()])}
    assert desc["faithfulness_semantic"]["implemented"] is False
    assert desc["faithfulness_llm"]["implemented"] is False
    assert desc["factual_correctness"]["implemented"] is False


def test_all_generation_metrics_run():
    case = make_citation_case()
    result = evaluate_dataset(
        [case],
        ["faithfulness", "context_utilization", "citation_correctness", "citation_completeness", "factual_consistency"],
    )
    assert all(r["status"] == "success" for r in result["results"])
    assert len(result["summary"]) == 5
