from app.evaluation.engine import evaluate_dataset
from app.evaluation.registry import registry
from app.schemas import EvaluationCase


def make_case(**overrides):
    values = {
        "sample_id": "q-1::a",
        "question_id": "q-1",
        "query": "员工年假有几天",
        "answer": "员工年假有十天",
        "reference_answer": "员工年假有十天",
        "key_points": ["十天"],
    }
    values.update(overrides)
    return EvaluationCase(**values)


def test_registry_reports_per_metric_field_coverage():
    descriptions = {item["name"]: item for item in registry.describe_for([make_case()])}

    assert descriptions["token_f1"]["runnable"] is True
    assert descriptions["format_compliance"]["runnable"] is False
    assert descriptions["faithfulness"]["implemented"] is False
    assert descriptions["faithfulness"]["eligible_samples"] == 0


def test_end_to_end_metrics_return_summary_and_details():
    result = evaluate_dataset(
        [make_case()], ["exact_match", "token_f1", "key_point_completeness"]
    )

    assert len(result["summary"]) == 3
    assert all(item["average"] == 1.0 for item in result["summary"])
    assert all(item["status"] == "success" for item in result["results"])


def test_missing_fields_are_not_applicable():
    result = evaluate_dataset([make_case(reference_answer=None)], ["exact_match"])

    assert result["results"][0]["status"] == "not_applicable"
    assert result["summary"][0]["average"] is None
