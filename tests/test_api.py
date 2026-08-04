import json

from fastapi.testclient import TestClient

from app.main import app


def test_upload_then_run_evaluation():
    case = {
        "sample_id": "q-1::a",
        "question_id": "q-1",
        "query": "员工年假有几天？",
        "answer": "员工年假有十天。",
        "reference_answer": "员工年假有十天。",
    }
    with TestClient(app) as client:
        upload = client.post(
            "/api/datasets/upload",
            files={"file": ("sample.jsonl", json.dumps(case, ensure_ascii=False).encode(), "application/jsonl")},
        )
        assert upload.status_code == 200
        body = upload.json()
        assert body["sample_count"] == 1
        assert "reference_answer" in body["detected_fields"]

        run = client.post(
            "/api/evaluations/run",
            json={"dataset_id": body["dataset_id"], "metric_names": ["exact_match"]},
        )
        assert run.status_code == 200
        assert run.json()["summary"][0]["average"] == 1.0
