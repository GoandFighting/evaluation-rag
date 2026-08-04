import json

from app.services.datasets import detect_present_fields, parse_dataset


def test_aliases_are_normalized_to_canonical_fields():
    payload = json.dumps(
        {
            "items": [
                {
                    "id": "q-1",
                    "question": "年假有几天？",
                    "docs": [
                        {
                            "doc_id": "policy-1",
                            "text": "员工每年有十天年假。",
                            "source": "leave.pdf",
                        }
                    ],
                    "result": "十天。",
                    "ground_truth": {
                        "reference_answer": "员工每年有十天年假。",
                        "key_points": ["十天"],
                    },
                }
            ]
        },
        ensure_ascii=False,
    ).encode()

    cases = parse_dataset("dataset.json", payload)

    assert cases[0].query == "年假有几天？"
    assert cases[0].answer == "十天。"
    assert cases[0].chunks[0].document_id == "policy-1"
    assert cases[0].chunks[0].source == "leave.pdf"
    assert "chunks.source" in detect_present_fields(cases)
