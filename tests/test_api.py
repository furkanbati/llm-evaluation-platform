from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
    }


def test_create_evaluation() -> None:
    response = client.post(
        "/evaluations",
        json={
            "dataset": {
                "name": "math-test",
                "items": [
                    {
                        "input": "2 + 2",
                        "expected_output": "4",
                    },
                ],
            },
            "model_name": "llama3",
            "generated_outputs": ["4"],
            "metrics": ["exact_match"],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["dataset_id"]
    assert data["model_name"] == "llama3"
    assert data["status"] == "completed"
    assert len(data["results"]) == 1
    assert data["results"][0]["generated_output"] == "4"
    assert data["results"][0]["overall_score"] == 1.0
    assert data["results"][0]["metrics"][0]["metric_name"] == "exact_match"
    assert data["results"][0]["metrics"][0]["score"] == 1.0
    assert data["results"][0]["metrics"][0]["passed"] is True
    assert data["completed_at"] is not None


def test_create_evaluation_rejects_invalid_request() -> None:
    response = client.post(
        "/evaluations",
        json={
            "dataset": {
                "name": "math-test",
                "items": [
                    {
                        "input": "2 + 2",
                        "expected_output": "4",
                    },
                ],
            },
            "model_name": "",
            "generated_outputs": ["4"],
            "metrics": ["exact_match"],
        },
    )

    assert response.status_code == 422


def test_create_evaluation_rejects_mismatched_generated_outputs() -> None:
    response = client.post(
        "/evaluations",
        json={
            "dataset": {
                "name": "math-test",
                "items": [
                    {
                        "input": "2 + 2",
                        "expected_output": "4",
                    },
                    {
                        "input": "3 + 3",
                        "expected_output": "6",
                    },
                ],
            },
            "model_name": "llama3",
            "generated_outputs": ["4"],
            "metrics": ["exact_match"],
        },
    )

    assert response.status_code == 400