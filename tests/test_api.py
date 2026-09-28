
from uuid import uuid4

import app.api as api
from fastapi.testclient import TestClient

from app.api import (
    app,
    create_evaluation_runner,
    evaluation_repository,
)
from app.models import (
    EvaluationRun,
)
from app.services.model_client import ModelClient


class FakeModelClient(ModelClient):
    def generate(
        self,
        model_name: str,
        inputs: list[str],
    ) -> list[str]:
        return ["4"]


def test_health() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
    }


def test_create_evaluation(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    client = TestClient(app)

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
    client = TestClient(app)

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
            "metrics": ["exact_match"],
        },
    )

    assert response.status_code == 422


def test_create_evaluation_returns_400_for_unknown_metric(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    client = TestClient(app)

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
            "metrics": ["unknown_metric"],
        },
    )

    assert response.status_code == 400
    assert "unknown_metric" in response.json()["detail"]


def test_create_evaluation_returns_500_for_unexpected_error(
    monkeypatch,
) -> None:
    class FailingRunner:
        def run(
            self,
            dataset,
            model_name,
            metrics,
        ):
            raise RuntimeError("unexpected failure")

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        FailingRunner(),
    )

    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

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
            "metrics": ["exact_match"],
        },
    )

    assert response.status_code == 500


def test_create_evaluation_returns_error_detail_for_unexpected_error(
    monkeypatch,
) -> None:
    class FailingRunner:
        def run(
            self,
            dataset,
            model_name,
            metrics,
        ):
            raise RuntimeError("unexpected failure")

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        FailingRunner(),
    )

    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

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
            "metrics": ["exact_match"],
        },
    )

    assert response.status_code == 500


def test_create_evaluation_hides_internal_error_details(
    monkeypatch,
) -> None:
    class FailingRunner:
        def run(
            self,
            dataset,
            model_name,
            metrics,
        ):
            raise RuntimeError("database password leaked")

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        FailingRunner(),
    )

    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

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
            "metrics": ["exact_match"],
        },
    )

    assert response.status_code == 500

    data = response.json()

    assert data["detail"] == "Internal server error"


def test_create_evaluation_returns_json_for_internal_errors(
    monkeypatch,
) -> None:
    class FailingRunner:
        def run(
            self,
            dataset,
            model_name,
            metrics,
        ):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        FailingRunner(),
    )

    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

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
            "metrics": ["exact_match"],
        },
    )

    assert response.status_code == 500
    assert response.headers["content-type"].startswith(
        "application/json"
    )
    assert response.json() == {
        "detail": "Internal server error",
    }


def test_get_evaluation_returns_saved_evaluation(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )
    monkeypatch.setattr(
        api,
        "evaluation_repository",
        runner.repository,
    )

    client = TestClient(app)

    create_response = client.post(
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
            "metrics": ["exact_match"],
        },
    )

    assert create_response.status_code == 200

    evaluation_id = create_response.json()["id"]

    response = client.get(
        f"/evaluations/{evaluation_id}",
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == evaluation_id
    assert data["model_name"] == "llama3"
    assert data["status"] == "completed"
    assert len(data["results"]) == 1


def test_get_evaluation_returns_404_for_unknown_id() -> None:
    client = TestClient(app)

    evaluation_id = "00000000-0000-0000-0000-000000000000"

    response = client.get(
        f"/evaluations/{evaluation_id}",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Evaluation not found",
    }


def test_get_evaluation_rejects_invalid_uuid() -> None:
    client = TestClient(app)

    response = client.get(
        "/evaluations/not-a-uuid",
    )

    assert response.status_code == 422


def test_get_evaluation_response_contains_expected_fields(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )
    monkeypatch.setattr(
        api,
        "evaluation_repository",
        runner.repository,
    )

    client = TestClient(app)

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
            "metrics": ["exact_match"],
        },
    )

    evaluation_id = response.json()["id"]

    get_response = client.get(
        f"/evaluations/{evaluation_id}",
    )

    data = get_response.json()

    assert set(data) == {
        "id",
        "dataset_id",
        "model_name",
        "status",
        "results",
        "created_at",
        "completed_at",
    }

def test_list_evaluations_returns_saved_evaluations(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    client = TestClient(app)

    for expected_output in ["4", "4"]:
        response = client.post(
            "/evaluations",
            json={
                "dataset": {
                    "name": "math-test",
                    "items": [
                        {
                            "input": "2 + 2",
                            "expected_output": expected_output,
                        },
                    ],
                },
                "model_name": "llama3",
                "metrics": ["exact_match"],
            },
        )

        assert response.status_code == 200

    response = client.get("/evaluations")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2
    assert data[0]["model_name"] == "llama3"
    assert data[1]["model_name"] == "llama3"


def test_list_evaluations_returns_empty_list(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    client = TestClient(app)

    response = client.get("/evaluations")

    assert response.status_code == 200
    assert response.json() == []


def test_get_evaluation_summary_returns_statistics(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )
    monkeypatch.setattr(
        api,
        "evaluation_repository",
        runner.repository,
    )

    client = TestClient(app)

    create_response = client.post(
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
            "metrics": ["exact_match"],
        },
    )

    evaluation_id = create_response.json()["id"]

    response = client.get(
        f"/evaluations/{evaluation_id}/summary",
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_items"] == 1
    assert data["passed_items"] == 1
    assert data["failed_items"] == 0
    assert data["average_score"] == 1.0
    assert data["success_rate"] == 1.0


def test_get_evaluation_summary_returns_404_for_unknown_id() -> None:
    client = TestClient(app)

    response = client.get(
        "/evaluations/00000000-0000-0000-0000-000000000000/summary",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Evaluation not found",
    }


def test_delete_evaluation_removes_saved_evaluation() -> None:
    evaluation = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
    )

    evaluation_repository.save(
        evaluation,
    )

    client = TestClient(app)

    response = client.delete(
        f"/evaluations/{evaluation.id}",
    )

    assert response.status_code == 204

    assert (
        evaluation_repository.get(
            evaluation.id,
        )
        is None
    )


def test_delete_evaluation_returns_404_for_unknown_id() -> None:
    client = TestClient(app)

    response = client.delete(
        f"/evaluations/{uuid4()}",
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Evaluation not found",
    }


def test_list_evaluations_filters_by_status(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    completed = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
        status="completed",
    )
    failed = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
        status="failed",
    )

    runner.repository.save(completed)
    runner.repository.save(failed)

    client = TestClient(app)

    response = client.get(
        "/evaluations?status=completed",
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["id"] == str(completed.id)
    assert data[0]["status"] == "completed"


def test_list_evaluations_filters_by_model_name(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    llama3 = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
    )
    llama32 = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3.2",
    )

    runner.repository.save(llama3)
    runner.repository.save(llama32)

    client = TestClient(app)

    response = client.get(
        "/evaluations?model_name=llama3",
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["id"] == str(llama3.id)
    assert data[0]["model_name"] == "llama3"


def test_list_evaluations_filters_by_status_and_model_name(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    matching = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
        status="completed",
    )
    wrong_status = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
        status="failed",
    )
    wrong_model = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3.2",
        status="completed",
    )

    runner.repository.save(matching)
    runner.repository.save(wrong_status)
    runner.repository.save(wrong_model)

    client = TestClient(app)

    response = client.get(
        "/evaluations"
        "?status=completed"
        "&model_name=llama3",
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["id"] == str(matching.id)


def test_list_evaluations_rejects_invalid_status() -> None:
    client = TestClient(app)

    response = client.get(
        "/evaluations?status=invalid",
    )

    assert response.status_code == 422


def test_list_evaluations_supports_limit(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    for model_name in ["llama3", "llama3.2", "llama3.3"]:
        runner.repository.save(
            EvaluationRun(
                dataset_id=uuid4(),
                model_name=model_name,
            )
        )

    client = TestClient(app)

    response = client.get(
        "/evaluations?limit=2",
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2
    assert data[0]["model_name"] == "llama3"
    assert data[1]["model_name"] == "llama3.2"


def test_list_evaluations_supports_offset(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    for model_name in ["llama3", "llama3.2", "llama3.3"]:
        runner.repository.save(
            EvaluationRun(
                dataset_id=uuid4(),
                model_name=model_name,
            )
        )

    client = TestClient(app)

    response = client.get(
        "/evaluations?offset=1",
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2
    assert data[0]["model_name"] == "llama3.2"
    assert data[1]["model_name"] == "llama3.3"


def test_list_evaluations_applies_filters_before_pagination(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    for model_name, status in [
        ("llama3", "completed"),
        ("llama3", "completed"),
        ("llama3", "failed"),
        ("llama3.2", "completed"),
    ]:
        runner.repository.save(
            EvaluationRun(
                dataset_id=uuid4(),
                model_name=model_name,
                status=status,
            )
        )

    client = TestClient(app)

    response = client.get(
        "/evaluations"
        "?status=completed"
        "&model_name=llama3"
        "&limit=1"
        "&offset=1",
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["model_name"] == "llama3"
    assert data[0]["status"] == "completed"


def test_openapi_contains_application_metadata() -> None:
    client = TestClient(app)

    response = client.get(
        "/openapi.json",
    )

    assert response.status_code == 200

    data = response.json()

    assert data["info"]["title"] == (
        "LLM Evaluation Platform"
    )

    assert data["info"]["version"] == "1.0.0"

    assert (
        "evaluating Large Language Model outputs"
        in data["info"]["description"]
    )


def test_create_evaluation_supports_similarity_metric(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    client = TestClient(app)

    response = client.post(
        "/evaluations",
        json={
            "dataset": {
                "name": "test",
                "items": [
                    {
                        "input": "Question",
                        "expected_output": "4",
                    },
                ],
            },
            "model_name": "llama3",
            "metrics": ["similarity"],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["results"][0]["metrics"][0]["metric_name"]
        == "similarity"
    )


def test_list_metrics_returns_available_metrics() -> None:
    client = TestClient(app)

    response = client.get(
        "/metrics",
    )

    assert response.status_code == 200

    assert response.json() == {
        "metrics": [
            "exact_match",
            "similarity",
            "llm_judge",
        ],
    }


def test_create_evaluation_runner_registers_llm_judge() -> None:
    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
    )

    assert (
        runner._engine._registry.get("llm_judge")
        is not None
    )

