from uuid import uuid4

from fastapi.testclient import TestClient

import app.api as api
from app.api import (
    app,
    create_evaluation_runner,
    evaluation_repository,
)
from app.models import (
    EvaluationRun,
    GenerationOptions,
)
from app.services.model_client import ModelClient
from app.storage.evaluation_repository import EvaluationRepository
from app.storage.sqlite_evaluation_repository import (
    SQLiteEvaluationRepository,
)


class FakeModelClient(ModelClient):
    def generate(
        self,
        model_name: str,
        inputs: list[str],
        options: GenerationOptions | None = None,
    ) -> list[str]:
        return ["4"]


class FakeEvaluationWithJudgeModelClient(ModelClient):
    def generate(
        self,
        model_name: str,
        inputs: list[str],
        options: GenerationOptions | None = None,
    ) -> list[str]:
        if "Respond with only YES or NO." in inputs[0]:
            return ["YES"]

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

    assert response.status_code == 202

    accepted = response.json()

    assert accepted["status"] == "pending"
    assert accepted["results"] == []
    assert accepted["completed_at"] is None
    assert response.headers["location"] == (f"/evaluations/{accepted['id']}")

    response = client.get(
        f"/evaluations/{accepted['id']}",
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
    assert runner.repository.list() == []


def test_create_evaluation_returns_500_for_unexpected_error(
    monkeypatch,
) -> None:
    class FailingRunner:
        def create_run(
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
        def create_run(
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
        def create_run(
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
        def create_run(
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
    assert response.headers["content-type"].startswith("application/json")
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

    assert create_response.status_code == 202

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
        "dataset",
        "model_name",
        "metrics",
        "judge_model_name",
        "generation_options",
        "status",
        "results",
        "created_at",
        "completed_at",
        "error",
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

        assert response.status_code == 202

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
        "/evaluations?status=completed&model_name=llama3",
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
        "/evaluations?status=completed&model_name=llama3&limit=1&offset=1",
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

    assert data["info"]["title"] == ("LLM Evaluation Platform")

    assert data["info"]["version"] == "1.0.0"

    assert "evaluating Large Language Model outputs" in data["info"]["description"]


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

    assert response.status_code == 202

    response = client.get(
        f"/evaluations/{response.json()['id']}",
    )

    assert response.status_code == 200

    data = response.json()

    assert data["results"][0]["metrics"][0]["metric_name"] == "similarity"


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

    assert runner._engine._registry.get("llm_judge") is not None


def test_create_evaluation_supports_llm_judge_metric(
    monkeypatch,
) -> None:
    runner = create_evaluation_runner(
        model_client=FakeEvaluationWithJudgeModelClient(),
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
            "metrics": ["llm_judge"],
        },
    )

    assert response.status_code == 202

    response = client.get(
        f"/evaluations/{response.json()['id']}",
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert len(data["results"]) == 1

    metric = data["results"][0]["metrics"][0]

    assert metric["metric_name"] == "llm_judge"
    assert metric["score"] == 1.0
    assert metric["passed"] is True


def test_default_runner_uses_deterministic_judge_client() -> None:
    runner = create_evaluation_runner()

    judge = runner._engine._registry.get("llm_judge")

    assert judge._model_client._options == {
        "temperature": 0.0,
    }


def test_runner_uses_separate_judge_model_client() -> None:
    judge_client = FakeEvaluationWithJudgeModelClient()

    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
        judge_model_client=judge_client,
    )

    judge = runner._engine._registry.get("llm_judge")

    assert judge._model_client is judge_client


def test_create_evaluation_returns_before_running_evaluation(
    monkeypatch,
) -> None:
    from fastapi import BackgroundTasks, Response

    from app.models import EvaluationRequest

    class CountingModelClient(ModelClient):
        def __init__(self) -> None:
            self.calls = 0

        def generate(
            self,
            model_name: str,
            inputs: list[str],
            options: GenerationOptions | None = None,
        ) -> list[str]:
            self.calls += 1
            return ["4"]

    model_client = CountingModelClient()

    runner = create_evaluation_runner(
        model_client=model_client,
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    background_tasks = BackgroundTasks()

    accepted = api.create_evaluation(
        request=EvaluationRequest(
            dataset={
                "name": "math-test",
                "items": [
                    {
                        "input": "2 + 2",
                        "expected_output": "4",
                    },
                ],
            },
            model_name="llama3",
            metrics=["exact_match"],
        ),
        background_tasks=background_tasks,
        response=Response(),
    )

    assert accepted.status == "pending"
    assert model_client.calls == 0
    assert len(background_tasks.tasks) == 1

    task = background_tasks.tasks[0]
    task.func(*task.args, **task.kwargs)

    stored = runner.repository.get(accepted.id)

    assert model_client.calls == 1
    assert stored.status == "completed"
    assert accepted.status == "pending"


def test_create_evaluation_reports_failure_through_get(
    monkeypatch,
) -> None:
    class UnavailableModelClient(ModelClient):
        def generate(
            self,
            model_name: str,
            inputs: list[str],
            options: GenerationOptions | None = None,
        ) -> list[str]:
            raise RuntimeError("Ollama is unavailable")

    runner = create_evaluation_runner(
        model_client=UnavailableModelClient(),
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

    assert response.status_code == 202

    response = client.get(
        f"/evaluations/{response.json()['id']}",
    )

    data = response.json()

    assert data["status"] == "failed"
    assert data["error"] == "Ollama is unavailable"


def test_create_repository_uses_sqlite_when_path_is_set(
    tmp_path,
) -> None:
    repository = api.create_repository(
        str(tmp_path / "evaluations.db"),
    )

    assert isinstance(repository, SQLiteEvaluationRepository)


def test_create_repository_uses_memory_when_path_is_empty() -> None:
    assert isinstance(
        api.create_repository(""),
        EvaluationRepository,
    )


def test_startup_marks_interrupted_runs_as_failed(
    monkeypatch,
    tmp_path,
) -> None:
    repository = SQLiteEvaluationRepository(
        str(tmp_path / "evaluations.db"),
    )

    interrupted = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
        status="running",
    )
    repository.save(interrupted)

    runner = create_evaluation_runner(
        model_client=FakeModelClient(),
        repository=repository,
    )

    monkeypatch.setattr(
        api,
        "evaluation_runner",
        runner,
    )

    with TestClient(app) as client:
        response = client.get(
            f"/evaluations/{interrupted.id}",
        )

    data = response.json()

    assert data["status"] == "failed"
    assert data["error"] == "Interrupted by application restart"


def test_list_evaluations_rejects_invalid_pagination() -> None:
    client = TestClient(app)

    for query in (
        "limit=0",
        "limit=1001",
        "limit=-5",
        "offset=-1",
    ):
        response = client.get(f"/evaluations?{query}")

        assert response.status_code == 422, query


def test_create_evaluation_uses_request_generation_options(
    monkeypatch,
) -> None:
    class OptionsCaptureModelClient(ModelClient):
        def __init__(self) -> None:
            self.options: GenerationOptions | None = None

        def generate(
            self,
            model_name: str,
            inputs: list[str],
            options: GenerationOptions | None = None,
        ) -> list[str]:
            self.options = options
            return ["4"]

    model_client = OptionsCaptureModelClient()

    runner = create_evaluation_runner(
        model_client=model_client,
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
            "generation_options": {
                "temperature": 0.7,
                "seed": 42,
            },
        },
    )

    assert response.status_code == 202
    assert response.json()["generation_options"] == {
        "temperature": 0.7,
        "seed": 42,
    }
    assert model_client.options == GenerationOptions(temperature=0.7, seed=42)


def test_create_evaluation_defaults_to_zero_temperature(
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

    assert response.json()["generation_options"] == {
        "temperature": 0.0,
        "seed": None,
    }


def test_create_evaluation_rejects_invalid_generation_options() -> None:
    client = TestClient(app)

    for options in (
        {"temperature": 3.0},
        {"seed": "abc"},
        {"top_k": 10},
    ):
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
                "generation_options": options,
            },
        )

        assert response.status_code == 422, options
