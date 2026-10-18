import threading
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.models import (
    Dataset,
    DatasetItem,
    EvaluationResult,
    EvaluationRun,
    EvaluationStatus,
    MetricResult,
)
from app.storage.evaluation_repository import EvaluationRepository
from app.storage.sqlite_evaluation_repository import (
    SQLiteEvaluationRepository,
)


@pytest.fixture(params=["memory", "sqlite"])
def repository(request, tmp_path):
    if request.param == "memory":
        return EvaluationRepository()

    return SQLiteEvaluationRepository(
        str(tmp_path / "evaluations.db"),
    )


def create_evaluation_run(
    model_name: str = "llama3",
    status: EvaluationStatus = EvaluationStatus.PENDING,
) -> EvaluationRun:
    return EvaluationRun(
        dataset_id=uuid4(),
        model_name=model_name,
        status=status,
    )


def test_save_stores_evaluation(repository) -> None:
    evaluation = create_evaluation_run()

    repository.save(evaluation)

    assert repository.get(evaluation.id) == evaluation


def test_save_returns_saved_evaluation(repository) -> None:
    evaluation = create_evaluation_run()

    result = repository.save(evaluation)

    assert result is evaluation


def test_get_returns_none_for_unknown_id(repository) -> None:

    result = repository.get(uuid4())

    assert result is None


def test_save_updates_existing_evaluation(repository) -> None:
    evaluation = create_evaluation_run()

    repository.save(evaluation)

    evaluation.model_name = "llama3.2"

    repository.save(evaluation)

    result = repository.get(evaluation.id)

    assert result == evaluation
    assert result.model_name == "llama3.2"


def test_list_returns_saved_evaluations(repository) -> None:

    first = create_evaluation_run()
    second = create_evaluation_run()

    repository.save(first)
    repository.save(second)

    result = repository.list()

    assert result == [first, second]


def test_list_returns_empty_list_when_repository_is_empty(repository) -> None:

    result = repository.list()

    assert result == []


def test_list_filters_by_status(repository) -> None:

    completed = create_evaluation_run(
        status=EvaluationStatus.COMPLETED,
    )
    failed = create_evaluation_run(
        status=EvaluationStatus.FAILED,
    )

    repository.save(completed)
    repository.save(failed)

    result = repository.list(
        status=EvaluationStatus.COMPLETED,
    )

    assert result == [completed]


def test_list_filters_by_model_name(repository) -> None:

    llama3 = create_evaluation_run(
        model_name="llama3",
    )
    llama32 = create_evaluation_run(
        model_name="llama3.2",
    )

    repository.save(llama3)
    repository.save(llama32)

    result = repository.list(
        model_name="llama3",
    )

    assert result == [llama3]


def test_list_filters_by_status_and_model_name(repository) -> None:

    matching = create_evaluation_run(
        model_name="llama3",
        status=EvaluationStatus.COMPLETED,
    )
    wrong_status = create_evaluation_run(
        model_name="llama3",
        status=EvaluationStatus.FAILED,
    )
    wrong_model = create_evaluation_run(
        model_name="llama3.2",
        status=EvaluationStatus.COMPLETED,
    )

    repository.save(matching)
    repository.save(wrong_status)
    repository.save(wrong_model)

    result = repository.list(
        status=EvaluationStatus.COMPLETED,
        model_name="llama3",
    )

    assert result == [matching]


def test_list_returns_empty_list_when_status_has_no_matches(repository) -> None:

    evaluation = create_evaluation_run(
        status=EvaluationStatus.COMPLETED,
    )

    repository.save(evaluation)

    result = repository.list(
        status=EvaluationStatus.FAILED,
    )

    assert result == []


def test_list_returns_empty_list_when_model_name_has_no_matches(repository) -> None:

    evaluation = create_evaluation_run(
        model_name="llama3",
    )

    repository.save(evaluation)

    result = repository.list(
        model_name="llama3.2",
    )

    assert result == []


def test_delete_removes_saved_evaluation(repository) -> None:

    evaluation = create_evaluation_run()

    repository.save(evaluation)

    deleted = repository.delete(
        evaluation.id,
    )

    assert deleted is True
    assert (
        repository.get(
            evaluation.id,
        )
        is None
    )


def test_delete_returns_false_for_unknown_id(repository) -> None:

    result = repository.delete(
        uuid4(),
    )

    assert result is False


def test_repository_handles_concurrent_saves_and_lists() -> None:
    repository = EvaluationRepository()
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(200):
                repository.save(
                    EvaluationRun(
                        dataset_id=uuid4(),
                        model_name="llama3",
                    )
                )
                repository.list()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []
    assert len(repository.list()) == 1_600


def test_list_supports_limit_and_offset(repository) -> None:
    evaluations = [
        create_evaluation_run(model_name=f"model-{index}") for index in range(5)
    ]

    for evaluation in evaluations:
        repository.save(evaluation)

    result = repository.list(
        limit=2,
        offset=1,
    )

    assert [evaluation.model_name for evaluation in result] == [
        "model-1",
        "model-2",
    ]


def test_list_applies_filters_before_limit(repository) -> None:
    repository.save(create_evaluation_run(model_name="other"))
    repository.save(create_evaluation_run(model_name="llama3"))
    repository.save(create_evaluation_run(model_name="llama3"))

    result = repository.list(
        model_name="llama3",
        limit=1,
    )

    assert len(result) == 1
    assert result[0].model_name == "llama3"


def test_list_keeps_insertion_order_after_update(repository) -> None:
    first = create_evaluation_run(model_name="first")
    second = create_evaluation_run(model_name="second")

    repository.save(first)
    repository.save(second)

    first.status = EvaluationStatus.COMPLETED
    repository.save(first)

    assert [evaluation.id for evaluation in repository.list()] == [
        first.id,
        second.id,
    ]


def test_sqlite_repository_persists_across_instances(tmp_path) -> None:
    path = str(tmp_path / "evaluations.db")

    evaluation = create_evaluation_run()

    SQLiteEvaluationRepository(path).save(evaluation)

    reopened = SQLiteEvaluationRepository(path)

    assert reopened.get(evaluation.id) == evaluation
    assert len(reopened.list()) == 1


def test_sqlite_repository_round_trips_full_run(tmp_path) -> None:
    repository = SQLiteEvaluationRepository(
        str(tmp_path / "evaluations.db"),
    )

    dataset = Dataset(
        name="capital-test",
        items=[
            DatasetItem(
                input="What is the capital of France?",
                expected_output="Paris",
                metadata={"difficulty": "easy"},
            ),
        ],
    )

    evaluation = EvaluationRun(
        dataset_id=dataset.id,
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match", "llm_judge"],
        judge_model_name="judge-model",
        status=EvaluationStatus.FAILED,
        results=[
            EvaluationResult(
                item_id=dataset.items[0].id,
                generated_output="Paris",
                metrics=[
                    MetricResult(
                        metric_name="exact_match",
                        score=1.0,
                        passed=True,
                    ),
                ],
                overall_score=1.0,
            ),
        ],
        completed_at=datetime.now(UTC),
        error="Invalid LLM judge response: MAYBE",
    )

    repository.save(evaluation)

    assert repository.get(evaluation.id) == evaluation


def test_sqlite_repository_creates_missing_directory(tmp_path) -> None:
    path = tmp_path / "nested" / "dir" / "evaluations.db"

    SQLiteEvaluationRepository(str(path))

    assert path.exists()


def test_sqlite_repository_handles_concurrent_saves(tmp_path) -> None:
    repository = SQLiteEvaluationRepository(
        str(tmp_path / "evaluations.db"),
    )
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(25):
                evaluation = create_evaluation_run()
                repository.save(evaluation)
                repository.get(evaluation.id)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []
    assert len(repository.list()) == 100
