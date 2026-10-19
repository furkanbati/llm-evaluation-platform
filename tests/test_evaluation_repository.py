import json
import sqlite3
import threading
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.models import (
    Dataset,
    DatasetItem,
    EvaluationErrorCode,
    EvaluationResult,
    EvaluationRun,
    EvaluationStatus,
    ItemError,
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


def create_result(item_id, output: str = "Paris") -> EvaluationResult:
    return EvaluationResult(
        item_id=item_id,
        generated_output=output,
        metrics=[MetricResult(metric_name="exact_match", score=1.0, passed=True)],
        overall_score=1.0,
        passed=True,
    )


def create_item_error(item_id) -> ItemError:
    return ItemError(
        item_id=item_id,
        error_code=EvaluationErrorCode.MODEL_TIMEOUT,
        error="The model provider did not respond in time.",
        attempts=3,
    )


def create_run_with_dataset(size: int = 3) -> EvaluationRun:
    dataset = Dataset(
        name="items",
        items=[
            DatasetItem(input=f"q{index}", expected_output=f"a{index}")
            for index in range(size)
        ],
    )

    return EvaluationRun(
        dataset_id=dataset.id,
        dataset=dataset,
        model_name="llama3",
    )


def test_items_are_returned_in_dataset_order(repository) -> None:
    run = repository.save(create_run_with_dataset())
    assert run.dataset is not None
    items = run.dataset.items

    assert repository.save_result(run.id, 2, create_result(items[2].id, "c"))
    assert repository.save_item_error(run.id, 1, create_item_error(items[1].id))
    assert repository.save_result(run.id, 0, create_result(items[0].id, "a"))

    stored = repository.get(run.id)

    assert stored is not None
    assert [result.generated_output for result in stored.results] == ["a", "c"]
    assert [error.item_id for error in stored.item_errors] == [items[1].id]


def test_saving_an_item_again_replaces_it(repository) -> None:
    run = repository.save(create_run_with_dataset(1))
    assert run.dataset is not None
    item_id = run.dataset.items[0].id

    repository.save_item_error(run.id, 0, create_item_error(item_id))
    repository.save_result(run.id, 0, create_result(item_id))

    stored = repository.get(run.id)

    assert stored is not None
    assert len(stored.results) == 1
    assert stored.item_errors == []


def test_items_are_not_stored_for_unknown_runs(repository) -> None:
    run_id = uuid4()

    assert not repository.save_result(run_id, 0, create_result(uuid4()))
    assert not repository.save_item_error(run_id, 0, create_item_error(uuid4()))
    assert repository.get(run_id) is None


def test_update_run_keeps_items_and_ignores_deleted_runs(repository) -> None:
    run = repository.save(create_run_with_dataset(1))
    assert run.dataset is not None
    repository.save_result(run.id, 0, create_result(run.dataset.items[0].id))

    run.status = EvaluationStatus.COMPLETED
    assert repository.update_run(run)

    stored = repository.get(run.id)
    assert stored is not None
    assert stored.status == EvaluationStatus.COMPLETED
    assert len(stored.results) == 1

    assert repository.delete(run.id)
    assert not repository.update_run(run)
    assert repository.get(run.id) is None


def test_delete_removes_items(repository) -> None:
    run = repository.save(create_run_with_dataset(1))
    assert run.dataset is not None
    repository.save_result(run.id, 0, create_result(run.dataset.items[0].id))

    repository.delete(run.id)
    repository.save(run.model_copy(update={"results": []}))

    stored = repository.get(run.id)

    assert stored is not None
    assert stored.results == []


def test_list_includes_items(repository) -> None:
    first = repository.save(create_run_with_dataset(2))
    second = repository.save(create_run_with_dataset(1))
    assert first.dataset is not None

    repository.save_result(first.id, 1, create_result(first.dataset.items[1].id))
    repository.save_item_error(
        first.id,
        0,
        create_item_error(first.dataset.items[0].id),
    )

    runs = {run.id: run for run in repository.list()}

    assert len(runs[first.id].results) == 1
    assert len(runs[first.id].item_errors) == 1
    assert runs[second.id].results == []


def test_changing_a_returned_run_does_not_change_the_stored_one(
    repository,
) -> None:
    run = repository.save(create_run_with_dataset(1))
    assert run.dataset is not None

    loaded = repository.get(run.id)
    assert loaded is not None

    loaded.status = EvaluationStatus.FAILED
    loaded.results.append(create_result(run.dataset.items[0].id))

    stored = repository.get(run.id)

    assert stored is not None
    assert stored.status == EvaluationStatus.PENDING
    assert stored.results == []


def test_sqlite_repository_migrates_results_stored_in_run_json(tmp_path) -> None:
    path = tmp_path / "evaluations.db"
    run = create_run_with_dataset(2)
    assert run.dataset is not None
    run.results = [create_result(run.dataset.items[1].id, "b")]
    run.item_errors = [create_item_error(run.dataset.items[0].id)]
    run.status = EvaluationStatus.COMPLETED_WITH_ERRORS

    # Write the run the way databases created before per-item storage did.
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE evaluation_runs (id TEXT PRIMARY KEY, status TEXT NOT NULL, "
        "model_name TEXT NOT NULL, created_at TEXT NOT NULL, data TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO evaluation_runs VALUES (?, ?, ?, ?, ?)",
        (
            str(run.id),
            run.status.value,
            run.model_name,
            run.created_at.isoformat(),
            run.model_dump_json(),
        ),
    )
    connection.commit()
    connection.close()

    repository = SQLiteEvaluationRepository(str(path))

    assert repository.get(run.id) == run

    # The run's JSON no longer holds the items, and migrating again is a no-op.
    connection = sqlite3.connect(path)
    [data] = connection.execute("SELECT data FROM evaluation_runs").fetchone()
    connection.close()

    assert "results" not in json.loads(data)
    assert SQLiteEvaluationRepository(str(path)).get(run.id) == run
