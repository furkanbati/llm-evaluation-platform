from uuid import uuid4

from app.models import EvaluationRun, EvaluationStatus
from app.storage.evaluation_repository import EvaluationRepository


def create_evaluation_run(
    model_name: str = "llama3",
    status: EvaluationStatus = EvaluationStatus.PENDING,
) -> EvaluationRun:
    return EvaluationRun(
        dataset_id=uuid4(),
        model_name=model_name,
        status=status,
    )


def test_save_stores_evaluation() -> None:
    repository = EvaluationRepository()
    evaluation = create_evaluation_run()

    repository.save(evaluation)

    assert repository.get(evaluation.id) == evaluation


def test_save_returns_saved_evaluation() -> None:
    repository = EvaluationRepository()
    evaluation = create_evaluation_run()

    result = repository.save(evaluation)

    assert result is evaluation


def test_get_returns_none_for_unknown_id() -> None:
    repository = EvaluationRepository()

    result = repository.get(uuid4())

    assert result is None


def test_save_updates_existing_evaluation() -> None:
    repository = EvaluationRepository()
    evaluation = create_evaluation_run()

    repository.save(evaluation)

    evaluation.model_name = "llama3.2"

    repository.save(evaluation)

    result = repository.get(evaluation.id)

    assert result is evaluation
    assert result.model_name == "llama3.2"


def test_list_returns_saved_evaluations() -> None:
    repository = EvaluationRepository()

    first = create_evaluation_run()
    second = create_evaluation_run()

    repository.save(first)
    repository.save(second)

    result = repository.list()

    assert result == [first, second]


def test_list_returns_empty_list_when_repository_is_empty() -> None:
    repository = EvaluationRepository()

    result = repository.list()

    assert result == []


def test_list_filters_by_status() -> None:
    repository = EvaluationRepository()

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


def test_list_filters_by_model_name() -> None:
    repository = EvaluationRepository()

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


def test_list_filters_by_status_and_model_name() -> None:
    repository = EvaluationRepository()

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


def test_list_returns_empty_list_when_status_has_no_matches() -> None:
    repository = EvaluationRepository()

    evaluation = create_evaluation_run(
        status=EvaluationStatus.COMPLETED,
    )

    repository.save(evaluation)

    result = repository.list(
        status=EvaluationStatus.FAILED,
    )

    assert result == []


def test_list_returns_empty_list_when_model_name_has_no_matches() -> None:
    repository = EvaluationRepository()

    evaluation = create_evaluation_run(
        model_name="llama3",
    )

    repository.save(evaluation)

    result = repository.list(
        model_name="llama3.2",
    )

    assert result == []


def test_delete_removes_saved_evaluation() -> None:
    repository = EvaluationRepository()

    evaluation = create_evaluation_run()

    repository.save(evaluation)

    deleted = repository.delete(
        evaluation.id,
    )

    assert deleted is True
    assert repository.get(
        evaluation.id,
    ) is None


def test_delete_returns_false_for_unknown_id() -> None:
    repository = EvaluationRepository()

    result = repository.delete(
        uuid4(),
    )

    assert result is False