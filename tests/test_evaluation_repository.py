from uuid import uuid4

from app.models import EvaluationRun
from app.storage.evaluation_repository import EvaluationRepository


def create_evaluation_run() -> EvaluationRun:
    return EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
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

