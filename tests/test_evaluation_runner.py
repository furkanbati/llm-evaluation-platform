from app.evaluators.exact_match import ExactMatchEvaluator
from app.evaluators.registry import EvaluatorRegistry
from app.models import (
    Dataset,
    DatasetItem,
    EvaluationStatus,
    EvaluationResult,
)
from app.services.evaluation_engine import EvaluationEngine
from app.services.evaluation_runner import EvaluationRunner
from app.services.model_client import ModelClient
from app.storage.evaluation_repository import EvaluationRepository

class FakeModelClient(ModelClient):
    def generate(
        self,
        model_name: str,
        inputs: list[str],
    ) -> list[str]:
        return ["Paris"]


class FailingModelClient(ModelClient):
    def generate(
        self,
        model_name: str,
        inputs: list[str],
    ) -> list[str]:
        raise RuntimeError("Model generation failed")


class FailingEvaluationEngine:
    def evaluate(
        self,
        item: DatasetItem,
        generated_output: str,
        metrics: list[str],
    ) -> EvaluationResult:
        raise RuntimeError("Evaluation failed")


def create_runner(
    model_client: ModelClient,
    engine: EvaluationEngine | FailingEvaluationEngine | None = None,
) -> EvaluationRunner:
    if engine is None:
        registry = EvaluatorRegistry()
        registry.register(
            "exact_match",
            ExactMatchEvaluator(),
        )

        engine = EvaluationEngine(registry)

    return EvaluationRunner(
        engine=engine,
        model_client=model_client,
    )


def test_runner_completes_successful_evaluation() -> None:
    runner = create_runner(
        model_client=FakeModelClient(),
    )

    dataset = Dataset(
        name="capital-test",
        items=[
            DatasetItem(
                input="What is the capital of France?",
                expected_output="Paris",
            ),
        ],
    )

    result = runner.run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert result.status == EvaluationStatus.COMPLETED
    assert result.created_at is not None
    assert result.completed_at is not None
    assert result.created_at <= result.completed_at
    assert len(result.results) == 1
    assert result.results[0].generated_output == "Paris"
    assert result.results[0].overall_score == 1.0


def test_runner_marks_evaluation_as_failed_when_model_generation_fails() -> None:
    runner = create_runner(
        model_client=FailingModelClient(),
    )

    dataset = Dataset(
        name="capital-test",
        items=[
            DatasetItem(
                input="What is the capital of France?",
                expected_output="Paris",
            ),
        ],
    )

    result = runner.run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert result.status == EvaluationStatus.FAILED
    assert result.created_at is not None
    assert result.completed_at is not None
    assert result.created_at <= result.completed_at
    assert result.results == []


def test_runner_marks_evaluation_as_failed_when_engine_fails() -> None:
    runner = create_runner(
        model_client=FakeModelClient(),
        engine=FailingEvaluationEngine(),
    )

    dataset = Dataset(
        name="capital-test",
        items=[
            DatasetItem(
                input="What is the capital of France?",
                expected_output="Paris",
            ),
        ],
    )

    result = runner.run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert result.status == EvaluationStatus.FAILED
    assert result.created_at is not None
    assert result.completed_at is not None
    assert result.created_at <= result.completed_at
    assert result.results == []

def test_runner_saves_completed_evaluation_to_repository() -> None:
    runner = create_runner(
        model_client=FakeModelClient(),
    )

    dataset = Dataset(
        name="capital-test",
        items=[
            DatasetItem(
                input="What is the capital of France?",
                expected_output="Paris",
            ),
        ],
    )

    result = runner.run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )

    saved = runner._repository.get(result.id)

    assert saved is result
    assert saved.status == EvaluationStatus.COMPLETED


def test_runner_saves_failed_evaluation_to_repository() -> None:
    runner = create_runner(
        model_client=FailingModelClient(),
    )

    dataset = Dataset(
        name="capital-test",
        items=[
            DatasetItem(
                input="What is the capital of France?",
                expected_output="Paris",
            ),
        ],
    )

    result = runner.run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )

    saved = runner._repository.get(result.id)

    assert saved is result
    assert saved.status == EvaluationStatus.FAILED

def test_runner_uses_provided_repository() -> None:
    repository = EvaluationRepository()

    runner = create_runner(
        model_client=FakeModelClient(),
    )

    runner = EvaluationRunner(
        engine=runner._engine,
        model_client=FakeModelClient(),
        repository=repository,
    )

    dataset = Dataset(
        name="capital-test",
        items=[
            DatasetItem(
                input="What is the capital of France?",
                expected_output="Paris",
            ),
        ],
    )

    result = runner.run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert repository.get(result.id) is result

