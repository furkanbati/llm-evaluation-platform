import pytest

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
    def validate_metrics(
        self,
        metrics: list[str],
    ) -> None:
        pass

    def evaluate(
        self,
        item: DatasetItem,
        generated_output: str,
        metrics: list[str],
    ) -> EvaluationResult:
        raise RuntimeError("Evaluation failed")


class RecordingEvaluationRepository(EvaluationRepository):
    def __init__(self) -> None:
        super().__init__()
        self.statuses: list[EvaluationStatus] = []

    def save(self, evaluation):
        self.statuses.append(evaluation.status)
        return super().save(evaluation)


def create_runner(
    model_client: ModelClient,
    engine: EvaluationEngine | FailingEvaluationEngine | None = None,
    repository: EvaluationRepository | None = None,
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
        repository=repository,
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


def test_runner_transitions_from_pending_to_running() -> None:
    repository = RecordingEvaluationRepository()

    runner = create_runner(
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

    assert repository.statuses[0] == EvaluationStatus.PENDING
    assert repository.statuses[1] == EvaluationStatus.RUNNING
    assert result.status == EvaluationStatus.COMPLETED

class RecordingModelClient(ModelClient):
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        model_name: str,
        inputs: list[str],
    ) -> list[str]:
        self.calls += 1
        return ["Paris"]


class StaticModelClient(ModelClient):
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs

    def generate(
        self,
        model_name: str,
        inputs: list[str],
    ) -> list[str]:
        return self._outputs


class InvalidJudgementEvaluator(ExactMatchEvaluator):
    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
        input_text: str | None = None,
    ):
        raise ValueError("Invalid LLM judge response: MAYBE")


def create_dataset() -> Dataset:
    return Dataset(
        name="capital-test",
        items=[
            DatasetItem(
                input="What is the capital of France?",
                expected_output="Paris",
            ),
        ],
    )


def test_runner_rejects_unknown_metric_before_generation() -> None:
    model_client = RecordingModelClient()
    repository = EvaluationRepository()

    runner = create_runner(
        model_client=model_client,
        repository=repository,
    )

    with pytest.raises(ValueError, match="unknown_metric"):
        runner.run(
            dataset=create_dataset(),
            model_name="llama3",
            metrics=["unknown_metric"],
        )

    assert model_client.calls == 0
    assert repository.list() == []


def test_runner_does_not_store_run_for_mismatched_provided_outputs() -> None:
    repository = EvaluationRepository()

    runner = create_runner(
        model_client=FakeModelClient(),
        repository=repository,
    )

    with pytest.raises(ValueError):
        runner.run(
            dataset=create_dataset(),
            model_name="llama3",
            metrics=["exact_match"],
            generated_outputs=["Paris", "Ankara"],
        )

    assert repository.list() == []


def test_runner_records_error_when_model_generation_fails() -> None:
    runner = create_runner(
        model_client=FailingModelClient(),
    )

    result = runner.run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert result.status == EvaluationStatus.FAILED
    assert result.error == "Model generation failed"


def test_runner_completed_evaluation_has_no_error() -> None:
    runner = create_runner(
        model_client=FakeModelClient(),
    )

    result = runner.run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert result.status == EvaluationStatus.COMPLETED
    assert result.error is None


def test_runner_marks_failed_when_evaluator_raises_value_error() -> None:
    registry = EvaluatorRegistry()
    registry.register(
        "llm_judge",
        InvalidJudgementEvaluator(),
    )

    repository = EvaluationRepository()

    runner = create_runner(
        model_client=FakeModelClient(),
        engine=EvaluationEngine(registry),
        repository=repository,
    )

    result = runner.run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["llm_judge"],
    )

    assert result.status == EvaluationStatus.FAILED
    assert "Invalid LLM judge response" in result.error
    assert repository.get(result.id).status == EvaluationStatus.FAILED


def test_runner_marks_failed_when_model_returns_empty_output() -> None:
    runner = create_runner(
        model_client=StaticModelClient([""]),
    )

    result = runner.run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert result.status == EvaluationStatus.FAILED
    assert result.error is not None
    assert result.completed_at is not None


def test_runner_marks_failed_when_model_returns_wrong_output_count() -> None:
    runner = create_runner(
        model_client=StaticModelClient(["Paris", "Ankara"]),
    )

    result = runner.run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert result.status == EvaluationStatus.FAILED
    assert "2 outputs" in result.error


def test_runner_truncates_long_error_messages() -> None:
    class VerboseFailingModelClient(ModelClient):
        def generate(
            self,
            model_name: str,
            inputs: list[str],
        ) -> list[str]:
            raise RuntimeError("x" * 5_000)

    runner = create_runner(
        model_client=VerboseFailingModelClient(),
    )

    result = runner.run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert result.status == EvaluationStatus.FAILED
    assert len(result.error) == 1_000
