import pytest

from app.errors import InvalidJudgeResponseError, ModelUnavailableError
from app.evaluators.exact_match import ExactMatchEvaluator
from app.evaluators.registry import EvaluatorRegistry
from app.models import (
    Dataset,
    DatasetItem,
    EvaluationErrorCode,
    EvaluationResult,
    EvaluationRun,
    EvaluationStatus,
    GenerationOptions,
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
        options: GenerationOptions | None = None,
    ) -> list[str]:
        return ["Paris"]


class FailingModelClient(ModelClient):
    def generate(
        self,
        model_name: str,
        inputs: list[str],
        options: GenerationOptions | None = None,
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
        options: GenerationOptions | None = None,
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
        options: GenerationOptions | None = None,
    ) -> list[str]:
        return self._outputs


class InvalidJudgementEvaluator(ExactMatchEvaluator):
    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
        input_text: str | None = None,
    ):
        raise InvalidJudgeResponseError("Invalid LLM judge response: MAYBE")


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
    assert result.error_code == EvaluationErrorCode.INTERNAL_ERROR
    assert result.error == "An internal error occurred while running the evaluation."


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
    assert result.error_code is None


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
    assert result.error_code == EvaluationErrorCode.INVALID_JUDGE_RESPONSE
    assert result.error == ("The LLM judge returned a response that was not YES or NO.")
    assert "MAYBE" not in result.error
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
    assert result.error_code == EvaluationErrorCode.INVALID_MODEL_OUTPUT
    assert result.error == "The model returned an invalid output."
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
    assert result.error_code == EvaluationErrorCode.INVALID_MODEL_OUTPUT
    assert result.error == "The model returned an invalid output."


def test_runner_does_not_expose_raw_error_messages() -> None:
    class VerboseFailingModelClient(ModelClient):
        def generate(
            self,
            model_name: str,
            inputs: list[str],
            options: GenerationOptions | None = None,
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
    assert result.error_code == EvaluationErrorCode.INTERNAL_ERROR
    assert result.error == "An internal error occurred while running the evaluation."


def test_create_run_stores_pending_run_without_generating() -> None:
    model_client = RecordingModelClient()
    repository = EvaluationRepository()

    runner = create_runner(
        model_client=model_client,
        repository=repository,
    )

    run = runner.create_run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert run.status == EvaluationStatus.PENDING
    assert repository.get(run.id) is run
    assert model_client.calls == 0


def test_execute_completes_created_run() -> None:
    runner = create_runner(
        model_client=FakeModelClient(),
    )

    dataset = create_dataset()

    run = runner.create_run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )

    result = runner.execute(
        evaluation_run=run,
        dataset=dataset,
        metrics=["exact_match"],
    )

    assert result is run
    assert result.status == EvaluationStatus.COMPLETED
    assert result.completed_at is not None
    assert len(result.results) == 1


def test_execute_does_not_raise_when_engine_fails() -> None:
    runner = create_runner(
        model_client=FakeModelClient(),
        engine=FailingEvaluationEngine(),
    )

    dataset = create_dataset()

    run = runner.create_run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )

    result = runner.execute(
        evaluation_run=run,
        dataset=dataset,
        metrics=["exact_match"],
    )

    assert result.status == EvaluationStatus.FAILED
    assert result.error_code == EvaluationErrorCode.INTERNAL_ERROR
    assert result.error == "An internal error occurred while running the evaluation."


def test_create_run_stores_dataset_and_settings() -> None:
    registry = EvaluatorRegistry()
    registry.register("exact_match", ExactMatchEvaluator())
    registry.register("llm_judge", ExactMatchEvaluator())

    runner = EvaluationRunner(
        engine=EvaluationEngine(registry),
        model_client=FakeModelClient(),
        judge_model_name="judge-model",
    )

    dataset = create_dataset()

    run = runner.create_run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match", "llm_judge"],
    )

    assert run.dataset == dataset
    assert run.metrics == ["exact_match", "llm_judge"]
    assert run.judge_model_name == "judge-model"


def test_create_run_omits_judge_model_without_llm_judge() -> None:
    registry = EvaluatorRegistry()
    registry.register("exact_match", ExactMatchEvaluator())

    runner = EvaluationRunner(
        engine=EvaluationEngine(registry),
        model_client=FakeModelClient(),
        judge_model_name="judge-model",
    )

    run = runner.create_run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert run.judge_model_name is None


def test_fail_interrupted_runs_marks_unfinished_runs_failed() -> None:
    repository = EvaluationRepository()

    runner = create_runner(
        model_client=FakeModelClient(),
        repository=repository,
    )

    dataset = create_dataset()

    pending = runner.create_run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )

    running = runner.create_run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )
    running.status = EvaluationStatus.RUNNING
    repository.save(running)

    completed = runner.run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )

    updated = runner.fail_interrupted_runs()

    assert updated == 2

    for run in (pending, running):
        stored = repository.get(run.id)

        assert stored.status == EvaluationStatus.FAILED
        assert stored.error == "Interrupted by application restart"
        assert stored.error_code == EvaluationErrorCode.INTERRUPTED
        assert stored.completed_at is not None

    assert repository.get(completed.id).status == EvaluationStatus.COMPLETED
    assert repository.get(completed.id).error is None


def test_fail_interrupted_runs_returns_zero_when_nothing_to_do() -> None:
    runner = create_runner(
        model_client=FakeModelClient(),
    )

    assert runner.fail_interrupted_runs() == 0


def test_execute_fails_run_without_model_client() -> None:
    registry = EvaluatorRegistry()
    registry.register("exact_match", ExactMatchEvaluator())

    runner = EvaluationRunner(
        engine=EvaluationEngine(registry),
    )

    dataset = create_dataset()

    run = EvaluationRun(
        dataset_id=dataset.id,
        model_name="llama3",
    )

    result = runner.execute(
        evaluation_run=run,
        dataset=dataset,
        metrics=["exact_match"],
    )

    assert result.status == EvaluationStatus.FAILED
    assert result.error_code == EvaluationErrorCode.INTERNAL_ERROR
    assert result.error == "An internal error occurred while running the evaluation."


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
        return ["Paris"]


def create_options_runner(
    model_client: ModelClient,
    defaults: GenerationOptions | None = None,
) -> EvaluationRunner:
    registry = EvaluatorRegistry()
    registry.register("exact_match", ExactMatchEvaluator())

    return EvaluationRunner(
        engine=EvaluationEngine(registry),
        model_client=model_client,
        default_generation_options=defaults,
    )


def test_create_run_stores_default_generation_options() -> None:
    runner = create_options_runner(
        model_client=FakeModelClient(),
        defaults=GenerationOptions(temperature=0.0, seed=7),
    )

    run = runner.create_run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert run.generation_options == GenerationOptions(temperature=0.0, seed=7)


def test_create_run_merges_request_options_with_defaults() -> None:
    runner = create_options_runner(
        model_client=FakeModelClient(),
        defaults=GenerationOptions(temperature=0.0, seed=7),
    )

    run = runner.create_run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
        generation_options=GenerationOptions(seed=42),
    )

    assert run.generation_options == GenerationOptions(temperature=0.0, seed=42)


def test_create_run_does_not_share_default_options_between_runs() -> None:
    defaults = GenerationOptions(temperature=0.0)

    runner = create_options_runner(
        model_client=FakeModelClient(),
        defaults=defaults,
    )

    run = runner.create_run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
    )

    run.generation_options.seed = 99

    assert defaults.seed is None


def test_execute_sends_stored_generation_options_to_model() -> None:
    model_client = OptionsCaptureModelClient()

    runner = create_options_runner(
        model_client=model_client,
        defaults=GenerationOptions(temperature=0.0),
    )

    result = runner.run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
        generation_options=GenerationOptions(temperature=0.8, seed=42),
    )

    assert result.status == EvaluationStatus.COMPLETED
    assert model_client.options == GenerationOptions(temperature=0.8, seed=42)
    assert model_client.options == result.generation_options


def test_runner_records_public_message_for_evaluation_errors() -> None:
    class UnavailableModelClient(ModelClient):
        def generate(
            self,
            model_name: str,
            inputs: list[str],
            options: GenerationOptions | None = None,
        ) -> list[str]:
            raise ModelUnavailableError(
                "[WinError 10061] connection refused: http://ollama:11434"
            )

    runner = create_runner(
        model_client=UnavailableModelClient(),
    )

    result = runner.run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
    )

    assert result.status == EvaluationStatus.FAILED
    assert result.error_code == EvaluationErrorCode.MODEL_UNAVAILABLE
    assert result.error == "Could not connect to the model provider."
    assert "ollama:11434" not in result.error


def test_runner_rejects_empty_provided_outputs_as_invalid_output() -> None:
    runner = create_runner(
        model_client=FakeModelClient(),
    )

    result = runner.run(
        dataset=create_dataset(),
        model_name="llama3",
        metrics=["exact_match"],
        generated_outputs=["   "],
    )

    assert result.error_code == EvaluationErrorCode.INVALID_MODEL_OUTPUT
