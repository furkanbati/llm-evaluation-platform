"""Item-by-item execution: partial results, retries, early stops, resume."""

from collections.abc import Callable

import pytest

from app.errors import (
    InvalidJudgeResponseError,
    ModelNotFoundError,
    ModelTimeoutError,
    ModelUnavailableError,
)
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
    MetricResult,
)
from app.services.evaluation_engine import EvaluationEngine
from app.services.evaluation_runner import EvaluationRunner
from app.services.model_client import ModelClient
from app.storage.base import BaseEvaluationRepository
from app.storage.evaluation_repository import EvaluationRepository


class ScriptedModelClient(ModelClient):
    """Answers each input with the input itself, unless a step says otherwise.

    `script` maps an input to a list of outcomes used one per call: an
    exception is raised, a string is returned.
    """

    def __init__(
        self,
        script: dict[str, list[Exception | str]] | None = None,
        on_call: Callable[[str], None] | None = None,
    ) -> None:
        self._script = script or {}
        self._on_call = on_call
        self.calls: list[str] = []

    def generate(
        self,
        model_name: str,
        inputs: list[str],
        options: GenerationOptions | None = None,
    ) -> list[str]:
        assert len(inputs) == 1
        [input_text] = inputs
        self.calls.append(input_text)

        if self._on_call is not None:
            self._on_call(input_text)

        steps = self._script.get(input_text)

        if steps:
            outcome = steps.pop(0)

            if isinstance(outcome, Exception):
                raise outcome

            return [outcome]

        return [input_text]


class FlakyJudgeEvaluator(ExactMatchEvaluator):
    """Raises an invalid judge response a number of times, then scores."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
        input_text: str | None = None,
    ) -> MetricResult:
        self.calls += 1

        if self.calls <= self.failures:
            raise InvalidJudgeResponseError("Invalid LLM judge response: MAYBE")

        return super().evaluate(expected_output, generated_output, input_text)


def create_dataset(*inputs: str) -> Dataset:
    # Every expected output equals its input, so default answers pass.
    return Dataset(
        name="items",
        items=[
            DatasetItem(input=input_text, expected_output=input_text)
            for input_text in inputs
        ],
    )


def create_runner(
    model_client: ModelClient,
    repository: BaseEvaluationRepository | None = None,
    evaluator: ExactMatchEvaluator | None = None,
    sleeps: list[float] | None = None,
    **options,
) -> EvaluationRunner:
    registry = EvaluatorRegistry()
    registry.register("exact_match", evaluator or ExactMatchEvaluator())

    return EvaluationRunner(
        engine=EvaluationEngine(registry),
        model_client=model_client,
        repository=repository,
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
        **options,
    )


def run(runner: EvaluationRunner, dataset: Dataset) -> EvaluationRun:
    return runner.run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )


def test_failed_item_does_not_discard_other_results() -> None:
    repository = EvaluationRepository()
    client = ScriptedModelClient({"b": [RuntimeError("boom")]})
    runner = create_runner(client, repository=repository)

    result = run(runner, create_dataset("a", "b", "c"))

    assert result.status == EvaluationStatus.COMPLETED_WITH_ERRORS
    assert result.error_code is None
    assert [r.generated_output for r in result.results] == ["a", "c"]
    assert len(result.item_errors) == 1
    assert result.item_errors[0].error_code == EvaluationErrorCode.INTERNAL_ERROR
    assert result.item_errors[0].attempts == 1

    stored = repository.get(result.id)

    assert stored == result


def test_run_fails_when_no_item_could_be_scored() -> None:
    client = ScriptedModelClient(
        {
            "a": [ModelUnavailableError("down")],
            "b": [ModelUnavailableError("down")],
        }
    )
    runner = create_runner(client)

    result = run(runner, create_dataset("a", "b"))

    assert result.status == EvaluationStatus.FAILED
    assert result.error_code == EvaluationErrorCode.MODEL_UNAVAILABLE
    assert result.error == "Could not connect to the model provider."
    assert result.results == []
    assert len(result.item_errors) == 2


def test_retryable_errors_are_retried_with_exponential_backoff() -> None:
    sleeps: list[float] = []
    client = ScriptedModelClient(
        {"a": [ModelTimeoutError("slow"), ModelUnavailableError("down")]}
    )
    runner = create_runner(
        client,
        sleeps=sleeps,
        max_attempts=3,
        retry_backoff_seconds=0.5,
    )

    result = run(runner, create_dataset("a"))

    assert result.status == EvaluationStatus.COMPLETED
    assert client.calls == ["a", "a", "a"]
    assert sleeps == [0.5, 1.0]


def test_item_error_records_attempts_when_retries_run_out() -> None:
    client = ScriptedModelClient({"a": [ModelTimeoutError("slow")] * 3})
    runner = create_runner(client, max_attempts=3)

    result = run(runner, create_dataset("a", "b"))

    assert result.status == EvaluationStatus.COMPLETED_WITH_ERRORS
    assert client.calls == ["a", "a", "a", "b"]
    [item_error] = result.item_errors
    assert item_error.error_code == EvaluationErrorCode.MODEL_TIMEOUT
    assert item_error.attempts == 3
    assert item_error.generated_output is None


def test_non_retryable_errors_are_not_retried() -> None:
    client = ScriptedModelClient({"a": [RuntimeError("bug")]})
    runner = create_runner(client, max_attempts=3)

    run(runner, create_dataset("a"))

    assert client.calls == ["a"]


def test_empty_model_output_is_retried() -> None:
    client = ScriptedModelClient({"a": ["  "]})
    runner = create_runner(client, max_attempts=2)

    result = run(runner, create_dataset("a"))

    assert result.status == EvaluationStatus.COMPLETED
    assert client.calls == ["a", "a"]


def test_judge_retries_do_not_generate_again() -> None:
    client = ScriptedModelClient()
    judge = FlakyJudgeEvaluator(failures=1)
    runner = create_runner(client, evaluator=judge, max_attempts=2)

    result = run(runner, create_dataset("a"))

    assert result.status == EvaluationStatus.COMPLETED
    assert client.calls == ["a"]
    assert judge.calls == 2


def test_invalid_judge_response_fails_only_that_item() -> None:
    client = ScriptedModelClient()
    judge = FlakyJudgeEvaluator(failures=2)
    runner = create_runner(client, evaluator=judge, max_attempts=2)

    result = run(runner, create_dataset("a", "b"))

    assert result.status == EvaluationStatus.COMPLETED_WITH_ERRORS
    [item_error] = result.item_errors
    assert item_error.error_code == EvaluationErrorCode.INVALID_JUDGE_RESPONSE
    assert item_error.attempts == 2
    # The answer was generated, so it is kept for debugging the judge.
    assert item_error.generated_output == "a"
    assert [r.generated_output for r in result.results] == ["b"]


def test_fatal_error_stops_the_run_and_keeps_earlier_results() -> None:
    client = ScriptedModelClient({"b": [ModelNotFoundError("llama3")]})
    runner = create_runner(client, max_attempts=3)

    result = run(runner, create_dataset("a", "b", "c"))

    assert result.status == EvaluationStatus.FAILED
    assert result.error_code == EvaluationErrorCode.MODEL_NOT_FOUND
    assert client.calls == ["a", "b"]
    assert [r.generated_output for r in result.results] == ["a"]
    assert result.item_errors == []


def test_run_stops_after_too_many_consecutive_failures() -> None:
    error = ModelUnavailableError("down")
    client = ScriptedModelClient({name: [error] for name in "abcde"})
    runner = create_runner(client, max_consecutive_failures=2)

    result = run(runner, create_dataset(*"abcde"))

    assert result.status == EvaluationStatus.FAILED
    assert result.error_code == EvaluationErrorCode.MODEL_UNAVAILABLE
    assert result.error is not None
    assert "stopped after 2 items in a row failed" in result.error
    assert client.calls == ["a", "b"]
    assert len(result.item_errors) == 2


def test_successful_item_resets_consecutive_failures() -> None:
    error = RuntimeError("boom")
    client = ScriptedModelClient({"a": [error], "c": [error], "e": [error]})
    runner = create_runner(client, max_consecutive_failures=2)

    result = run(runner, create_dataset(*"abcde"))

    assert result.status == EvaluationStatus.COMPLETED_WITH_ERRORS
    assert len(result.results) == 2
    assert len(result.item_errors) == 3


def test_each_item_is_stored_as_soon_as_it_finishes() -> None:
    repository = EvaluationRepository()
    stored_counts: list[int] = []
    run_ids: list = []

    def record(_: str) -> None:
        if run_ids:
            stored = repository.get(run_ids[0])
            assert stored is not None
            stored_counts.append(len(stored.results))

    client = ScriptedModelClient(on_call=record)
    runner = create_runner(client, repository=repository)
    dataset = create_dataset("a", "b", "c")

    evaluation_run = runner.create_run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )
    run_ids.append(evaluation_run.id)

    runner.execute(
        evaluation_run=evaluation_run,
        dataset=dataset,
        metrics=["exact_match"],
    )

    assert stored_counts == [0, 1, 2]


def test_execute_skips_items_that_already_have_an_outcome() -> None:
    repository = EvaluationRepository()
    client = ScriptedModelClient()
    runner = create_runner(client, repository=repository)
    dataset = create_dataset("a", "b", "c")

    evaluation_run = runner.create_run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )

    # Simulate a process that scored the second item before it stopped.
    repository.save_result(
        evaluation_run.id,
        1,
        EvaluationResult(
            item_id=dataset.items[1].id,
            generated_output="b",
            metrics=[MetricResult(metric_name="exact_match", score=1.0, passed=True)],
            overall_score=1.0,
            passed=True,
        ),
    )

    loaded = repository.get(evaluation_run.id)
    assert loaded is not None

    result = runner.execute(
        evaluation_run=loaded,
        dataset=dataset,
        metrics=["exact_match"],
    )

    assert client.calls == ["a", "c"]
    assert result.status == EvaluationStatus.COMPLETED
    assert [r.generated_output for r in result.results] == ["a", "b", "c"]
    assert repository.get(evaluation_run.id) == result


def test_run_deleted_during_execution_is_not_recreated() -> None:
    repository = EvaluationRepository()
    run_ids: list = []

    def delete_on_second_item(input_text: str) -> None:
        if input_text == "b":
            repository.delete(run_ids[0])

    client = ScriptedModelClient(on_call=delete_on_second_item)
    runner = create_runner(client, repository=repository)
    dataset = create_dataset("a", "b", "c")

    evaluation_run = runner.create_run(
        dataset=dataset,
        model_name="llama3",
        metrics=["exact_match"],
    )
    run_ids.append(evaluation_run.id)

    runner.execute(
        evaluation_run=evaluation_run,
        dataset=dataset,
        metrics=["exact_match"],
    )

    assert client.calls == ["a", "b"]
    assert repository.get(evaluation_run.id) is None
    assert repository.list() == []


@pytest.mark.parametrize(
    "options",
    [
        {"max_attempts": 0},
        {"retry_backoff_seconds": -1.0},
        {"max_consecutive_failures": 0},
    ],
)
def test_runner_rejects_invalid_retry_settings(options) -> None:
    with pytest.raises(ValueError):
        create_runner(ScriptedModelClient(), **options)
