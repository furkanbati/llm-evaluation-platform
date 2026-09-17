import pytest

from app.evaluators.exact_match import ExactMatchEvaluator
from app.evaluators.registry import EvaluatorRegistry
from app.models import (
    Dataset,
    DatasetItem,
    EvaluationRun,
    EvaluationStatus,
)
from app.services.evaluation_engine import EvaluationEngine
from app.services.evaluation_runner import EvaluationRunner
from app.services.model_client import ModelClient


class FakeModelClient(ModelClient):
    def generate(
        self,
        model_name: str,
        inputs: list[str],
    ) -> list[str]:
        return [
            "Paris",
            "Ankara",
        ]


def create_runner(
    model_client: ModelClient | None = None,
) -> EvaluationRunner:
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


def create_dataset() -> Dataset:
    return Dataset(
        name="test dataset",
        items=[
            DatasetItem(
                input="q1",
                expected_output="Paris",
            ),
            DatasetItem(
                input="q2",
                expected_output="Ankara",
            ),
        ],
    )


def test_runner_evaluates_all_dataset_items():
    runner = create_runner()

    dataset = create_dataset()

    run = runner.run(
        dataset=dataset,
        model_name="test-model",
        generated_outputs=[
            "Paris",
            "Ankara",
        ],
        metrics=["exact_match"],
    )

    assert len(run.results) == 2


def test_runner_generates_outputs_with_model_client():
    model_client = FakeModelClient()

    runner = create_runner(
        model_client=model_client,
    )

    dataset = create_dataset()

    run = runner.run(
        dataset=dataset,
        model_name="test-model",
        metrics=["exact_match"],
    )

    assert [result.generated_output for result in run.results] == [
        "Paris",
        "Ankara",
    ]


def test_runner_returns_evaluation_run():
    runner = create_runner()

    dataset = create_dataset()

    run = runner.run(
        dataset=dataset,
        model_name="test-model",
        generated_outputs=[
            "Paris",
            "Ankara",
        ],
        metrics=["exact_match"],
    )

    assert isinstance(
        run,
        EvaluationRun,
    )


def test_runner_populates_evaluation_run_fields():
    runner = create_runner()

    dataset = create_dataset()

    run = runner.run(
        dataset=dataset,
        model_name="test-model",
        generated_outputs=[
            "Paris",
            "Ankara",
        ],
        metrics=["exact_match"],
    )

    assert run.dataset_id == dataset.id
    assert run.model_name == "test-model"
    assert run.status == EvaluationStatus.COMPLETED


def test_runner_raises_error_when_output_count_mismatch():
    runner = create_runner()

    dataset = create_dataset()

    with pytest.raises(ValueError):
        runner.run(
            dataset=dataset,
            model_name="test-model",
            generated_outputs=[
                "Paris",
            ],
            metrics=["exact_match"],
        )


def test_runner_sets_created_at_before_completed_at():
    runner = create_runner()

    dataset = create_dataset()

    run = runner.run(
        dataset=dataset,
        model_name="test-model",
        generated_outputs=[
            "Paris",
            "Ankara",
        ],
        metrics=["exact_match"],
    )

    assert run.completed_at is not None
    assert run.created_at <= run.completed_at

