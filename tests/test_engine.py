from uuid import UUID

from app.evaluators.exact_match import ExactMatchEvaluator
from app.evaluators.registry import EvaluatorRegistry
from app.models import DatasetItem
from app.services.evaluation_engine import EvaluationEngine


def test_engine_runs_exact_match() -> None:
    registry = EvaluatorRegistry()

    registry.register(
        "exact_match",
        ExactMatchEvaluator(),
    )

    engine = EvaluationEngine(registry)

    item = DatasetItem(
        input="What is the capital of France?",
        expected_output="Paris",
    )

    result = engine.evaluate(
        item=item,
        generated_output="Paris",
        metrics=["exact_match"],
    )

    assert result.generated_output == "Paris"
    assert result.overall_score == 1.0
    assert len(result.metrics) == 1


def test_engine_returns_metric_results() -> None:
    registry = EvaluatorRegistry()

    registry.register(
        "exact_match",
        ExactMatchEvaluator(),
    )

    engine = EvaluationEngine(registry)

    item = DatasetItem(
        input="Question",
        expected_output="Paris",
    )

    result = engine.evaluate(
        item=item,
        generated_output="London",
        metrics=["exact_match"],
    )

    metric = result.metrics[0]

    assert metric.metric_name == "exact_match"
    assert metric.score == 0.0
    assert metric.passed is False


def test_engine_calculates_overall_score() -> None:
    registry = EvaluatorRegistry()

    registry.register(
        "exact_match",
        ExactMatchEvaluator(),
    )

    engine = EvaluationEngine(registry)

    item = DatasetItem(
        input="Question",
        expected_output="Paris",
    )

    result = engine.evaluate(
        item=item,
        generated_output="Paris",
        metrics=["exact_match"],
    )

    assert result.overall_score == 1.0