from app.evaluators.base import Evaluator
from app.evaluators.exact_match import ExactMatchEvaluator
from app.evaluators.registry import EvaluatorRegistry
from app.models import DatasetItem, MetricResult
from app.services.evaluation_engine import EvaluationEngine


class FixedScoreEvaluator(Evaluator):
    def __init__(
        self,
        metric_name: str,
        score: float,
    ) -> None:
        self._metric_name = metric_name
        self._score = score

    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
    ) -> MetricResult:
        return MetricResult(
            metric_name=self._metric_name,
            score=self._score,
            passed=self._score == 1.0,
        )


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


def test_engine_calculates_average_of_multiple_metrics() -> None:
    registry = EvaluatorRegistry()

    registry.register(
        "metric_one",
        FixedScoreEvaluator(
            metric_name="metric_one",
            score=1.0,
        ),
    )
    registry.register(
        "metric_two",
        FixedScoreEvaluator(
            metric_name="metric_two",
            score=0.5,
        ),
    )

    engine = EvaluationEngine(registry)

    item = DatasetItem(
        input="Question",
        expected_output="Answer",
    )

    result = engine.evaluate(
        item=item,
        generated_output="Generated answer",
        metrics=["metric_one", "metric_two"],
    )

    assert len(result.metrics) == 2
    assert result.metrics[0].metric_name == "metric_one"
    assert result.metrics[1].metric_name == "metric_two"
    assert result.overall_score == 0.75


def test_engine_preserves_metric_order() -> None:
    registry = EvaluatorRegistry()

    registry.register(
        "metric_one",
        FixedScoreEvaluator(
            metric_name="metric_one",
            score=0.2,
        ),
    )
    registry.register(
        "metric_two",
        FixedScoreEvaluator(
            metric_name="metric_two",
            score=0.8,
        ),
    )

    engine = EvaluationEngine(registry)

    item = DatasetItem(
        input="Question",
        expected_output="Answer",
    )

    result = engine.evaluate(
        item=item,
        generated_output="Generated answer",
        metrics=["metric_two", "metric_one"],
    )

    assert [
        metric.metric_name
        for metric in result.metrics
    ] == ["metric_two", "metric_one"]

    assert result.overall_score == 0.5


def test_engine_raises_error_for_unknown_metric() -> None:
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

    try:
        engine.evaluate(
            item=item,
            generated_output="Paris",
            metrics=["unknown_metric"],
        )
    except ValueError as exc:
        assert "unknown_metric" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown metric")





