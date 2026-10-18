from app.evaluators.registry import EvaluatorRegistry
from app.models import (
    AggregationOptions,
    Dataset,
    DatasetItem,
    EvaluationResult,
    MetricResult,
    PassRule,
)


class EvaluationEngine:
    """Runs evaluators and produces evaluation results."""

    def __init__(
        self,
        registry: EvaluatorRegistry,
    ) -> None:
        self._registry = registry

    def validate_metrics(
        self,
        metrics: list[str],
    ) -> None:
        """Raise ValueError if any metric is not registered."""

        available = set(self._registry.available_metrics())

        unknown = [
            metric_name for metric_name in metrics if metric_name not in available
        ]

        if unknown:
            raise ValueError(f"Unknown metric: {', '.join(unknown)}")

    def validate_dataset(
        self,
        dataset: Dataset,
        metrics: list[str],
    ) -> None:
        """Raise ValueError if a metric cannot score an item of the dataset."""

        for metric_name in metrics:
            evaluator = self._registry.get(metric_name)

            for index, item in enumerate(dataset.items, start=1):
                try:
                    evaluator.validate_expected_output(item.expected_output)
                except ValueError as exc:
                    raise ValueError(
                        f"Metric '{metric_name}' cannot score dataset item "
                        f"{index}: {exc}"
                    ) from exc

    def available_metrics(self) -> list[str]:
        """Return the registered metric names."""
        return self._registry.available_metrics()

    def evaluate(
        self,
        item: DatasetItem,
        generated_output: str,
        metrics: list[str],
        aggregation: AggregationOptions | None = None,
    ) -> EvaluationResult:
        aggregation = aggregation or AggregationOptions()

        metric_results: list[MetricResult] = []

        for metric_name in metrics:
            evaluator = self._registry.get(metric_name)

            metric_result = evaluator.evaluate(
                expected_output=item.expected_output,
                generated_output=generated_output,
                input_text=item.input,
            )

            metric_results.append(metric_result)

        overall_score = self._overall_score(metric_results, aggregation)

        return EvaluationResult(
            item_id=item.id,
            generated_output=generated_output,
            metrics=metric_results,
            overall_score=overall_score,
            passed=self._passed(metric_results, overall_score, aggregation),
        )

    @staticmethod
    def _overall_score(
        metric_results: list[MetricResult],
        aggregation: AggregationOptions,
    ) -> float:
        """Return the weighted mean of the metric scores."""

        weights = [
            aggregation.weight_for(result.metric_name) for result in metric_results
        ]
        total_weight = sum(weights)

        if total_weight == 0:
            raise ValueError("at least one metric needs a positive weight")

        weighted = sum(
            weight * result.score
            for weight, result in zip(weights, metric_results, strict=True)
        )

        # Guard against float error pushing the mean just outside [0, 1].
        return min(1.0, max(0.0, weighted / total_weight))

    @staticmethod
    def _passed(
        metric_results: list[MetricResult],
        overall_score: float,
        aggregation: AggregationOptions,
    ) -> bool:
        """Apply the pass rule. Metrics with weight 0 are informational."""

        if aggregation.pass_rule == PassRule.OVERALL_SCORE:
            assert aggregation.pass_threshold is not None
            return overall_score >= aggregation.pass_threshold

        counted = [
            result.passed
            for result in metric_results
            if aggregation.weight_for(result.metric_name) > 0
        ]

        if aggregation.pass_rule == PassRule.ANY_METRIC:
            return any(counted)

        return all(counted)
