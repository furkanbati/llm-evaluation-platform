from app.evaluators.registry import EvaluatorRegistry
from app.models import (
    DatasetItem,
    EvaluationResult,
    MetricResult,
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
            metric_name
            for metric_name in metrics
            if metric_name not in available
        ]

        if unknown:
            raise ValueError(
                f"Unknown metric: {', '.join(unknown)}"
            )

    def evaluate(
        self,
        item: DatasetItem,
        generated_output: str,
        metrics: list[str],
    ) -> EvaluationResult:

        metric_results: list[MetricResult] = []

        for metric_name in metrics:
            evaluator = self._registry.get(metric_name)

            metric_result = evaluator.evaluate(
                expected_output=item.expected_output,
                generated_output=generated_output,
                input_text=item.input,
            )

            metric_results.append(metric_result)

        overall_score = (
            sum(
                result.score
                for result in metric_results
            )
            / len(metric_results)
        )

        return EvaluationResult(
            item_id=item.id,
            generated_output=generated_output,
            metrics=metric_results,
            overall_score=overall_score,
        )


        