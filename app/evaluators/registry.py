from app.evaluators.base import Evaluator


class EvaluatorRegistry:
    """Stores and retrieves evaluators by metric name."""

    def __init__(self) -> None:
        self._evaluators: dict[str, Evaluator] = {}

    def register(
        self,
        metric_name: str,
        evaluator: Evaluator,
    ) -> None:
        """Register a new evaluator."""

        if metric_name in self._evaluators:
            raise ValueError(
                f"Evaluator already registered: {metric_name}"
            )

        self._evaluators[metric_name] = evaluator

    def get(
        self,
        metric_name: str,
    ) -> Evaluator:
        """Retrieve an evaluator by metric name."""

        try:
            return self._evaluators[metric_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown metric: {metric_name}"
            ) from exc

    def available_metrics(self) -> list[str]:
        """Return all registered metric names."""

        return sorted(self._evaluators.keys())