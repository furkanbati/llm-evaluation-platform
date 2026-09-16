from datetime import datetime, timezone

from app.models import (
    Dataset,
    EvaluationRun,
    EvaluationStatus,
)
from app.services.evaluation_engine import EvaluationEngine


class EvaluationRunner:
    """Runs evaluations for all dataset items."""

    def __init__(
        self,
        engine: EvaluationEngine,
    ) -> None:
        self._engine = engine

    def run(
        self,
        dataset: Dataset,
        model_name: str,
        generated_outputs: list[str],
        metrics: list[str],
    ) -> EvaluationRun:
        """Evaluate all dataset items and return an evaluation run."""

        if len(dataset.items) != len(generated_outputs):
            raise ValueError(
                "Number of generated outputs must match dataset items"
            )

        results = []

        for item, generated_output in zip(
            dataset.items,
            generated_outputs,
            strict=True,
        ):
            result = self._engine.evaluate(
                item=item,
                generated_output=generated_output,
                metrics=metrics,
            )

            results.append(result)

        return EvaluationRun(
            dataset_id=dataset.id,
            model_name=model_name,
            status=EvaluationStatus.COMPLETED,
            results=results,
            completed_at=datetime.now(timezone.utc),
        )