from datetime import datetime, timezone

from app.models import (
    Dataset,
    EvaluationRun,
    EvaluationStatus,
)
from app.services.evaluation_engine import EvaluationEngine
from app.services.model_client import ModelClient
from app.storage.evaluation_repository import EvaluationRepository


class EvaluationRunner:
    """Runs evaluations for all dataset items."""

    def __init__(
        self,
        engine: EvaluationEngine,
        model_client: ModelClient | None = None,
        repository: EvaluationRepository | None = None,
    ) -> None:
        self._engine = engine
        self._model_client = model_client
        self._repository = repository or EvaluationRepository()

    def run(
        self,
        dataset: Dataset,
        model_name: str,
        metrics: list[str],
        generated_outputs: list[str] | None = None,
    ) -> EvaluationRun:
        """Evaluate all dataset items and return an evaluation run."""

        created_at = datetime.now(timezone.utc)

        evaluation_run = EvaluationRun(
            dataset_id=dataset.id,
            model_name=model_name,
            status=EvaluationStatus.RUNNING,
            created_at=created_at,
        )

        try:
            if generated_outputs is None:
                if self._model_client is None:
                    raise ValueError(
                        "Model client is required when generated outputs "
                        "are not provided"
                    )

                inputs = [
                    item.input
                    for item in dataset.items
                ]

                generated_outputs = self._model_client.generate(
                    model_name=model_name,
                    inputs=inputs,
                )

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

            evaluation_run.results = results
            evaluation_run.status = EvaluationStatus.COMPLETED
            evaluation_run.completed_at = datetime.now(timezone.utc)

            return self._repository.save(evaluation_run)

        except ValueError:
            raise

        except Exception:
            evaluation_run.status = EvaluationStatus.FAILED
            evaluation_run.completed_at = datetime.now(timezone.utc)

            return self._repository.save(evaluation_run)

