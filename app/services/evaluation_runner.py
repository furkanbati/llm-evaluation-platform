import logging
from datetime import datetime, timezone

from app.models import (
    Dataset,
    EvaluationRun,
    EvaluationStatus,
)
from app.services.evaluation_engine import EvaluationEngine
from app.services.model_client import ModelClient
from app.storage.base import BaseEvaluationRepository
from app.storage.evaluation_repository import EvaluationRepository


logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 1_000

LLM_JUDGE_METRIC = "llm_judge"

INTERRUPTED_ERROR = "Interrupted by application restart"


class EvaluationRunner:
    """Runs evaluations for all dataset items."""

    def __init__(
        self,
        engine: EvaluationEngine,
        model_client: ModelClient | None = None,
        repository: BaseEvaluationRepository | None = None,
        judge_model_name: str | None = None,
    ) -> None:
        self._engine = engine
        self._model_client = model_client
        self._repository = repository or EvaluationRepository()
        self._judge_model_name = judge_model_name

    @property
    def repository(self) -> BaseEvaluationRepository:
        """Return the repository used by the runner."""
        return self._repository

    def run(
        self,
        dataset: Dataset,
        model_name: str,
        metrics: list[str],
        generated_outputs: list[str] | None = None,
    ) -> EvaluationRun:
        """Create and execute an evaluation run synchronously."""

        evaluation_run = self.create_run(
            dataset=dataset,
            model_name=model_name,
            metrics=metrics,
            generated_outputs=generated_outputs,
        )

        return self.execute(
            evaluation_run=evaluation_run,
            dataset=dataset,
            metrics=metrics,
            generated_outputs=generated_outputs,
        )

    def create_run(
        self,
        dataset: Dataset,
        model_name: str,
        metrics: list[str],
        generated_outputs: list[str] | None = None,
    ) -> EvaluationRun:
        """Validate the request and store a PENDING evaluation run.

        Invalid requests raise ValueError before a run is created.
        """

        self._validate_request(
            dataset=dataset,
            metrics=metrics,
            generated_outputs=generated_outputs,
        )

        judge_model_name = (
            self._judge_model_name
            if LLM_JUDGE_METRIC in metrics
            else None
        )

        evaluation_run = EvaluationRun(
            dataset_id=dataset.id,
            dataset=dataset,
            model_name=model_name,
            metrics=metrics,
            judge_model_name=judge_model_name,
            status=EvaluationStatus.PENDING,
        )

        return self._repository.save(evaluation_run)

    def execute(
        self,
        evaluation_run: EvaluationRun,
        dataset: Dataset,
        metrics: list[str],
        generated_outputs: list[str] | None = None,
    ) -> EvaluationRun:
        """Execute a created run and record its outcome.

        Never raises: failures are recorded on a FAILED run. The status
        is assigned last so readers never see a finished status without
        its results or completion time.
        """

        evaluation_run.status = EvaluationStatus.RUNNING
        self._repository.save(evaluation_run)

        try:
            if generated_outputs is None:
                inputs = [
                    item.input
                    for item in dataset.items
                ]

                generated_outputs = self._model_client.generate(
                    model_name=evaluation_run.model_name,
                    inputs=inputs,
                )

                if len(dataset.items) != len(generated_outputs):
                    raise ValueError(
                        "Model client returned "
                        f"{len(generated_outputs)} outputs "
                        f"for {len(dataset.items)} dataset items"
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
            evaluation_run.completed_at = datetime.now(timezone.utc)
            evaluation_run.status = EvaluationStatus.COMPLETED

        except Exception as exc:
            logger.exception(
                "Evaluation run %s failed",
                evaluation_run.id,
            )

            evaluation_run.error = self._format_error(exc)
            evaluation_run.completed_at = datetime.now(timezone.utc)
            evaluation_run.status = EvaluationStatus.FAILED

        return self._repository.save(evaluation_run)

    def fail_interrupted_runs(self) -> int:
        """Mark runs left PENDING or RUNNING by a previous process as FAILED.

        Background evaluations do not survive a restart, so these runs
        would otherwise never finish. Returns the number of runs updated.
        """

        interrupted = [
            evaluation_run
            for status in (
                EvaluationStatus.PENDING,
                EvaluationStatus.RUNNING,
            )
            for evaluation_run in self._repository.list(status=status)
        ]

        for evaluation_run in interrupted:
            evaluation_run.error = INTERRUPTED_ERROR
            evaluation_run.completed_at = datetime.now(timezone.utc)
            evaluation_run.status = EvaluationStatus.FAILED

            self._repository.save(evaluation_run)

        if interrupted:
            logger.warning(
                "Marked %d interrupted evaluation runs as failed",
                len(interrupted),
            )

        return len(interrupted)

    def _validate_request(
        self,
        dataset: Dataset,
        metrics: list[str],
        generated_outputs: list[str] | None,
    ) -> None:
        self._engine.validate_metrics(metrics)

        if generated_outputs is None:
            if self._model_client is None:
                raise ValueError(
                    "Model client is required when generated outputs "
                    "are not provided"
                )

        elif len(dataset.items) != len(generated_outputs):
            raise ValueError(
                "Number of generated outputs must match dataset items"
            )

    @staticmethod
    def _format_error(exc: Exception) -> str:
        message = str(exc) or type(exc).__name__

        return message[:MAX_ERROR_LENGTH]
