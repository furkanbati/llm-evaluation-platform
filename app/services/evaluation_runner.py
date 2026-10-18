import logging
from datetime import UTC, datetime

from app.errors import EvaluationError, InvalidModelOutputError
from app.models import (
    AggregationOptions,
    Dataset,
    EvaluationErrorCode,
    EvaluationRun,
    EvaluationStatus,
    GenerationOptions,
)
from app.services.evaluation_engine import EvaluationEngine
from app.services.model_client import ModelClient
from app.storage.base import BaseEvaluationRepository
from app.storage.evaluation_repository import EvaluationRepository

logger = logging.getLogger(__name__)

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
        default_generation_options: GenerationOptions | None = None,
    ) -> None:
        self._engine = engine
        self._model_client = model_client
        self._repository = repository or EvaluationRepository()
        self._judge_model_name = judge_model_name
        self._default_generation_options = (
            default_generation_options or GenerationOptions()
        )

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
        generation_options: GenerationOptions | None = None,
        aggregation: AggregationOptions | None = None,
    ) -> EvaluationRun:
        """Create and execute an evaluation run synchronously."""

        evaluation_run = self.create_run(
            dataset=dataset,
            model_name=model_name,
            metrics=metrics,
            generated_outputs=generated_outputs,
            generation_options=generation_options,
            aggregation=aggregation,
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
        generation_options: GenerationOptions | None = None,
        aggregation: AggregationOptions | None = None,
    ) -> EvaluationRun:
        """Validate the request and store a PENDING evaluation run.

        Invalid requests raise ValueError before a run is created.
        Generation options left unset fall back to the runner defaults,
        and the resolved options are stored on the run.
        """

        self._validate_request(
            dataset=dataset,
            metrics=metrics,
            generated_outputs=generated_outputs,
        )

        judge_model_name = (
            self._judge_model_name if LLM_JUDGE_METRIC in metrics else None
        )

        evaluation_run = EvaluationRun(
            dataset_id=dataset.id,
            dataset=dataset,
            model_name=model_name,
            metrics=metrics,
            judge_model_name=judge_model_name,
            generation_options=self._resolve_generation_options(
                generation_options,
            ),
            aggregation=aggregation or AggregationOptions(),
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
                if self._model_client is None:
                    raise ValueError(
                        "Model client is required when generated outputs "
                        "are not provided"
                    )

                inputs = [item.input for item in dataset.items]

                generated_outputs = self._model_client.generate(
                    model_name=evaluation_run.model_name,
                    inputs=inputs,
                    options=evaluation_run.generation_options,
                )

                if len(dataset.items) != len(generated_outputs):
                    raise InvalidModelOutputError(
                        "Model client returned "
                        f"{len(generated_outputs)} outputs "
                        f"for {len(dataset.items)} dataset items"
                    )

            for item, generated_output in zip(
                dataset.items,
                generated_outputs,
                strict=True,
            ):
                if not generated_output.strip():
                    raise InvalidModelOutputError(
                        f"Empty output for dataset item {item.id}"
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
                    aggregation=evaluation_run.aggregation,
                )

                results.append(result)

            evaluation_run.results = results
            evaluation_run.completed_at = datetime.now(UTC)
            evaluation_run.status = EvaluationStatus.COMPLETED

        except Exception as exc:
            logger.exception(
                "Evaluation run %s failed",
                evaluation_run.id,
            )

            evaluation_run.error_code, evaluation_run.error = self._classify_error(
                exc,
            )
            evaluation_run.completed_at = datetime.now(UTC)
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
            evaluation_run.error_code = EvaluationErrorCode.INTERRUPTED
            evaluation_run.error = INTERRUPTED_ERROR
            evaluation_run.completed_at = datetime.now(UTC)
            evaluation_run.status = EvaluationStatus.FAILED

            self._repository.save(evaluation_run)

        if interrupted:
            logger.warning(
                "Marked %d interrupted evaluation runs as failed",
                len(interrupted),
            )

        return len(interrupted)

    def _resolve_generation_options(
        self,
        overrides: GenerationOptions | None,
    ) -> GenerationOptions:
        if overrides is None:
            return self._default_generation_options.model_copy()

        return self._default_generation_options.model_copy(
            update=overrides.model_dump(exclude_none=True),
        )

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
                    "Model client is required when generated outputs are not provided"
                )

        elif len(dataset.items) != len(generated_outputs):
            raise ValueError("Number of generated outputs must match dataset items")

    @staticmethod
    def _classify_error(
        exc: Exception,
    ) -> tuple[EvaluationErrorCode, str]:
        """Return a public error code and message for a failure.

        Details of the original exception are only logged, so provider
        addresses and system messages are not exposed through the API.
        """

        if isinstance(exc, EvaluationError):
            return exc.code, exc.public_message

        return (
            EvaluationErrorCode.INTERNAL_ERROR,
            EvaluationError.public_message,
        )
