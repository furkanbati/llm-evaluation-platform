import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeVar

from app.errors import EvaluationError, InvalidModelOutputError
from app.models import (
    AggregationOptions,
    Dataset,
    DatasetItem,
    EvaluationErrorCode,
    EvaluationResult,
    EvaluationRun,
    EvaluationStatus,
    GenerationOptions,
    ItemError,
)
from app.services.evaluation_engine import EvaluationEngine
from app.services.model_client import ModelClient
from app.storage.base import BaseEvaluationRepository
from app.storage.evaluation_repository import EvaluationRepository

logger = logging.getLogger(__name__)

LLM_JUDGE_METRIC = "llm_judge"

INTERRUPTED_ERROR = "Interrupted by application restart"

T = TypeVar("T")


class _ItemFailure(Exception):
    """An item could not be scored, after `attempts` tries of the last step."""

    def __init__(
        self,
        cause: Exception,
        attempts: int,
        generated_output: str | None = None,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.attempts = attempts
        self.generated_output = generated_output

    @property
    def fatal(self) -> bool:
        return isinstance(self.cause, EvaluationError) and self.cause.fatal


class _RunDeleted(Exception):
    """The run was deleted while it was executing."""


class EvaluationRunner:
    """Runs evaluations item by item and stores each outcome as it finishes.

    An item that cannot be scored is recorded as an item error and the run
    moves on, so one failure does not discard the other results. Calls that
    fail with a retryable error are tried up to `max_attempts` times with
    exponential backoff. The run stops early when an error affects every
    item (such as an unknown model) or when `max_consecutive_failures` items
    in a row fail.
    """

    def __init__(
        self,
        engine: EvaluationEngine,
        model_client: ModelClient | None = None,
        repository: BaseEvaluationRepository | None = None,
        judge_model_name: str | None = None,
        default_generation_options: GenerationOptions | None = None,
        max_attempts: int = 1,
        retry_backoff_seconds: float = 0.0,
        max_consecutive_failures: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")

        if max_consecutive_failures is not None and max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be at least 1")

        self._engine = engine
        self._model_client = model_client
        self._repository = repository or EvaluationRepository()
        self._judge_model_name = judge_model_name
        self._default_generation_options = (
            default_generation_options or GenerationOptions()
        )
        self._max_attempts = max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._max_consecutive_failures = max_consecutive_failures
        self._sleep = sleep

    def available_metrics(self) -> list[str]:
        """Return the metrics this runner can evaluate."""
        return self._engine.available_metrics()

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
        """Execute a stored run and record each item as it finishes.

        Items that already have a result or error on `evaluation_run` are
        skipped, so a run loaded from the repository resumes where it
        stopped. Never raises: failures are recorded on the run. The status
        is assigned last so readers never see a finished status without its
        completion time.
        """

        evaluation_run.status = EvaluationStatus.RUNNING

        if not self._repository.update_run(evaluation_run):
            logger.info("Evaluation run %s was deleted", evaluation_run.id)
            return evaluation_run

        try:
            stop = self._evaluate_items(
                evaluation_run=evaluation_run,
                dataset=dataset,
                metrics=metrics,
                generated_outputs=generated_outputs,
            )
        except _RunDeleted:
            logger.info(
                "Evaluation run %s was deleted while executing",
                evaluation_run.id,
            )
            return evaluation_run
        except Exception as exc:
            logger.exception("Evaluation run %s failed", evaluation_run.id)
            stop = self._classify_error(exc)

        self._sort_items(evaluation_run, dataset)
        self._finish(evaluation_run, stop)

        return evaluation_run

    def _evaluate_items(
        self,
        evaluation_run: EvaluationRun,
        dataset: Dataset,
        metrics: list[str],
        generated_outputs: list[str] | None,
    ) -> tuple[EvaluationErrorCode, str] | None:
        """Process the remaining items.

        Returns the error code and message if the run had to stop early,
        otherwise None.
        """

        if generated_outputs is None and self._model_client is None:
            raise ValueError(
                "Model client is required when generated outputs are not provided"
            )

        finished = {result.item_id for result in evaluation_run.results} | {
            item_error.item_id for item_error in evaluation_run.item_errors
        }

        consecutive_failures = 0

        for position, item in enumerate(dataset.items):
            if item.id in finished:
                continue

            provided_output = (
                generated_outputs[position] if generated_outputs is not None else None
            )

            try:
                result = self._evaluate_item(
                    evaluation_run=evaluation_run,
                    item=item,
                    metrics=metrics,
                    provided_output=provided_output,
                )
            except _ItemFailure as failure:
                logger.warning(
                    "Item %s of evaluation run %s failed after %d attempt(s)",
                    item.id,
                    evaluation_run.id,
                    failure.attempts,
                    exc_info=failure.cause,
                )

                code, message = self._classify_error(failure.cause)

                if failure.fatal:
                    return code, message

                item_error = ItemError(
                    item_id=item.id,
                    error_code=code,
                    error=message,
                    attempts=failure.attempts,
                    generated_output=failure.generated_output,
                )

                if not self._repository.save_item_error(
                    evaluation_run.id,
                    position,
                    item_error,
                ):
                    raise _RunDeleted from None

                evaluation_run.item_errors.append(item_error)
                consecutive_failures += 1

                if (
                    self._max_consecutive_failures is not None
                    and consecutive_failures >= self._max_consecutive_failures
                ):
                    return code, (
                        f"{message} The run stopped after "
                        f"{consecutive_failures} items in a row failed."
                    )

                continue

            if not self._repository.save_result(
                evaluation_run.id,
                position,
                result,
            ):
                raise _RunDeleted

            evaluation_run.results.append(result)
            consecutive_failures = 0

        return None

    def _evaluate_item(
        self,
        evaluation_run: EvaluationRun,
        item: DatasetItem,
        metrics: list[str],
        provided_output: str | None,
    ) -> EvaluationResult:
        """Generate (unless provided) and score one item.

        Generation and scoring are retried separately, so a failed judge
        call does not generate the answer again. Raises _ItemFailure.
        """

        if provided_output is None:
            generated_output = self._with_retry(
                lambda: self._generate(evaluation_run, item),
            )
        elif not provided_output.strip():
            raise _ItemFailure(
                InvalidModelOutputError(f"Empty output for dataset item {item.id}"),
                attempts=1,
            )
        else:
            generated_output = provided_output

        try:
            return self._with_retry(
                lambda: self._engine.evaluate(
                    item=item,
                    generated_output=generated_output,
                    metrics=metrics,
                    aggregation=evaluation_run.aggregation,
                ),
            )
        except _ItemFailure as failure:
            failure.generated_output = generated_output
            raise

    def _generate(
        self,
        evaluation_run: EvaluationRun,
        item: DatasetItem,
    ) -> str:
        assert self._model_client is not None

        outputs = self._model_client.generate(
            model_name=evaluation_run.model_name,
            inputs=[item.input],
            options=evaluation_run.generation_options,
        )

        if len(outputs) != 1:
            raise InvalidModelOutputError(
                f"Model client returned {len(outputs)} outputs for 1 dataset item"
            )

        if not outputs[0].strip():
            raise InvalidModelOutputError(f"Empty output for dataset item {item.id}")

        return outputs[0]

    def _with_retry(self, operation: Callable[[], T]) -> T:
        """Call `operation`, retrying retryable errors with backoff.

        Raises _ItemFailure with the last error once the attempts are used
        up or the error is not retryable.
        """

        attempt = 1

        while True:
            try:
                return operation()
            except Exception as exc:
                retryable = isinstance(exc, EvaluationError) and exc.retryable

                if not retryable or attempt >= self._max_attempts:
                    raise _ItemFailure(exc, attempts=attempt) from exc

                delay = self._retry_backoff_seconds * 2 ** (attempt - 1)

                logger.info(
                    "Attempt %d failed with %s; retrying in %.1fs",
                    attempt,
                    type(exc).__name__,
                    delay,
                )

                if delay > 0:
                    self._sleep(delay)

                attempt += 1

    def _finish(
        self,
        evaluation_run: EvaluationRun,
        stop: tuple[EvaluationErrorCode, str] | None,
    ) -> None:
        """Record the final status. Results are already stored."""

        if stop is None and evaluation_run.item_errors:
            if evaluation_run.results:
                status = EvaluationStatus.COMPLETED_WITH_ERRORS
            else:
                # Nothing could be scored: report it like any other failure.
                last = evaluation_run.item_errors[-1]
                stop = (last.error_code, last.error)
                status = EvaluationStatus.FAILED
        elif stop is None:
            status = EvaluationStatus.COMPLETED
        else:
            status = EvaluationStatus.FAILED

        if stop is not None:
            evaluation_run.error_code, evaluation_run.error = stop
        else:
            evaluation_run.error_code = None
            evaluation_run.error = None

        evaluation_run.completed_at = datetime.now(UTC)
        evaluation_run.status = status

        if not self._repository.update_run(evaluation_run):
            logger.info(
                "Evaluation run %s was deleted before it finished",
                evaluation_run.id,
            )

    @staticmethod
    def _sort_items(evaluation_run: EvaluationRun, dataset: Dataset) -> None:
        """Order results and errors by dataset position, as stored."""

        positions = {item.id: index for index, item in enumerate(dataset.items)}

        evaluation_run.results.sort(key=lambda result: positions[result.item_id])
        evaluation_run.item_errors.sort(
            key=lambda item_error: positions[item_error.item_id],
        )

    def interrupted_runs(self) -> list[EvaluationRun]:
        """Return runs left PENDING or RUNNING by a previous process."""

        return [
            evaluation_run
            for status in (
                EvaluationStatus.PENDING,
                EvaluationStatus.RUNNING,
            )
            for evaluation_run in self._repository.list(status=status)
        ]

    def prepare_resume(self, evaluation_run: EvaluationRun) -> None:
        """Mark an interrupted run as queued again before it is resubmitted."""

        evaluation_run.status = EvaluationStatus.PENDING
        self._repository.update_run(evaluation_run)

    def fail_interrupted_run(self, evaluation_run: EvaluationRun) -> None:
        """Mark an interrupted run that will not be resumed as FAILED.

        Items finished before the interruption keep their results.
        """

        evaluation_run.error_code = EvaluationErrorCode.INTERRUPTED
        evaluation_run.error = INTERRUPTED_ERROR
        evaluation_run.completed_at = datetime.now(UTC)
        evaluation_run.status = EvaluationStatus.FAILED

        self._repository.update_run(evaluation_run)

    def fail_interrupted_runs(self) -> int:
        """Mark every run left PENDING or RUNNING as FAILED.

        Returns the number of runs updated.
        """

        interrupted = self.interrupted_runs()

        for evaluation_run in interrupted:
            self.fail_interrupted_run(evaluation_run)

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
        self._engine.validate_dataset(dataset, metrics)

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
