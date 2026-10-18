import logging
import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


class EvaluationQueueFullError(Exception):
    """Raised when no more evaluations can be accepted."""


class EvaluationExecutor:
    """Runs evaluations with limits on concurrency and queue size.

    At most `max_concurrent` evaluations run at once; others wait in the
    executor's queue without occupying a thread. `max_queued` caps the
    evaluations that are queued or running together.
    """

    def __init__(
        self,
        max_concurrent: int,
        max_queued: int,
        executor: Executor | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")

        if max_queued < max_concurrent:
            raise ValueError("max_queued must be at least max_concurrent")

        self._max_queued = max_queued
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="evaluation",
        )
        self._in_flight = 0
        self._lock = threading.Lock()

    @property
    def in_flight(self) -> int:
        """Number of evaluations that are queued or running."""
        with self._lock:
            return self._in_flight

    def reserve(self) -> None:
        """Reserve a slot for one evaluation.

        Call before creating the run so a full queue rejects the request
        without storing anything. Raises EvaluationQueueFullError.
        """

        with self._lock:
            if self._in_flight >= self._max_queued:
                raise EvaluationQueueFullError(
                    f"{self._in_flight} evaluations are already queued or running"
                )

            self._in_flight += 1

    def release(self) -> None:
        """Release a reserved slot that will not be submitted."""
        with self._lock:
            self._in_flight -= 1

    def submit(
        self,
        function: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        """Submit work for a previously reserved slot."""

        future = self._executor.submit(function, *args, **kwargs)
        future.add_done_callback(self._on_done)

        return future

    def shutdown(self) -> None:
        """Stop accepting work and cancel evaluations that have not started."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _on_done(self, future: Future[Any]) -> None:
        self.release()

        if not future.cancelled() and future.exception() is not None:
            logger.error(
                "Evaluation task failed unexpectedly",
                exc_info=future.exception(),
            )
