import os
from collections.abc import Callable
from concurrent.futures import Executor, Future
from typing import Any

import pytest

# Keep tests in memory: importing app.api must not create the default
# SQLite database file. Tests that need SQLite use a temporary path.
os.environ["DATABASE_PATH"] = ""


class ImmediateExecutor(Executor):
    """Runs submitted work in the calling thread before returning."""

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        future: Future[Any] = Future()

        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)

        return future


class ManualExecutor(Executor):
    """Holds submitted work until the test runs it."""

    def __init__(self) -> None:
        self.pending: list[tuple[Future[Any], Callable[[], Any]]] = []

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        future: Future[Any] = Future()
        self.pending.append((future, lambda: fn(*args, **kwargs)))
        return future

    def run_next(self) -> None:
        future, work = self.pending.pop(0)
        future.set_result(work())


@pytest.fixture(autouse=True)
def immediate_evaluation_executor(
    monkeypatch: pytest.MonkeyPatch,
):
    """Give each test its own executor that finishes evaluations on submit.

    API tests can then read results right after POST, and a test that runs
    the app lifespan cannot shut down the executor used by other tests.
    """

    import app.api as api
    from app.services.evaluation_executor import EvaluationExecutor

    executor = EvaluationExecutor(
        max_concurrent=1,
        max_queued=100,
        executor=ImmediateExecutor(),
    )

    monkeypatch.setattr(
        api,
        "evaluation_executor",
        executor,
    )

    return executor
