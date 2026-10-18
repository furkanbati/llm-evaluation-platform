import threading
import time

import pytest

from app.services.evaluation_executor import (
    EvaluationExecutor,
    EvaluationQueueFullError,
)


def wait_until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout

    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("Condition not met in time")

        time.sleep(0.01)


def submit(executor: EvaluationExecutor, function, *args):
    executor.reserve()
    return executor.submit(function, *args)


def test_executor_limits_concurrent_evaluations() -> None:
    executor = EvaluationExecutor(
        max_concurrent=2,
        max_queued=10,
    )

    release = threading.Event()
    lock = threading.Lock()
    running = 0
    max_running = 0
    started = 0

    def job() -> None:
        nonlocal running, max_running, started

        with lock:
            running += 1
            started += 1
            max_running = max(max_running, running)

        release.wait(timeout=5)

        with lock:
            running -= 1

    futures = [submit(executor, job) for _ in range(5)]

    wait_until(lambda: started == 2)
    time.sleep(0.1)

    assert started == 2
    assert executor.in_flight == 5

    release.set()

    for future in futures:
        future.result(timeout=5)

    assert max_running == 2
    wait_until(lambda: executor.in_flight == 0)

    executor.shutdown()


def test_executor_rejects_when_queue_is_full() -> None:
    executor = EvaluationExecutor(
        max_concurrent=1,
        max_queued=2,
    )

    executor.reserve()
    executor.reserve()

    with pytest.raises(EvaluationQueueFullError):
        executor.reserve()

    executor.release()
    executor.reserve()

    assert executor.in_flight == 2


def test_executor_releases_slot_when_job_completes() -> None:
    executor = EvaluationExecutor(
        max_concurrent=1,
        max_queued=1,
    )

    submit(executor, lambda: None).result(timeout=5)

    wait_until(lambda: executor.in_flight == 0)

    submit(executor, lambda: None).result(timeout=5)

    executor.shutdown()


def test_executor_releases_slot_when_job_raises() -> None:
    executor = EvaluationExecutor(
        max_concurrent=1,
        max_queued=1,
    )

    def failing_job() -> None:
        raise RuntimeError("boom")

    future = submit(executor, failing_job)

    with pytest.raises(RuntimeError):
        future.result(timeout=5)

    wait_until(lambda: executor.in_flight == 0)

    executor.shutdown()


def test_executor_shutdown_cancels_queued_jobs_and_releases_slots() -> None:
    executor = EvaluationExecutor(
        max_concurrent=1,
        max_queued=5,
    )

    release = threading.Event()
    started = threading.Event()

    def blocking_job() -> None:
        started.set()
        release.wait(timeout=5)

    running = submit(executor, blocking_job)
    queued = [submit(executor, lambda: None) for _ in range(3)]

    started.wait(timeout=5)
    executor.shutdown()

    assert all(future.cancelled() for future in queued)

    release.set()
    running.result(timeout=5)

    wait_until(lambda: executor.in_flight == 0)


@pytest.mark.parametrize(
    ("max_concurrent", "max_queued"),
    [
        (0, 10),
        (3, 2),
    ],
)
def test_executor_rejects_invalid_limits(
    max_concurrent: int,
    max_queued: int,
) -> None:
    with pytest.raises(ValueError):
        EvaluationExecutor(
            max_concurrent=max_concurrent,
            max_queued=max_queued,
        )
