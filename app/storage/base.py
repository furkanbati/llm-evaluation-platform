from abc import ABC, abstractmethod
from uuid import UUID

from app.models import EvaluationResult, EvaluationRun, EvaluationStatus, ItemError


class BaseEvaluationRepository(ABC):
    """Interface for storing evaluation runs.

    A run's settings and status are stored separately from its per-item
    outcomes, so a running evaluation can record each item as it finishes
    without rewriting the whole run.
    """

    @abstractmethod
    def save(self, evaluation: EvaluationRun) -> EvaluationRun:
        """Insert or replace a run, including its results and item errors."""

    @abstractmethod
    def update_run(self, evaluation: EvaluationRun) -> bool:
        """Update a stored run's settings and status, not its items.

        Returns False, and stores nothing, if the run no longer exists, so
        a run deleted while it executes is not recreated.
        """

    @abstractmethod
    def save_result(
        self,
        evaluation_id: UUID,
        position: int,
        result: EvaluationResult,
    ) -> bool:
        """Store the result of the dataset item at `position`.

        Returns False, and stores nothing, if the run no longer exists.
        """

    @abstractmethod
    def save_item_error(
        self,
        evaluation_id: UUID,
        position: int,
        item_error: ItemError,
    ) -> bool:
        """Store why the dataset item at `position` could not be scored.

        Returns False, and stores nothing, if the run no longer exists.
        """

    @abstractmethod
    def get(self, evaluation_id: UUID) -> EvaluationRun | None:
        """Return an evaluation run by ID, if it exists.

        Results and item errors are ordered by dataset position.
        """

    @abstractmethod
    def delete(self, evaluation_id: UUID) -> bool:
        """Delete an evaluation run and report whether it existed."""

    @abstractmethod
    def list(
        self,
        status: EvaluationStatus | None = None,
        model_name: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[EvaluationRun]:
        """Return matching runs in insertion order."""
