from abc import ABC, abstractmethod
from uuid import UUID

from app.models import EvaluationRun, EvaluationStatus


class BaseEvaluationRepository(ABC):
    """Interface for storing evaluation runs."""

    @abstractmethod
    def save(self, evaluation: EvaluationRun) -> EvaluationRun:
        """Insert or update an evaluation run and return it."""

    @abstractmethod
    def get(self, evaluation_id: UUID) -> EvaluationRun | None:
        """Return an evaluation run by ID, if it exists."""

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
