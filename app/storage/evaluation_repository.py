import threading
from uuid import UUID

from app.models import EvaluationRun, EvaluationStatus


class EvaluationRepository:
    """Thread-safe in-memory repository for evaluation runs."""

    def __init__(self) -> None:
        self._evaluations: dict[UUID, EvaluationRun] = {}
        self._lock = threading.Lock()

    def save(self, evaluation: EvaluationRun) -> EvaluationRun:
        """Store an evaluation run and return it."""
        with self._lock:
            self._evaluations[evaluation.id] = evaluation
        return evaluation

    def get(self, evaluation_id: UUID) -> EvaluationRun | None:
        """Return an evaluation run by ID, if it exists."""
        with self._lock:
            return self._evaluations.get(evaluation_id)

    def delete(
        self,
        evaluation_id: UUID,
    ) -> bool:
        """Delete an evaluation by id."""

        with self._lock:
            if evaluation_id not in self._evaluations:
                return False

            del self._evaluations[evaluation_id]

        return True

    def list(
        self,
        status: EvaluationStatus | None = None,
        model_name: str | None = None,
    ) -> list[EvaluationRun]:
        """Return evaluations matching the provided filters."""

        with self._lock:
            evaluations = list(
                self._evaluations.values()
            )

        if status is not None:
            evaluations = [
                evaluation
                for evaluation in evaluations
                if evaluation.status == status
            ]

        if model_name is not None:
            evaluations = [
                evaluation
                for evaluation in evaluations
                if evaluation.model_name == model_name
            ]

        return evaluations
