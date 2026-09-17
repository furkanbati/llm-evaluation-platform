from uuid import UUID

from app.models import EvaluationRun


class EvaluationRepository:
    """In-memory repository for evaluation runs."""

    def __init__(self) -> None:
        self._evaluations: dict[UUID, EvaluationRun] = {}

    def save(self, evaluation: EvaluationRun) -> EvaluationRun:
        """Store an evaluation run and return it."""
        self._evaluations[evaluation.id] = evaluation
        return evaluation

    def get(self, evaluation_id: UUID) -> EvaluationRun | None:
        """Return an evaluation run by ID, if it exists."""
        return self._evaluations.get(evaluation_id)

    def list(self) -> list[EvaluationRun]:
        """Return all stored evaluation runs."""
        return list(self._evaluations.values())

