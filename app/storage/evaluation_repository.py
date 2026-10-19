import threading
from uuid import UUID

from app.models import EvaluationResult, EvaluationRun, EvaluationStatus, ItemError
from app.storage.base import BaseEvaluationRepository

_ItemOutcome = EvaluationResult | ItemError


class EvaluationRepository(BaseEvaluationRepository):
    """Thread-safe in-memory repository for evaluation runs.

    Runs are stored and returned as copies, like the SQLite repository, so
    assigning fields of a returned run, or appending to its results, does
    not change the stored run. The copies are shallow: nested objects such
    as the dataset are shared and must not be modified in place.
    """

    def __init__(self) -> None:
        self._evaluations: dict[UUID, EvaluationRun] = {}
        # Per run: item ID -> (dataset position, result or error).
        self._items: dict[UUID, dict[UUID, tuple[int, _ItemOutcome]]] = {}
        self._lock = threading.Lock()

    def save(self, evaluation: EvaluationRun) -> EvaluationRun:
        """Store a run with its results and item errors, and return it."""

        positions = _item_positions(evaluation)
        items: dict[UUID, tuple[int, _ItemOutcome]] = {}

        outcomes: list[_ItemOutcome] = [*evaluation.results, *evaluation.item_errors]

        for outcome in outcomes:
            items[outcome.item_id] = (
                positions.get(outcome.item_id, len(items)),
                outcome,
            )

        with self._lock:
            self._evaluations[evaluation.id] = _without_items(evaluation)
            self._items[evaluation.id] = items

        return evaluation

    def update_run(self, evaluation: EvaluationRun) -> bool:
        with self._lock:
            if evaluation.id not in self._evaluations:
                return False

            self._evaluations[evaluation.id] = _without_items(evaluation)

        return True

    def save_result(
        self,
        evaluation_id: UUID,
        position: int,
        result: EvaluationResult,
    ) -> bool:
        return self._save_item(evaluation_id, position, result)

    def save_item_error(
        self,
        evaluation_id: UUID,
        position: int,
        item_error: ItemError,
    ) -> bool:
        return self._save_item(evaluation_id, position, item_error)

    def _save_item(
        self,
        evaluation_id: UUID,
        position: int,
        outcome: _ItemOutcome,
    ) -> bool:
        with self._lock:
            if evaluation_id not in self._evaluations:
                return False

            self._items[evaluation_id][outcome.item_id] = (position, outcome)

        return True

    def get(self, evaluation_id: UUID) -> EvaluationRun | None:
        """Return an evaluation run by ID, if it exists."""
        with self._lock:
            return self._assemble(evaluation_id)

    def delete(
        self,
        evaluation_id: UUID,
    ) -> bool:
        """Delete an evaluation by id."""

        with self._lock:
            if evaluation_id not in self._evaluations:
                return False

            del self._evaluations[evaluation_id]
            del self._items[evaluation_id]

        return True

    def list(
        self,
        status: EvaluationStatus | None = None,
        model_name: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[EvaluationRun]:
        """Return evaluations matching the provided filters."""

        with self._lock:
            matching = [
                evaluation.id
                for evaluation in self._evaluations.values()
                if (status is None or evaluation.status == status)
                and (model_name is None or evaluation.model_name == model_name)
            ]

            if limit is None:
                selected = matching[offset:]
            else:
                selected = matching[offset : offset + limit]

            return [
                run
                for evaluation_id in selected
                if (run := self._assemble(evaluation_id)) is not None
            ]

    def _assemble(self, evaluation_id: UUID) -> EvaluationRun | None:
        """Return a copy of a run with its items. Call with the lock held."""

        stored = self._evaluations.get(evaluation_id)

        if stored is None:
            return None

        results: list[EvaluationResult] = []
        item_errors: list[ItemError] = []

        for _, outcome in sorted(
            self._items[evaluation_id].values(),
            key=lambda entry: entry[0],
        ):
            if isinstance(outcome, EvaluationResult):
                results.append(outcome)
            else:
                item_errors.append(outcome)

        return stored.model_copy(
            update={
                "results": results,
                "item_errors": item_errors,
            },
        )


def _without_items(evaluation: EvaluationRun) -> EvaluationRun:
    return evaluation.model_copy(
        update={
            "results": [],
            "item_errors": [],
        },
    )


def _item_positions(evaluation: EvaluationRun) -> dict[UUID, int]:
    """Map item IDs to their dataset position, if the dataset is stored."""

    if evaluation.dataset is None:
        return {}

    return {item.id: index for index, item in enumerate(evaluation.dataset.items)}
