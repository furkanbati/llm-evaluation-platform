import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from app.models import EvaluationResult, EvaluationRun, EvaluationStatus, ItemError
from app.storage.base import BaseEvaluationRepository

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_status
    ON evaluation_runs (status);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_model_name
    ON evaluation_runs (model_name);

CREATE TABLE IF NOT EXISTS evaluation_items (
    run_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('result', 'error')),
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, item_id)
);

CREATE INDEX IF NOT EXISTS idx_evaluation_items_run_position
    ON evaluation_items (run_id, position);
"""

RESULT = "result"
ERROR = "error"

# Results and item errors live in evaluation_items, not in the run's JSON.
_ITEM_FIELDS = {"results", "item_errors"}

_UPSERT_ITEM = """
    INSERT INTO evaluation_items
        (run_id, item_id, position, kind, data, updated_at)
    SELECT ?, ?, ?, ?, ?, ?
    WHERE EXISTS (SELECT 1 FROM evaluation_runs WHERE id = ?)
    ON CONFLICT (run_id, item_id) DO UPDATE SET
        position = excluded.position,
        kind = excluded.kind,
        data = excluded.data,
        updated_at = excluded.updated_at
"""


class SQLiteEvaluationRepository(BaseEvaluationRepository):
    """SQLite repository for evaluation runs.

    A run's settings and status are stored as JSON in `evaluation_runs`.
    Each item's result or error is a row of `evaluation_items`, so recording
    an item does not rewrite the run.
    """

    def __init__(self, path: str) -> None:
        self._path = path

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
            _migrate_embedded_items(connection)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open a connection for one transaction.

        A connection per operation keeps the repository safe to use from
        background task threads.
        """

        connection = sqlite3.connect(
            self._path,
            timeout=30.0,
        )

        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, evaluation: EvaluationRun) -> EvaluationRun:
        """Insert or replace a run, including its results and item errors."""

        positions = _item_positions(evaluation)
        outcomes: list[EvaluationResult | ItemError] = [
            *evaluation.results,
            *evaluation.item_errors,
        ]

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs
                    (id, status, model_name, created_at, data)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    status = excluded.status,
                    model_name = excluded.model_name,
                    data = excluded.data
                """,
                (
                    str(evaluation.id),
                    evaluation.status.value,
                    evaluation.model_name,
                    evaluation.created_at.isoformat(),
                    _run_json(evaluation),
                ),
            )

            connection.execute(
                "DELETE FROM evaluation_items WHERE run_id = ?",
                (str(evaluation.id),),
            )

            for index, outcome in enumerate(outcomes):
                _upsert_item(
                    connection,
                    evaluation.id,
                    positions.get(outcome.item_id, index),
                    outcome,
                )

        return evaluation

    def update_run(self, evaluation: EvaluationRun) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE evaluation_runs
                SET status = ?, model_name = ?, data = ?
                WHERE id = ?
                """,
                (
                    evaluation.status.value,
                    evaluation.model_name,
                    _run_json(evaluation),
                    str(evaluation.id),
                ),
            )

        return cursor.rowcount > 0

    def save_result(
        self,
        evaluation_id: UUID,
        position: int,
        result: EvaluationResult,
    ) -> bool:
        with self._connection() as connection:
            return _upsert_item(connection, evaluation_id, position, result)

    def save_item_error(
        self,
        evaluation_id: UUID,
        position: int,
        item_error: ItemError,
    ) -> bool:
        with self._connection() as connection:
            return _upsert_item(connection, evaluation_id, position, item_error)

    def get(self, evaluation_id: UUID) -> EvaluationRun | None:
        """Return an evaluation run by ID, if it exists."""

        with self._connection() as connection:
            row = connection.execute(
                "SELECT data FROM evaluation_runs WHERE id = ?",
                (str(evaluation_id),),
            ).fetchone()

            if row is None:
                return None

            evaluation = EvaluationRun.model_validate_json(row[0])
            _attach_items(connection, [evaluation])

        return evaluation

    def delete(
        self,
        evaluation_id: UUID,
    ) -> bool:
        """Delete an evaluation and its items."""

        with self._connection() as connection:
            connection.execute(
                "DELETE FROM evaluation_items WHERE run_id = ?",
                (str(evaluation_id),),
            )
            cursor = connection.execute(
                "DELETE FROM evaluation_runs WHERE id = ?",
                (str(evaluation_id),),
            )

        return cursor.rowcount > 0

    def list(
        self,
        status: EvaluationStatus | None = None,
        model_name: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[EvaluationRun]:
        """Return evaluations matching the provided filters."""

        conditions: list[str] = []
        parameters: list[Any] = []

        if status is not None:
            conditions.append("status = ?")
            parameters.append(status.value)

        if model_name is not None:
            conditions.append("model_name = ?")
            parameters.append(model_name)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # SQLite treats a negative LIMIT as "no limit".
        parameters.extend(
            [
                -1 if limit is None else limit,
                offset,
            ]
        )

        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT data FROM evaluation_runs {where} "
                "ORDER BY rowid LIMIT ? OFFSET ?",
                parameters,
            ).fetchall()

            evaluations = [EvaluationRun.model_validate_json(row[0]) for row in rows]
            _attach_items(connection, evaluations)

        return evaluations


def _run_json(evaluation: EvaluationRun) -> str:
    return evaluation.model_dump_json(exclude=_ITEM_FIELDS)


def _upsert_item(
    connection: sqlite3.Connection,
    evaluation_id: UUID,
    position: int,
    outcome: EvaluationResult | ItemError,
) -> bool:
    """Store an item outcome if its run exists; return whether it was stored."""

    kind = RESULT if isinstance(outcome, EvaluationResult) else ERROR

    cursor = connection.execute(
        _UPSERT_ITEM,
        (
            str(evaluation_id),
            str(outcome.item_id),
            position,
            kind,
            outcome.model_dump_json(),
            datetime.now(UTC).isoformat(),
            str(evaluation_id),
        ),
    )

    return cursor.rowcount > 0


def _attach_items(
    connection: sqlite3.Connection,
    evaluations: list[EvaluationRun],
) -> None:
    """Load the results and item errors of the given runs in one query."""

    if not evaluations:
        return

    by_id = {str(evaluation.id): evaluation for evaluation in evaluations}
    placeholders = ", ".join("?" for _ in by_id)

    rows = connection.execute(
        f"SELECT run_id, kind, data FROM evaluation_items "
        f"WHERE run_id IN ({placeholders}) ORDER BY run_id, position",
        list(by_id),
    ).fetchall()

    for run_id, kind, data in rows:
        evaluation = by_id[run_id]

        if kind == RESULT:
            evaluation.results.append(EvaluationResult.model_validate_json(data))
        else:
            evaluation.item_errors.append(ItemError.model_validate_json(data))


def _item_positions(evaluation: EvaluationRun) -> dict[UUID, int]:
    if evaluation.dataset is None:
        return {}

    return {item.id: index for index, item in enumerate(evaluation.dataset.items)}


def _migrate_embedded_items(connection: sqlite3.Connection) -> None:
    """Move results stored inside run JSON into evaluation_items.

    Databases created before per-item storage kept results in the run's JSON.
    Runs without embedded results are skipped, so this is cheap to repeat.
    """

    rows = connection.execute(
        "SELECT id, data FROM evaluation_runs "
        "WHERE json_array_length(data, '$.results') > 0 "
        "OR json_array_length(data, '$.item_errors') > 0"
    ).fetchall()

    now = datetime.now(UTC).isoformat()

    for run_id, data in rows:
        run = json.loads(data)
        dataset = run.get("dataset") or {}
        positions = {
            item["id"]: index for index, item in enumerate(dataset.get("items", []))
        }

        outcomes = [(RESULT, result) for result in run.get("results") or []] + [
            (ERROR, error) for error in run.get("item_errors") or []
        ]

        for index, (kind, outcome) in enumerate(outcomes):
            connection.execute(
                """
                INSERT OR IGNORE INTO evaluation_items
                    (run_id, item_id, position, kind, data, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    outcome["item_id"],
                    positions.get(outcome["item_id"], index),
                    kind,
                    json.dumps(outcome),
                    now,
                ),
            )

        for field in _ITEM_FIELDS:
            run.pop(field, None)

        connection.execute(
            "UPDATE evaluation_runs SET data = ? WHERE id = ?",
            (json.dumps(run), run_id),
        )
