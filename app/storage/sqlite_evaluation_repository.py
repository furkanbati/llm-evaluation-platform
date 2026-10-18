import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

from app.models import EvaluationRun, EvaluationStatus
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
"""


class SQLiteEvaluationRepository(BaseEvaluationRepository):
    """SQLite repository that persists evaluation runs as JSON."""

    def __init__(self, path: str) -> None:
        self._path = path

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)

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
        """Insert or update an evaluation run and return it."""

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
                    evaluation.model_dump_json(),
                ),
            )

        return evaluation

    def get(self, evaluation_id: UUID) -> EvaluationRun | None:
        """Return an evaluation run by ID, if it exists."""

        with self._connection() as connection:
            row = connection.execute(
                "SELECT data FROM evaluation_runs WHERE id = ?",
                (str(evaluation_id),),
            ).fetchone()

        if row is None:
            return None

        return EvaluationRun.model_validate_json(row[0])

    def delete(
        self,
        evaluation_id: UUID,
    ) -> bool:
        """Delete an evaluation by id."""

        with self._connection() as connection:
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

        return [EvaluationRun.model_validate_json(row[0]) for row in rows]
