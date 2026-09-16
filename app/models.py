from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvaluationStatus(StrEnum):
    """Lifecycle states of an evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DatasetItem(BaseModel):
    """A single input/expected-output pair used for evaluation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: UUID = Field(default_factory=uuid4)
    input: str = Field(min_length=1, max_length=10_000)
    expected_output: str = Field(min_length=1, max_length=10_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """A collection of dataset items used for evaluation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2_000)
    items: list[DatasetItem] = Field(min_length=1)


class EvaluationRequest(BaseModel):
    """Configuration describing an evaluation that should be executed."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    dataset: Dataset
    model_name: str = Field(min_length=1, max_length=200)
    metrics: list[str] = Field(min_length=1)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, metrics: list[str]) -> list[str]:
        """Ensure metric names are non-empty and unique."""

        normalized = [metric.strip() for metric in metrics]

        if any(not metric for metric in normalized):
            raise ValueError("metric names cannot be empty")

        if len(normalized) != len(set(normalized)):
            raise ValueError("metric names must be unique")

        return normalized

class MetricResult(BaseModel):
    """The result produced by a single evaluation metric."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    metric_name: str = Field(min_length=1, max_length=200)
    score: float = Field(ge=0.0, le=1.0)
    passed: bool


class EvaluationResult(BaseModel):
    """Evaluation result for a single dataset item."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    item_id: UUID
    generated_output: str = Field(min_length=1, max_length=20_000)
    metrics: list[MetricResult] = Field(min_length=1)
    overall_score: float = Field(ge=0.0, le=1.0)


class EvaluationRun(BaseModel):
    """A concrete execution of an evaluation request."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: UUID = Field(default_factory=uuid4)
    dataset_id: UUID
    model_name: str = Field(min_length=1, max_length=200)
    status: EvaluationStatus = EvaluationStatus.PENDING
    results: list[EvaluationResult] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: datetime | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    status: str = Field(min_length=1, max_length=100)