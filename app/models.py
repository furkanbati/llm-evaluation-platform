from datetime import UTC, datetime
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


class EvaluationErrorCode(StrEnum):
    """Categories of failures recorded on FAILED runs."""

    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_TIMEOUT = "model_timeout"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_PROVIDER_ERROR = "model_provider_error"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    INVALID_JUDGE_RESPONSE = "invalid_judge_response"
    INTERRUPTED = "interrupted"
    INTERNAL_ERROR = "internal_error"


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


class GenerationOptions(BaseModel):
    """Sampling settings for the evaluated model.

    Unset fields are not sent to the model provider.
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    seed: int | None = None

    def to_provider_options(self) -> dict[str, Any]:
        """Return the options that are set, in provider format."""
        return self.model_dump(exclude_none=True)


class EvaluationRequest(BaseModel):
    """Configuration describing an evaluation that should be executed."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    dataset: Dataset
    model_name: str = Field(min_length=1, max_length=200)
    metrics: list[str] = Field(min_length=1)
    generation_options: GenerationOptions | None = None

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
    dataset: Dataset | None = None
    model_name: str = Field(min_length=1, max_length=200)
    metrics: list[str] = Field(default_factory=list)
    judge_model_name: str | None = Field(default=None, max_length=200)
    generation_options: GenerationOptions = Field(default_factory=GenerationOptions)
    status: EvaluationStatus = EvaluationStatus.PENDING
    results: list[EvaluationResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    error_code: EvaluationErrorCode | None = None
    error: str | None = Field(default=None, max_length=1_000)


class EvaluationRunListItem(BaseModel):
    """An evaluation run without its dataset and per-item results.

    Used by list responses; the full run is available by ID.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID
    dataset_id: UUID
    dataset_name: str | None
    model_name: str
    metrics: list[str]
    judge_model_name: str | None
    generation_options: GenerationOptions
    status: EvaluationStatus
    item_count: int = Field(ge=0)
    created_at: datetime
    completed_at: datetime | None
    error_code: EvaluationErrorCode | None
    error: str | None

    @classmethod
    def from_run(cls, run: EvaluationRun) -> "EvaluationRunListItem":
        # Runs stored before datasets were recorded only have results.
        item_count = len(run.dataset.items) if run.dataset else len(run.results)

        return cls(
            id=run.id,
            dataset_id=run.dataset_id,
            dataset_name=run.dataset.name if run.dataset else None,
            model_name=run.model_name,
            metrics=run.metrics,
            judge_model_name=run.judge_model_name,
            generation_options=run.generation_options,
            status=run.status,
            item_count=item_count,
            created_at=run.created_at,
            completed_at=run.completed_at,
            error_code=run.error_code,
            error=run.error,
        )


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    status: str = Field(min_length=1, max_length=100)


class ErrorResponse(BaseModel):
    """Standard API error response."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    detail: str = Field(min_length=1, max_length=1_000)


class MetricSummary(BaseModel):
    """Aggregated statistics for a metric."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    average_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    pass_rate: float = Field(
        ge=0.0,
        le=1.0,
    )


class EvaluationSummary(BaseModel):
    """Aggregated statistics for an evaluation run."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    total_items: int = Field(ge=0)
    passed_items: int = Field(ge=0)
    failed_items: int = Field(ge=0)
    average_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    success_rate: float = Field(
        ge=0.0,
        le=1.0,
    )
    metrics: dict[str, MetricSummary] = Field(
        default_factory=dict,
    )
