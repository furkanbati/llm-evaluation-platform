from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvaluationStatus(StrEnum):
    """Lifecycle states of an evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    # Finished, but some items could not be scored; see `item_errors`.
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


FINISHED_STATUSES = frozenset(
    {
        EvaluationStatus.COMPLETED,
        EvaluationStatus.COMPLETED_WITH_ERRORS,
        EvaluationStatus.FAILED,
    }
)


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

    @field_validator("items")
    @classmethod
    def validate_unique_item_ids(cls, items: list[DatasetItem]) -> list[DatasetItem]:
        """Item IDs identify stored results, so they must be unique."""

        ids = [item.id for item in items]

        if len(ids) != len(set(ids)):
            raise ValueError("dataset item ids must be unique")

        return items


class PassRule(StrEnum):
    """How the metric results of an item decide whether it passed."""

    ALL_METRICS = "all_metrics"
    ANY_METRIC = "any_metric"
    OVERALL_SCORE = "overall_score"


DEFAULT_PASS_THRESHOLD = 0.5


class AggregationOptions(BaseModel):
    """How metric results combine into an item score and pass decision.

    Metrics without a weight count with weight 1.0. `pass_threshold`
    applies only to the `overall_score` rule.
    """

    model_config = ConfigDict(extra="forbid")

    metric_weights: dict[str, float] = Field(default_factory=dict)
    pass_rule: PassRule = PassRule.ALL_METRICS
    pass_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("metric_weights")
    @classmethod
    def validate_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("metric weights cannot be negative")

        return weights

    @model_validator(mode="after")
    def resolve_threshold(self) -> "AggregationOptions":
        if self.pass_rule == PassRule.OVERALL_SCORE:
            if self.pass_threshold is None:
                self.pass_threshold = DEFAULT_PASS_THRESHOLD
        elif self.pass_threshold is not None:
            raise ValueError(
                "pass_threshold only applies to the overall_score pass rule"
            )

        return self

    def weight_for(self, metric_name: str) -> float:
        return self.metric_weights.get(metric_name, 1.0)


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
    aggregation: AggregationOptions | None = None

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

    @model_validator(mode="after")
    def validate_metric_weights(self) -> "EvaluationRequest":
        """Ensure weights refer to selected metrics and are not all zero."""

        if self.aggregation is None:
            return self

        unknown = sorted(set(self.aggregation.metric_weights) - set(self.metrics))

        if unknown:
            raise ValueError(
                f"metric_weights refer to metrics that are not selected: "
                f"{', '.join(unknown)}"
            )

        if not any(self.aggregation.weight_for(metric) > 0 for metric in self.metrics):
            raise ValueError("at least one selected metric needs a positive weight")

        return self


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
    # None for results stored before pass decisions were recorded.
    passed: bool | None = None


class ItemError(BaseModel):
    """A dataset item that could not be scored.

    `generated_output` is set when generation succeeded and scoring failed,
    e.g. when the LLM judge gave an invalid answer.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: UUID
    error_code: EvaluationErrorCode
    error: str = Field(min_length=1, max_length=1_000)
    attempts: int = Field(ge=1)
    generated_output: str | None = Field(default=None, max_length=20_000)


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
    aggregation: AggregationOptions = Field(default_factory=AggregationOptions)
    status: EvaluationStatus = EvaluationStatus.PENDING
    results: list[EvaluationResult] = Field(default_factory=list)
    item_errors: list[ItemError] = Field(default_factory=list)
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
    aggregation: AggregationOptions
    status: EvaluationStatus
    item_count: int = Field(ge=0)
    scored_items: int = Field(ge=0)
    errored_items: int = Field(ge=0)
    created_at: datetime
    completed_at: datetime | None
    error_code: EvaluationErrorCode | None
    error: str | None

    @classmethod
    def from_run(cls, run: EvaluationRun) -> "EvaluationRunListItem":
        # Runs stored before datasets were recorded only have results.
        item_count = (
            len(run.dataset.items)
            if run.dataset
            else len(run.results) + len(run.item_errors)
        )

        return cls(
            id=run.id,
            dataset_id=run.dataset_id,
            dataset_name=run.dataset.name if run.dataset else None,
            model_name=run.model_name,
            metrics=run.metrics,
            judge_model_name=run.judge_model_name,
            generation_options=run.generation_options,
            aggregation=run.aggregation,
            status=run.status,
            item_count=item_count,
            scored_items=len(run.results),
            errored_items=len(run.item_errors),
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
    """Aggregated statistics for an evaluation run.

    `passed_items` and `failed_items` split the scored items by the pass
    rule. Items that could not be scored are `errored_items`; items not
    processed yet, or skipped because the run stopped, are `pending_items`.
    Scores and rates are computed over scored items only, so read them
    together with `coverage`, the share of all items that were scored.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    total_items: int = Field(ge=0)
    scored_items: int = Field(ge=0)
    passed_items: int = Field(ge=0)
    failed_items: int = Field(ge=0)
    errored_items: int = Field(ge=0)
    pending_items: int = Field(ge=0)
    average_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    success_rate: float = Field(
        ge=0.0,
        le=1.0,
    )
    coverage: float = Field(
        ge=0.0,
        le=1.0,
    )
    metrics: dict[str, MetricSummary] = Field(
        default_factory=dict,
    )
