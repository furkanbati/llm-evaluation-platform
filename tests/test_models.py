from uuid import UUID

import pytest
from pydantic import ValidationError

from app.models import (
    Dataset,
    DatasetItem,
    EvaluationRequest,
    EvaluationResult,
    EvaluationRun,
    EvaluationStatus,
    MetricResult,
)


def test_dataset_item_is_created_with_defaults() -> None:
    item = DatasetItem(
        input="  What is 2 + 2?  ",
        expected_output="4",
    )

    assert isinstance(item.id, UUID)
    assert item.input == "What is 2 + 2?"
    assert item.expected_output == "4"
    assert item.metadata == {}


def test_dataset_item_rejects_empty_input() -> None:
    with pytest.raises(ValidationError):
        DatasetItem(
            input="",
            expected_output="4",
        )


def test_dataset_item_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DatasetItem(
            input="What is 2 + 2?",
            expected_output="4",
            unexpected="value",
        )


def test_dataset_requires_at_least_one_item() -> None:
    with pytest.raises(ValidationError):
        Dataset(
            name="Math Dataset",
            items=[],
        )


def test_dataset_accepts_dataset_items() -> None:
    dataset = Dataset(
        name="Math Dataset",
        description="Basic math questions",
        items=[
            DatasetItem(
                input="2 + 2",
                expected_output="4",
            )
        ],
    )

    assert len(dataset.items) == 1
    assert dataset.items[0].expected_output == "4"


def test_evaluation_request_validates_metrics() -> None:
    request = EvaluationRequest(
        dataset=Dataset(
            name="Math Dataset",
            items=[
                DatasetItem(
                    input="2 + 2",
                    expected_output="4",
                )
            ],
        ),
        model_name="llama3",
        metrics=["exact_match", "semantic_similarity"],
    )

    assert request.model_name == "llama3"
    assert request.metrics == [
        "exact_match",
        "semantic_similarity",
    ]


def test_evaluation_request_strips_metric_names() -> None:
    request = EvaluationRequest(
        dataset=Dataset(
            name="Math Dataset",
            items=[
                DatasetItem(
                    input="2 + 2",
                    expected_output="4",
                )
            ],
        ),
        model_name="llama3",
        metrics=[
            " exact_match ",
            " semantic_similarity ",
        ],
    )

    assert request.metrics == [
        "exact_match",
        "semantic_similarity",
    ]


def test_evaluation_request_rejects_duplicate_metrics() -> None:
    with pytest.raises(ValidationError):
        EvaluationRequest(
            dataset=Dataset(
                name="Math Dataset",
                items=[
                    DatasetItem(
                        input="2 + 2",
                        expected_output="4",
                    )
                ],
            ),
            model_name="llama3",
            metrics=[
                "exact_match",
                "exact_match",
            ],
        )


def test_evaluation_request_rejects_empty_metric_name() -> None:
    with pytest.raises(ValidationError):
        EvaluationRequest(
            dataset=Dataset(
                name="Math Dataset",
                items=[
                    DatasetItem(
                        input="2 + 2",
                        expected_output="4",
                    )
                ],
            ),
            model_name="llama3",
            metrics=[""],
        )


def test_metric_result_accepts_valid_score() -> None:
    result = MetricResult(
        metric_name="exact_match",
        score=1.0,
        passed=True,
    )

    assert result.score == 1.0
    assert result.passed is True


@pytest.mark.parametrize("score", [-0.1, 1.1, 2.0])
def test_metric_result_rejects_invalid_score(score: float) -> None:
    with pytest.raises(ValidationError):
        MetricResult(
            metric_name="exact_match",
            score=score,
            passed=False,
        )


def test_evaluation_result_contains_metric_results() -> None:
    item_id = UUID("12345678-1234-5678-1234-567812345678")

    result = EvaluationResult(
        item_id=item_id,
        generated_output="Paris",
        metrics=[
            MetricResult(
                metric_name="exact_match",
                score=1.0,
                passed=True,
            )
        ],
        overall_score=1.0,
    )

    assert result.item_id == item_id
    assert result.generated_output == "Paris"
    assert len(result.metrics) == 1


def test_evaluation_run_has_correct_defaults() -> None:
    dataset_id = UUID("12345678-1234-5678-1234-567812345678")

    run = EvaluationRun(
        dataset_id=dataset_id,
        model_name="llama3",
    )

    assert isinstance(run.id, UUID)
    assert run.status == EvaluationStatus.PENDING
    assert run.results == []
    assert run.created_at.tzinfo is not None
    assert run.completed_at is None


def test_evaluation_run_accepts_results() -> None:
    dataset_id = UUID("12345678-1234-5678-1234-567812345678")
    item_id = UUID("87654321-4321-8765-4321-876543218765")

    result = EvaluationResult(
        item_id=item_id,
        generated_output="Paris",
        metrics=[
            MetricResult(
                metric_name="exact_match",
                score=1.0,
                passed=True,
            )
        ],
        overall_score=1.0,
    )

    run = EvaluationRun(
        dataset_id=dataset_id,
        model_name="llama3",
        status=EvaluationStatus.COMPLETED,
        results=[result],
    )

    assert run.status == EvaluationStatus.COMPLETED
    assert len(run.results) == 1
    assert run.results[0].generated_output == "Paris"


def test_models_reject_invalid_types() -> None:
    with pytest.raises(ValidationError):
        MetricResult(
            metric_name="exact_match",
            score="not-a-score",
            passed=True,
        )