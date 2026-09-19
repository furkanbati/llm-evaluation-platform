from uuid import uuid4

from app.models import (
    EvaluationResult,
    EvaluationRun,
    MetricResult,
)


def create_result(
    score: float,
    passed: bool,
) -> EvaluationResult:
    return EvaluationResult(
        item_id=uuid4(),
        generated_output="output",
        metrics=[
            MetricResult(
                metric_name="exact_match",
                score=score,
                passed=passed,
            )
        ],
        overall_score=score,
    )


def test_summary_calculates_basic_statistics() -> None:
    from app.services.evaluation_summary_service import (
        EvaluationSummaryService,
    )

    evaluation = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
        results=[
            create_result(1.0, True),
            create_result(0.5, False),
            create_result(0.0, False),
        ],
    )

    summary = EvaluationSummaryService().build(
        evaluation,
    )

    assert summary["total_items"] == 3
    assert summary["passed_items"] == 1
    assert summary["failed_items"] == 2
    assert summary["average_score"] == 0.5
    assert summary["success_rate"] == (
        1 / 3
    )


def test_summary_handles_empty_results() -> None:
    from app.services.evaluation_summary_service import (
        EvaluationSummaryService,
    )

    evaluation = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
        results=[],
    )

    summary = EvaluationSummaryService().build(
        evaluation,
    )

    assert summary["total_items"] == 0
    assert summary["passed_items"] == 0
    assert summary["failed_items"] == 0
    assert summary["average_score"] == 0.0
    assert summary["success_rate"] == 0.0


