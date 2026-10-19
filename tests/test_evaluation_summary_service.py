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

    assert summary.total_items == 3
    assert summary.passed_items == 1
    assert summary.failed_items == 2
    assert summary.average_score == 0.5
    assert summary.success_rate == (1 / 3)


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

    assert summary.total_items == 0
    assert summary.passed_items == 0
    assert summary.failed_items == 0
    assert summary.average_score == 0
    assert summary.success_rate == 0


def test_summary_calculates_metric_statistics() -> None:
    from app.services.evaluation_summary_service import (
        EvaluationSummaryService,
    )

    evaluation = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
        results=[
            EvaluationResult(
                item_id=uuid4(),
                generated_output="output",
                metrics=[
                    MetricResult(
                        metric_name="exact_match",
                        score=1.0,
                        passed=True,
                    )
                ],
                overall_score=1.0,
            ),
            EvaluationResult(
                item_id=uuid4(),
                generated_output="output",
                metrics=[
                    MetricResult(
                        metric_name="exact_match",
                        score=0.0,
                        passed=False,
                    )
                ],
                overall_score=0.0,
            ),
        ],
    )

    summary = EvaluationSummaryService().build(
        evaluation,
    )

    assert summary.metrics["exact_match"].average_score == 0.5

    assert summary.metrics["exact_match"].pass_rate == 0.5


def test_summary_uses_stored_pass_decision() -> None:
    from app.services.evaluation_summary_service import (
        EvaluationSummaryService,
    )

    result = create_result(score=0.0, passed=False)
    result.passed = True

    evaluation = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
        results=[result],
    )

    summary = EvaluationSummaryService().build(evaluation)

    assert summary.passed_items == 1
    assert summary.success_rate == 1.0


def test_summary_falls_back_to_all_metrics_for_old_results() -> None:
    from app.services.evaluation_summary_service import (
        EvaluationSummaryService,
    )

    evaluation = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
        results=[
            create_result(score=1.0, passed=True),
            create_result(score=0.0, passed=False),
        ],
    )

    assert evaluation.results[0].passed is None

    summary = EvaluationSummaryService().build(evaluation)

    assert summary.passed_items == 1


def test_summary_reports_coverage_of_errored_and_pending_items() -> None:
    from app.models import Dataset, DatasetItem, EvaluationErrorCode, ItemError
    from app.services.evaluation_summary_service import EvaluationSummaryService

    dataset = Dataset(
        name="items",
        items=[
            DatasetItem(input=f"q{index}", expected_output="a") for index in range(5)
        ],
    )

    evaluation = EvaluationRun(
        dataset_id=dataset.id,
        dataset=dataset,
        model_name="llama3",
        results=[
            create_result(score=1.0, passed=True),
            create_result(score=0.0, passed=False),
        ],
        item_errors=[
            ItemError(
                item_id=dataset.items[2].id,
                error_code=EvaluationErrorCode.MODEL_TIMEOUT,
                error="The model provider did not respond in time.",
                attempts=3,
            )
        ],
    )

    summary = EvaluationSummaryService().build(evaluation)

    assert summary.total_items == 5
    assert summary.scored_items == 2
    assert summary.passed_items == 1
    assert summary.failed_items == 1
    assert summary.errored_items == 1
    assert summary.pending_items == 2
    # Rates cover scored items only; coverage shows how many those are.
    assert summary.success_rate == 0.5
    assert summary.average_score == 0.5
    assert summary.coverage == 0.4


def test_summary_without_scored_items_has_zero_coverage() -> None:
    from app.models import EvaluationErrorCode, ItemError
    from app.services.evaluation_summary_service import EvaluationSummaryService

    evaluation = EvaluationRun(
        dataset_id=uuid4(),
        model_name="llama3",
        item_errors=[
            ItemError(
                item_id=uuid4(),
                error_code=EvaluationErrorCode.MODEL_UNAVAILABLE,
                error="Could not connect to the model provider.",
                attempts=1,
            )
        ],
    )

    summary = EvaluationSummaryService().build(evaluation)

    assert summary.total_items == 1
    assert summary.errored_items == 1
    assert summary.scored_items == 0
    assert summary.coverage == 0.0
    assert summary.success_rate == 0.0
