from app.models import (
    EvaluationResult,
    EvaluationRun,
    EvaluationSummary,
    MetricSummary,
)


class EvaluationSummaryService:
    """Builds aggregated statistics for evaluation runs."""

    def build(
        self,
        evaluation: EvaluationRun,
    ) -> EvaluationSummary:
        results = evaluation.results

        total_items = len(results)

        if total_items == 0:
            return EvaluationSummary(
                total_items=0,
                passed_items=0,
                failed_items=0,
                average_score=0.0,
                success_rate=0.0,
                metrics={},
            )

        passed_items = sum(1 for result in results if self._passed(result))

        failed_items = total_items - passed_items

        average_score = sum(result.overall_score for result in results) / total_items

        success_rate = passed_items / total_items

        metric_scores: dict[str, list[float]] = {}
        metric_passes: dict[str, list[bool]] = {}

        for result in results:
            for metric in result.metrics:
                metric_scores.setdefault(
                    metric.metric_name,
                    [],
                ).append(metric.score)

                metric_passes.setdefault(
                    metric.metric_name,
                    [],
                ).append(metric.passed)

        metrics = {}

        for metric_name in metric_scores:
            scores = metric_scores[metric_name]
            passes = metric_passes[metric_name]

            metrics[metric_name] = MetricSummary(
                average_score=(sum(scores) / len(scores)),
                pass_rate=(sum(passes) / len(passes)),
            )

        return EvaluationSummary(
            total_items=total_items,
            passed_items=passed_items,
            failed_items=failed_items,
            average_score=average_score,
            success_rate=success_rate,
            metrics=metrics,
        )

    @staticmethod
    def _passed(result: EvaluationResult) -> bool:
        # Results stored before pass rules existed have no decision; they
        # were judged by requiring every metric to pass.
        if result.passed is not None:
            return result.passed

        return all(metric.passed for metric in result.metrics)
