from app.models import EvaluationRun


class EvaluationSummaryService:
    """Build summary statistics from evaluation runs."""

    def build(
        self,
        evaluation_run: EvaluationRun,
    ) -> dict:
        results = evaluation_run.results

        if not results:
            return {
                "total_items": 0,
                "passed_items": 0,
                "failed_items": 0,
                "average_score": 0.0,
                "success_rate": 0.0,
            }

        total_items = len(results)

        passed_items = sum(
            1
            for result in results
            if all(
                metric.passed
                for metric in result.metrics
            )
        )

        failed_items = total_items - passed_items

        average_score = (
            sum(
                result.overall_score
                for result in results
            )
            / total_items
        )

        success_rate = (
            passed_items
            / total_items
        )

        return {
            "total_items": total_items,
            "passed_items": passed_items,
            "failed_items": failed_items,
            "average_score": average_score,
            "success_rate": success_rate,
        }

