from app.models import MetricResult
from app.services.model_client import ModelClient

from .base import Evaluator


class LLMJudgeEvaluator(Evaluator):

    def __init__(
        self,
        model_client: ModelClient,
        model_name: str,
    ) -> None:
        self._model_client = model_client
        self._model_name = model_name

    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
    ) -> MetricResult:
        prompt = (
            "Evaluate the generated answer against the expected answer.\n"
            f"Expected answer: {expected_output}\n"
            f"Generated answer: {generated_output}\n\n"
            "Respond with only YES or NO."
        )

        judgement = self._model_client.generate(
            model_name=self._model_name,
            inputs=[prompt],
        )[0]

        return self._parse_judgement(
            judgement,
        )

    def _parse_judgement(
        self,
        judgement: str,
    ) -> MetricResult:
        normalized = judgement.strip().upper()

        if normalized == "YES":
            return MetricResult(
                metric_name="llm_judge",
                score=1.0,
                passed=True,
            )

        if normalized == "NO":
            return MetricResult(
                metric_name="llm_judge",
                score=0.0,
                passed=False,
            )

        raise ValueError(
            f"Invalid LLM judge response: {judgement}"
        )

