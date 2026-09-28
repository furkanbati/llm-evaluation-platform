from app.evaluators.llm_judge import LLMJudgeEvaluator
from app.services.model_client import ModelClient


class FakeJudgeModelClient(ModelClient):
    def generate(
        self,
        model_name: str,
        inputs: list[str],
    ) -> list[str]:
        return ["YES"]


def test_llm_judge_evaluates_generated_output() -> None:
    evaluator = LLMJudgeEvaluator(
        model_client=FakeJudgeModelClient(),
        model_name="judge-model",
    )

    result = evaluator.evaluate(
        expected_output="Paris",
        generated_output="The capital of France is Paris.",
    )

    assert result.metric_name == "llm_judge"
    assert result.score == 1.0
    assert result.passed is True