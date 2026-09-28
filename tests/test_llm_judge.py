
from app.config import JUDGE_MODEL_NAME, OLLAMA_BASE_URL
from app.evaluators.llm_judge import LLMJudgeEvaluator
from app.services.model_client import ModelClient
from app.services.ollama_model_client import OllamaModelClient


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


def test_llm_judge_works_with_real_ollama() -> None:
    model_client = OllamaModelClient(
        base_url=OLLAMA_BASE_URL,
    )

    evaluator = LLMJudgeEvaluator(
        model_client=model_client,
        model_name=JUDGE_MODEL_NAME,
    )

    result = evaluator.evaluate(
        expected_output="Paris",
        generated_output="The capital of France is Paris.",
    )

    assert result.metric_name == "llm_judge"
    assert result.score in [0.0, 1.0]
    assert result.passed is (result.score == 1.0)

def test_llm_judge_accepts_yes_with_extra_text() -> None:
    evaluator = LLMJudgeEvaluator(
        model_client=FakeJudgeModelClient(),
        model_name="judge-model",
    )

    result = evaluator._parse_judgement(
        "YES, the answer is correct.",
    )

    assert result.score == 1.0
    assert result.passed is True


def test_llm_judge_accepts_no_with_extra_text() -> None:
    evaluator = LLMJudgeEvaluator(
        model_client=FakeJudgeModelClient(),
        model_name="judge-model",
    )

    result = evaluator._parse_judgement(
        "NO, the answer is incorrect.",
    )

    assert result.score == 0.0
    assert result.passed is False


