import json
from pathlib import Path

import pytest

from app.config import JUDGE_MODEL_NAME, JUDGE_TEMPERATURE, OLLAMA_BASE_URL
from app.evaluators.llm_judge import LLMJudgeEvaluator
from app.models import GenerationOptions
from app.services.model_client import ModelClient
from app.services.ollama_model_client import OllamaModelClient

CASES_PATH = Path(__file__).parent / "data" / "llm_judge_cases.json"

GROUPS = {
    "paraphrase",
    "wrong",
    "context",
    "injection",
    "verbose",
}

# Measured with llama3 at temperature 0.0 over three Ollama sessions: the
# current prompt scored 22/23, accepting 9/10 correct answers and rejecting
# 13/13 wrong ones. The previous prompt, whose untagged examples did not match
# the tagged input, scored 19-20/23 and accepted only 6-7/10 correct answers.
# Decisions on borderline cases can change between sessions even at
# temperature 0, so the thresholds leave a one-case margin.
MIN_ACCURACY = 0.85
MIN_ACCEPTANCE_RATE = 0.8
MIN_REJECTION_RATE = 0.9


def load_cases() -> list[dict[str, str]]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def test_llm_judge_cases_are_well_formed() -> None:
    cases = load_cases()

    ids = [case["id"] for case in cases]

    assert len(ids) == len(set(ids))

    for case in cases:
        assert case["group"] in GROUPS
        assert case["expected_decision"] in {"YES", "NO"}

        for field in (
            "question",
            "expected_output",
            "generated_output",
        ):
            assert case[field].strip()

    decisions = {case["expected_decision"] for case in cases}

    assert decisions == {"YES", "NO"}


class RecordingModelClient(ModelClient):
    """Keeps the raw judge response for the failure report."""

    def __init__(self, inner: ModelClient) -> None:
        self._inner = inner
        self.last_output = ""

    def generate(
        self,
        model_name: str,
        inputs: list[str],
        options: GenerationOptions | None = None,
    ) -> list[str]:
        outputs = self._inner.generate(
            model_name=model_name,
            inputs=inputs,
            options=options,
        )
        self.last_output = outputs[0]
        return outputs


@pytest.mark.integration
def test_llm_judge_accuracy_on_labeled_cases() -> None:
    model_client = RecordingModelClient(
        OllamaModelClient(
            base_url=OLLAMA_BASE_URL,
            options={
                "temperature": JUDGE_TEMPERATURE,
            },
        )
    )

    evaluator = LLMJudgeEvaluator(
        model_client=model_client,
        model_name=JUDGE_MODEL_NAME,
    )

    cases = load_cases()
    failures: list[str] = []
    accepted = 0
    correct_answers = 0
    rejected = 0
    wrong_answers = 0

    for case in cases:
        expected_pass = case["expected_decision"] == "YES"

        try:
            passed = evaluator.evaluate(
                expected_output=case["expected_output"],
                generated_output=case["generated_output"],
                input_text=case["question"],
            ).passed
        except ValueError:
            # An unparseable judgement fails the whole run in production.
            passed = None

        if expected_pass:
            correct_answers += 1
            accepted += passed is True
        else:
            wrong_answers += 1
            rejected += passed is False

        if passed is not expected_pass:
            failures.append(
                f"{case['id']}: expected {case['expected_decision']}, "
                f"judge said {model_client.last_output[:60]!r}"
            )

    accuracy = (len(cases) - len(failures)) / len(cases)
    acceptance_rate = accepted / correct_answers
    rejection_rate = rejected / wrong_answers

    report = (
        f"accuracy {len(cases) - len(failures)}/{len(cases)}, "
        f"correct answers accepted {accepted}/{correct_answers}, "
        f"wrong answers rejected {rejected}/{wrong_answers}\n" + "\n".join(failures)
    )

    assert accuracy >= MIN_ACCURACY, report
    assert acceptance_rate >= MIN_ACCEPTANCE_RATE, report
    assert rejection_rate >= MIN_REJECTION_RATE, report
