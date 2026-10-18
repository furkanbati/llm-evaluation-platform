from abc import ABC, abstractmethod

from app.models import MetricResult


class Evaluator(ABC):

    @abstractmethod
    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
        input_text: str | None = None,
    ) -> MetricResult:
        """Score a generated output against the expected output.

        `input_text` is the original dataset input. Evaluators that do
        not need it may ignore it.
        """


