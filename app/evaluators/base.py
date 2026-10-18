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

    def validate_expected_output(self, expected_output: str) -> None:
        """Raise ValueError if this metric cannot score the expected output.

        Called for every dataset item before a run is created, so invalid
        datasets are rejected without calling the model.
        """
        # Optional hook: most metrics can score any expected output.
        return None
