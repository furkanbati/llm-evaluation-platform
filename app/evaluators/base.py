from abc import ABC, abstractmethod

from app.models import MetricResult


class Evaluator(ABC):

    @abstractmethod
    def evaluate(
        self,
        expected_output: str,
        generated_output: str,
    ) -> MetricResult:
        pass