from abc import ABC, abstractmethod


class ModelClient(ABC):
    """Interface for clients that generate LLM outputs."""

    @abstractmethod
    def generate(
        self,
        model_name: str,
        inputs: list[str],
    ) -> list[str]:
        """Generate one output for each input."""
        pass