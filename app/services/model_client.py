from abc import ABC, abstractmethod

from app.models import GenerationOptions


class ModelClient(ABC):
    """Interface for clients that generate LLM outputs."""

    @abstractmethod
    def generate(
        self,
        model_name: str,
        inputs: list[str],
        options: GenerationOptions | None = None,
    ) -> list[str]:
        """Generate one output for each input.

        `options` overrides the client's default sampling settings.
        """
