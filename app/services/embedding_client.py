from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    """Interface for clients that turn texts into embedding vectors."""

    @abstractmethod
    def embed(
        self,
        model_name: str,
        texts: list[str],
    ) -> list[list[float]]:
        """Return one embedding vector for each text."""
