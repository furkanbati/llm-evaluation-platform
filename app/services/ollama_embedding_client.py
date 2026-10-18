from app.errors import ModelProviderError
from app.services.embedding_client import EmbeddingClient
from app.services.ollama_model_client import post_to_ollama, read_field


class OllamaEmbeddingClient(EmbeddingClient):
    """Embedding client that uses Ollama's /api/embed endpoint."""

    def __init__(
        self,
        base_url: str,
    ) -> None:
        self._base_url = base_url.rstrip("/")

    def embed(
        self,
        model_name: str,
        texts: list[str],
    ) -> list[list[float]]:
        """Return one embedding vector for each text."""

        data = post_to_ollama(
            url=f"{self._base_url}/api/embed",
            payload={
                "model": model_name,
                "input": texts,
            },
            model_name=model_name,
        )

        embeddings = read_field(data, "embeddings")

        if not isinstance(embeddings, list):
            raise ModelProviderError("Ollama returned embeddings in an unknown format")

        if len(embeddings) != len(texts):
            raise ModelProviderError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
            )

        return [[float(value) for value in vector] for vector in embeddings]
