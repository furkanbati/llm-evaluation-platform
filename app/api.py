from fastapi import FastAPI, HTTPException

from app.config import APP_NAME, APP_VERSION, OLLAMA_BASE_URL
from app.evaluators.exact_match import ExactMatchEvaluator
from app.evaluators.registry import EvaluatorRegistry
from app.models import (
    EvaluationRequest,
    EvaluationRun,
    HealthResponse,
)
from app.services.evaluation_engine import EvaluationEngine
from app.services.evaluation_runner import EvaluationRunner
from app.services.model_client import ModelClient
from app.services.ollama_model_client import OllamaModelClient


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
)


def create_evaluation_runner(
    model_client: ModelClient | None = None,
) -> EvaluationRunner:
    """Create the evaluation runner with registered evaluators."""

    registry = EvaluatorRegistry()
    registry.register(
        "exact_match",
        ExactMatchEvaluator(),
    )

    engine = EvaluationEngine(registry)

    if model_client is None:
        model_client = OllamaModelClient(
            base_url=OLLAMA_BASE_URL,
        )

    return EvaluationRunner(
        engine=engine,
        model_client=model_client,
    )


evaluation_runner = create_evaluation_runner()


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(status="ready")


@app.post(
    "/evaluations",
    response_model=EvaluationRun,
)
def create_evaluation(
    request: EvaluationRequest,
) -> EvaluationRun:
    try:
        return evaluation_runner.run(
            dataset=request.dataset,
            model_name=request.model_name,
            metrics=request.metrics,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc