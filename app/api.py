from fastapi import FastAPI, HTTPException

from app.config import APP_NAME, APP_VERSION
from app.evaluators.exact_match import ExactMatchEvaluator
from app.evaluators.registry import EvaluatorRegistry
from app.models import (
    EvaluationRequest,
    EvaluationRun,
    HealthResponse,
)
from app.services.evaluation_engine import EvaluationEngine
from app.services.evaluation_runner import EvaluationRunner


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
)


def create_evaluation_runner() -> EvaluationRunner:
    """Create the evaluation runner with registered evaluators."""

    registry = EvaluatorRegistry()
    registry.register(
        "exact_match",
        ExactMatchEvaluator(),
    )

    engine = EvaluationEngine(registry)

    return EvaluationRunner(engine)


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
            generated_outputs=request.generated_outputs,
            metrics=request.metrics,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc