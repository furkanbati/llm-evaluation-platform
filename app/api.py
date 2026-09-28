from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from app.config import APP_NAME, APP_VERSION, OLLAMA_BASE_URL
from app.evaluators.exact_match import ExactMatchEvaluator
from app.evaluators.registry import EvaluatorRegistry
from app.models import (
    ErrorResponse,
    EvaluationRequest,
    EvaluationRun,
    EvaluationStatus,
    EvaluationSummary,
    HealthResponse,
)
from app.services.evaluation_engine import EvaluationEngine
from app.services.evaluation_runner import EvaluationRunner
from app.services.evaluation_summary_service import (
    EvaluationSummaryService,
)
from app.services.model_client import ModelClient
from app.services.ollama_model_client import OllamaModelClient
from app.storage.evaluation_repository import EvaluationRepository
from app.evaluators.similarity import SimilarityEvaluator

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "REST API for evaluating Large Language Model "
        "outputs against datasets."
    ),
)


@app.exception_handler(Exception)
def handle_unexpected_exception(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
        },
    )


def create_evaluation_runner(
    model_client: ModelClient | None = None,
    repository: EvaluationRepository | None = None,
) -> EvaluationRunner:
    """Create the evaluation runner with registered evaluators."""

    registry = EvaluatorRegistry()

    registry.register(
        "exact_match",
        ExactMatchEvaluator(),
    )

    registry.register(
        "similarity",
        SimilarityEvaluator(),
    )

    engine = EvaluationEngine(registry)

    if model_client is None:
        model_client = OllamaModelClient(
            base_url=OLLAMA_BASE_URL,
        )

    return EvaluationRunner(
        engine=engine,
        model_client=model_client,
        repository=repository,
    )


evaluation_repository = EvaluationRepository()

evaluation_runner = create_evaluation_runner(
    repository=evaluation_repository,
)

evaluation_summary_service = (
    EvaluationSummaryService()
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ready",
    )


@app.post(
    "/evaluations",
    response_model=EvaluationRun,
    responses={
        400: {
            "model": ErrorResponse,
        },
        500: {
            "model": ErrorResponse,
        },
    },
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


@app.get(
    "/evaluations",
    response_model=list[EvaluationRun],
    responses={
        500: {
            "model": ErrorResponse,
        },
    },
)

def list_evaluations(
    status: EvaluationStatus | None = None,
    model_name: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[EvaluationRun]:
    evaluations = evaluation_runner.repository.list(
        status=status,
        model_name=model_name,
    )

    return evaluations[offset : offset + limit]


@app.get(
    "/evaluations/{evaluation_id}",
    response_model=EvaluationRun,
    responses={
        404: {
            "model": ErrorResponse,
        },
        500: {
            "model": ErrorResponse,
        },
    },
)
def get_evaluation(
    evaluation_id: UUID,
) -> EvaluationRun:
    evaluation = evaluation_runner.repository.get(
        evaluation_id,
    )

    if evaluation is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found",
        )

    return evaluation


@app.get(
    "/evaluations/{evaluation_id}/summary",
    response_model=EvaluationSummary,
    responses={
        404: {
            "model": ErrorResponse,
        },
    },
)
def get_evaluation_summary(
    evaluation_id: UUID,
) -> EvaluationSummary:
    evaluation = evaluation_runner.repository.get(
        evaluation_id,
    )

    if evaluation is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found",
        )

    return evaluation_summary_service.build(
        evaluation,
    )


@app.delete(
    "/evaluations/{evaluation_id}",
    status_code=204,
    responses={
        404: {
            "model": ErrorResponse,
        },
    },
)
def delete_evaluation(
    evaluation_id: UUID,
) -> Response:
    deleted = evaluation_repository.delete(
        evaluation_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found",
        )

    return Response(
        status_code=204,
    )



