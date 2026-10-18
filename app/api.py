from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.responses import JSONResponse

from app.config import (
    APP_NAME,
    APP_VERSION,
    DATABASE_PATH,
    GENERATION_SEED,
    GENERATION_TEMPERATURE,
    JUDGE_MODEL_NAME,
    JUDGE_TEMPERATURE,
    OLLAMA_BASE_URL,
)
from app.evaluators.exact_match import ExactMatchEvaluator
from app.evaluators.llm_judge import LLMJudgeEvaluator
from app.evaluators.registry import EvaluatorRegistry
from app.evaluators.similarity import SimilarityEvaluator
from app.models import (
    ErrorResponse,
    EvaluationRequest,
    EvaluationRun,
    EvaluationStatus,
    EvaluationSummary,
    GenerationOptions,
    HealthResponse,
)
from app.services.evaluation_engine import EvaluationEngine
from app.services.evaluation_runner import EvaluationRunner
from app.services.evaluation_summary_service import (
    EvaluationSummaryService,
)
from app.services.model_client import ModelClient
from app.services.ollama_model_client import OllamaModelClient
from app.storage.base import BaseEvaluationRepository
from app.storage.evaluation_repository import EvaluationRepository
from app.storage.sqlite_evaluation_repository import (
    SQLiteEvaluationRepository,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    evaluation_runner.fail_interrupted_runs()
    yield


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "REST API for evaluating Large Language Model outputs against datasets."
    ),
    lifespan=lifespan,
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


def create_repository(
    database_path: str = DATABASE_PATH,
) -> BaseEvaluationRepository:
    """Create a SQLite repository, or an in-memory one if no path is set."""

    if database_path:
        return SQLiteEvaluationRepository(database_path)

    return EvaluationRepository()


def create_evaluation_runner(
    model_client: ModelClient | None = None,
    repository: BaseEvaluationRepository | None = None,
    judge_model_client: ModelClient | None = None,
) -> EvaluationRunner:
    """Create the evaluation runner with registered evaluators.

    When no clients are given, the judge uses its own Ollama client
    configured with JUDGE_TEMPERATURE so its decisions are stable.
    Otherwise the judge falls back to the provided model client.
    """

    registry = EvaluatorRegistry()

    registry.register(
        "exact_match",
        ExactMatchEvaluator(),
    )

    registry.register(
        "similarity",
        SimilarityEvaluator(),
    )

    if judge_model_client is None:
        if model_client is None:
            judge_model_client = OllamaModelClient(
                base_url=OLLAMA_BASE_URL,
                options={
                    "temperature": JUDGE_TEMPERATURE,
                },
            )
        else:
            judge_model_client = model_client

    if model_client is None:
        model_client = OllamaModelClient(
            base_url=OLLAMA_BASE_URL,
        )

    registry.register(
        "llm_judge",
        LLMJudgeEvaluator(
            model_client=judge_model_client,
            model_name=JUDGE_MODEL_NAME,
        ),
    )

    engine = EvaluationEngine(registry)

    return EvaluationRunner(
        engine=engine,
        model_client=model_client,
        repository=repository,
        judge_model_name=JUDGE_MODEL_NAME,
        default_generation_options=GenerationOptions(
            temperature=GENERATION_TEMPERATURE,
            seed=GENERATION_SEED,
        ),
    )


evaluation_repository = create_repository()

evaluation_runner = create_evaluation_runner(
    repository=evaluation_repository,
)

evaluation_summary_service = EvaluationSummaryService()


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ready",
    )


@app.get(
    "/metrics",
)
def list_metrics() -> dict[str, list[str]]:
    return {
        "metrics": [
            "exact_match",
            "similarity",
            "llm_judge",
        ],
    }


@app.post(
    "/evaluations",
    response_model=EvaluationRun,
    status_code=202,
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
    background_tasks: BackgroundTasks,
    response: Response,
) -> EvaluationRun:
    """Accept an evaluation and execute it in the background.

    The returned run is PENDING; poll GET /evaluations/{id} for the
    outcome.
    """

    runner = evaluation_runner

    try:
        evaluation_run = runner.create_run(
            dataset=request.dataset,
            model_name=request.model_name,
            metrics=request.metrics,
            generation_options=request.generation_options,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    background_tasks.add_task(
        runner.execute,
        evaluation_run=evaluation_run,
        dataset=request.dataset,
        metrics=request.metrics,
    )

    response.headers["Location"] = f"/evaluations/{evaluation_run.id}"

    return evaluation_run.model_copy(deep=True)


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
    limit: int = Query(default=100, ge=1, le=1_000),
    offset: int = Query(default=0, ge=0),
) -> list[EvaluationRun]:
    return evaluation_runner.repository.list(
        status=status,
        model_name=model_name,
        limit=limit,
        offset=offset,
    )


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
