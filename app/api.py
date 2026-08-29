from fastapi import FastAPI

from app.config import APP_NAME, APP_VERSION
from app.models import HealthResponse

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(status="ready")