import os

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://ollama:11434",
)

APP_NAME = "LLM Evaluation Platform"
APP_VERSION = "1.0.0"

JUDGE_MODEL_NAME = os.getenv(
    "JUDGE_MODEL_NAME",
    "llama3",
)

# Empty string keeps evaluations in memory only.
DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    "data/evaluations.db",
)

JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))

# Default sampling settings for the evaluated model. An empty value leaves the
# setting to the model provider. Requests can override both.
_generation_temperature = os.getenv("GENERATION_TEMPERATURE", "0.0")
GENERATION_TEMPERATURE = (
    float(_generation_temperature) if _generation_temperature else None
)

_generation_seed = os.getenv("GENERATION_SEED", "")
GENERATION_SEED = int(_generation_seed) if _generation_seed else None

# Evaluations that run at once; others wait in a queue as PENDING.
MAX_CONCURRENT_EVALUATIONS = int(os.getenv("MAX_CONCURRENT_EVALUATIONS", "2"))

# Evaluations that can be queued or running before new requests get 503.
MAX_QUEUED_EVALUATIONS = int(os.getenv("MAX_QUEUED_EVALUATIONS", "100"))
