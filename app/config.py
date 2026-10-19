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

# Embedding model and pass threshold for the semantic_similarity metric.
# The threshold depends on the model; recalibrate it when the model changes.
SEMANTIC_SIMILARITY_MODEL = os.getenv("SEMANTIC_SIMILARITY_MODEL", "nomic-embed-text")
SEMANTIC_SIMILARITY_THRESHOLD = float(
    os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.73")
)

# Attempts per model or judge call for errors that may be temporary, such as
# timeouts. Waits double after each failed attempt, starting at the backoff.
EVALUATION_MAX_ATTEMPTS = int(os.getenv("EVALUATION_MAX_ATTEMPTS", "3"))
EVALUATION_RETRY_BACKOFF_SECONDS = float(
    os.getenv("EVALUATION_RETRY_BACKOFF_SECONDS", "1.0")
)

# A run stops after this many items in a row could not be scored, so an
# unreachable provider does not cost a retry cycle for every remaining item.
EVALUATION_MAX_CONSECUTIVE_FAILURES = int(
    os.getenv("EVALUATION_MAX_CONSECUTIVE_FAILURES", "5")
)
