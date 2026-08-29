import os

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

APP_NAME = "LLM Evaluation Platform"
APP_VERSION = "1.0.0"