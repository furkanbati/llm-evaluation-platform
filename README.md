# LLM Evaluation Platform

A production-oriented REST API for evaluating Large Language Model outputs against predefined datasets.

The platform provides a structured evaluation workflow with pluggable evaluators, evaluation run lifecycle tracking, result storage, filtering, pagination, and evaluation summaries.

---

# Features

* Multiple evaluation metrics
* Exact Match evaluation
* String Similarity evaluation
* LLM-as-a-Judge evaluation
* Pluggable evaluator registry
* Evaluation run lifecycle tracking
* Persistent SQLite storage of evaluation runs
* Evaluation summaries and metric statistics
* Evaluation filtering by status and model
* Pagination for evaluation listing
* REST API
* Ollama model integration
* Docker support
* Automated tests

---

# Architecture

The evaluation flow is:

```text
                    +------------------+
                    |      Client      |
                    +------------------+
                              |
                              v
                       FastAPI REST API
                              |
                              v
                    EvaluationRunner
                              |
                    +---------+---------+
                    |                   |
                    v                   v
             ModelClient          EvaluationEngine
                    |                   |
                    v                   v
                  Ollama        EvaluatorRegistry
                                        |
                         +--------------+--------------+
                         |              |              |
                         v              v              v
                    Exact Match    Similarity     LLM Judge
                         |              |              |
                         +--------------+--------------+
                                        |
                                        v
                                  EvaluationRun
                                        |
                                        v
                              EvaluationRepository
                                        |
                                        v
                                   Summary
```

The architecture separates:

```text
EvaluationRunner
        ↓
Runs the evaluation

EvaluationEngine
        ↓
Executes the selected metrics

EvaluatorRegistry
        ↓
Finds the evaluator for each metric

EvaluationRepository
        ↓
Stores evaluation runs

EvaluationSummaryService
        ↓
Builds aggregate evaluation statistics
```

This separation keeps evaluation orchestration, metric implementations, and storage responsibilities independent.

---

# Evaluation Workflow

1. A client submits a dataset, model name, and selected metrics.
2. `EvaluationRunner` validates the request and stores a `PENDING` run.
3. The API responds immediately with `202 Accepted` and the run is executed in the background.
4. The configured model client generates outputs for the dataset items.
5. `EvaluationEngine` executes the selected evaluators.
6. Each evaluator produces a `MetricResult`.
7. The evaluation run is marked `COMPLETED` or `FAILED` in the repository.
8. The client polls the run and reads its results and summary statistics.

The evaluation lifecycle is:

```text
PENDING
   ↓
RUNNING
   ↓
COMPLETED
```

When an evaluation fails:

```text
PENDING
   ↓
RUNNING
   ↓
FAILED
```

Failed runs include an `error` field describing the cause, such as a model provider error or an invalid LLM judge response.

Invalid requests, such as unknown metrics, are rejected with `400 Bad Request` before a run is created and before any model calls are made.

---

# Evaluation Metrics

The current platform supports three metrics.

## Exact Match

Compares the expected output and generated output after trimming surrounding whitespace.

```text
Expected: 4
Generated: 4
Score: 1.0
```

```text
Expected: 4
Generated: The answer is 4.
Score: 0.0
```

Exact Match is useful for deterministic outputs but is intentionally strict.

---

## Similarity

Calculates string similarity between the expected and generated outputs.

The current implementation uses Python's `SequenceMatcher` and considers scores of `0.8` or higher as passing.

```text
Expected: The capital of France is Paris.
Generated: The capital of France is Paris.
Score: 1.0
```

This metric measures textual similarity rather than embedding-based semantic similarity.

---

## LLM Judge

Uses an LLM to evaluate whether the generated answer is semantically and factually correct compared with the expected answer.

Example:

```text
Expected: Paris
Generated: The capital of France is Paris.
Decision: YES
```

```text
Expected: Paris
Generated: The capital of France is London.
Decision: NO
```

The evaluator uses explicit positive and negative examples in its judging prompt and returns a binary score.

The judge also receives the original dataset input, so answers are judged in the context of the question.

The question, expected answer, and generated answer are wrapped in XML tags and the judge is instructed to treat their contents as data, never as instructions. Tags inside the content are removed so a generated answer cannot break out of its block.

The judge runs with its own Ollama client at temperature `0.0` by default so its decisions are stable. This can be changed with the `JUDGE_TEMPERATURE` environment variable.

---

# Evaluator Registry

Metrics are registered through an `EvaluatorRegistry`.

```text
Metric name
     ↓
EvaluatorRegistry
     ↓
Evaluator implementation
```

Current registrations:

```text
exact_match → ExactMatchEvaluator
similarity  → SimilarityEvaluator
llm_judge   → LLMJudgeEvaluator
```

This design allows new evaluators to be added without changing the core evaluation engine.

---

# API

## Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ready"
}
```

---

## List Available Metrics

```http
GET /metrics
```

Example response:

```json
{
  "metrics": [
    "exact_match",
    "similarity",
    "llm_judge"
  ]
}
```

---

## Create Evaluation

```http
POST /evaluations
```

Example request:

```json
{
  "dataset": {
    "name": "math-test",
    "items": [
      {
        "input": "2 + 2",
        "expected_output": "4"
      }
    ]
  },
  "model_name": "llama3",
  "metrics": [
    "exact_match",
    "similarity",
    "llm_judge"
  ]
}
```

Evaluations run in the background. A valid request returns `202 Accepted` immediately with the `PENDING` run and a `Location` header pointing to it:

```http
HTTP/1.1 202 Accepted
Location: /evaluations/3f6c1b8e-...
```

```json
{
  "id": "3f6c1b8e-...",
  "status": "pending",
  "results": [],
  "completed_at": null,
  "error": null
}
```

Poll `GET /evaluations/{evaluation_id}` until `status` is `completed` or `failed`.

Invalid requests, such as unknown metrics, return `400 Bad Request` and no run is created.

---

## Get Evaluation

```http
GET /evaluations/{evaluation_id}
```

Returns the stored evaluation run and its results.

---

## List Evaluations

```http
GET /evaluations
```

Supported filters:

```text
status
model_name
limit
offset
```

Example:

```http
GET /evaluations?status=completed&model_name=llama3&limit=10&offset=0
```

`limit` must be between 1 and 1000 (default: 100) and `offset` must be 0 or greater. Invalid values return `422 Unprocessable Entity`. Results are returned in creation order.

---

## Get Evaluation Summary

```http
GET /evaluations/{evaluation_id}/summary
```

The summary includes:

* Total items
* Passed items
* Failed items
* Average score
* Success rate
* Per-metric average score
* Per-metric pass rate

---

## Delete Evaluation

```http
DELETE /evaluations/{evaluation_id}
```

---

# Example Evaluation

A simple evaluation can use:

```text
Dataset: math-test
Model: llama3
Metrics:
  - exact_match
  - similarity
  - llm_judge
```

The resulting evaluation contains:

```text
EvaluationRun
├── status
├── model_name
├── dataset
├── results
│   ├── exact_match
│   ├── similarity
│   └── llm_judge
└── summary
```

---

# Requirements

* Docker
* Docker Compose

The application uses Ollama as the model provider.

---

# Run

Build and start the services:

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI interactive documentation:

```text
http://localhost:8000/docs
```

---

# Development

Development tools are kept out of the production image. Install them in a local virtual environment with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

Run the same checks as CI:

```bash
ruff check .
ruff format --check .
mypy
python -m pytest -m "not integration"
```

Ruff and mypy are configured in `pyproject.toml`. Use `ruff check --fix .` and `ruff format .` to apply fixes.

---

# Testing

Tests that require a running Ollama instance are marked as `integration`. Unit tests run without Ollama:

```bash
python -m pytest -m "not integration"
```

To run the integration tests, start the services with `docker compose up`, pull the model with `docker compose exec ollama ollama pull llama3`, set `OLLAMA_BASE_URL=http://localhost:11434`, and run:

```bash
python -m pytest -m integration
```

GitHub Actions runs linting, formatting checks, type checking, the unit tests, and a Docker build on every pull request and on pushes to `main`.

The test suite covers:

* Data models
* Evaluation engine
* Evaluation runner
* Evaluators
* Model clients
* Repository operations
* Evaluation summaries
* API endpoints
* Error handling
* Filtering
* Pagination

---

# Project Structure

```text
llm-evaluation-platform/
│
├── app/
│   ├── evaluators/
│   │   ├── base.py
│   │   ├── exact_match.py
│   │   ├── llm_judge.py
│   │   ├── registry.py
│   │   └── similarity.py
│   │
│   ├── services/
│   │   ├── evaluation_engine.py
│   │   ├── evaluation_runner.py
│   │   ├── evaluation_summary_service.py
│   │   ├── model_client.py
│   │   └── ollama_model_client.py
│   │
│   ├── storage/
│   │   ├── base.py
│   │   ├── evaluation_repository.py
│   │   └── sqlite_evaluation_repository.py
│   │
│   ├── api.py
│   ├── config.py
│   └── models.py
│
├── tests/
│   ├── test_api.py
│   ├── test_config.py
│   ├── test_engine.py
│   ├── test_evaluation_runner.py
│   ├── test_exact_match.py
│   ├── test_llm_judge.py
│   ├── test_model_client.py
│   ├── test_ollama_model_client.py
│   ├── test_registry.py
│   ├── test_runner.py
│   └── ...
│
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

# Storage

Evaluation runs are stored in SQLite, so evaluation history survives application restarts.

The database location is configured with the `DATABASE_PATH` environment variable (default: `data/evaluations.db`). Docker Compose stores it in the `evaluation_data` volume at `/app/data/evaluations.db`.

Setting `DATABASE_PATH` to an empty string keeps evaluations in memory only, which is what the test suite uses.

Each run stores everything needed to interpret it later:

* The full dataset
* The selected metrics
* The judge model name, when `llm_judge` is used
* Per-item results, status, timestamps, and any error

When the application starts, runs left `PENDING` or `RUNNING` by a previous process are marked `FAILED` with the error `Interrupted by application restart`. They are not re-executed automatically.

---

# Current Scope

The current implementation focuses on the core evaluation workflow:

```text
Dataset
   ↓
LLM Generation
   ↓
Metric Evaluation
   ↓
Evaluation Run
   ↓
Evaluation Summary
```

Background evaluations run inside the API process. Evaluations that are pending or running when the application restarts are marked as failed and must be submitted again.

The project currently does not include:

* A durable job queue
* Web dashboard
* Dataset versioning
* Model benchmarking across large evaluation suites
* Advanced RAG-specific metrics

---

# Future Improvements

Possible next steps include:

* Embedding-based semantic similarity
* Weighted metrics
* Dataset versioning
* Model-to-model comparison
* Evaluation dashboards
* RAG-specific metrics
* Evaluation reports
* Batch evaluation jobs
* Additional LLM judge strategies

---

# License

This project is licensed under the MIT License. See the `LICENSE` file for details.
