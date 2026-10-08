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
* Evaluation result storage
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
2. `EvaluationRunner` starts an evaluation run.
3. The configured model client generates outputs for the dataset items.
4. `EvaluationEngine` executes the selected evaluators.
5. Each evaluator produces a `MetricResult`.
6. The evaluation run is completed and stored in the repository.
7. The API exposes the run, individual results, and summary statistics.

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

# Testing

Run the complete test suite inside the API container:

```bash
docker compose exec evaluation-api python -m pytest
```

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
│   │   └── evaluation_repository.py
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
├── requirements.txt
└── README.md
```

---

# Storage

The current repository implementation stores evaluation runs in memory.

This keeps the architecture intentionally simple for the current version, but evaluation history is not persistent across application restarts.

Persistent database storage is planned as a future improvement.

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

The project currently does not include:

* Persistent database storage
* Web dashboard
* Dataset versioning
* Model benchmarking across large evaluation suites
* Advanced RAG-specific metrics

---

# Future Improvements

Possible next steps include:

* Embedding-based semantic similarity
* Weighted metrics
* Persistent database storage
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
