# LLM Evaluation Platform

A production-oriented REST API for evaluating Large Language Model outputs against predefined datasets.

## Features

* Exact Match evaluation
* Evaluation run lifecycle tracking
* Evaluation result storage
* Evaluation summary and metric statistics
* Evaluation filtering by status and model
* Pagination for evaluation listing
* REST API
* Ollama model integration
* Docker support
* Automated tests

## Architecture

The evaluation flow is:

```text
Client
  |
  v
FastAPI
  |
  v
EvaluationRunner
  |
  +----> OllamaModelClient
  |             |
  |             v
  |           Ollama
  |
  v
EvaluationEngine
  |
  v
EvaluatorRegistry
  |
  v
EvaluationRun
  |
  v
EvaluationRepository
```

## Requirements

* Docker
* Docker Compose

The application uses Ollama as the model provider.

## Run

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

## API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ready"
}
```

### Create Evaluation

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
    "exact_match"
  ]
}
```

### Get Evaluation

```http
GET /evaluations/{evaluation_id}
```

### List Evaluations

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

### Get Evaluation Summary

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

### Delete Evaluation

```http
DELETE /evaluations/{evaluation_id}
```

## Testing

Run the complete test suite inside the API container:

```bash
docker compose exec evaluation-api python -m pytest
```

The project includes unit and API tests covering:

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

## Project Structure

```text
app/
├── evaluators/
├── services/
├── storage/
├── api.py
├── config.py
└── models.py

tests/
```

## Current Evaluation Metric

### Exact Match

The current evaluation engine supports Exact Match evaluation.

The generated output is compared with the expected output and produces a score between `0.0` and `1.0`.

## License

This project is intended as a portfolio project demonstrating production-oriented LLM application development.
