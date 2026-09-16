from app.evaluators.exact_match import ExactMatchEvaluator
from app.evaluators.registry import EvaluatorRegistry


def test_registry_registers_evaluator() -> None:
    registry = EvaluatorRegistry()
    evaluator = ExactMatchEvaluator()

    registry.register(
        "exact_match",
        evaluator,
    )

    result = registry.get("exact_match")

    assert result is evaluator


def test_registry_rejects_duplicate_registration() -> None:
    registry = EvaluatorRegistry()

    registry.register(
        "exact_match",
        ExactMatchEvaluator(),
    )

    try:
        registry.register(
            "exact_match",
            ExactMatchEvaluator(),
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_registry_returns_registered_evaluator() -> None:
    registry = EvaluatorRegistry()
    evaluator = ExactMatchEvaluator()

    registry.register(
        "exact_match",
        evaluator,
    )

    assert registry.get("exact_match") is evaluator


def test_registry_rejects_unknown_evaluator() -> None:
    registry = EvaluatorRegistry()

    try:
        registry.get("unknown_metric")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_registry_lists_registered_metrics() -> None:
    registry = EvaluatorRegistry()

    registry.register(
        "exact_match",
        ExactMatchEvaluator(),
    )

    assert registry.available_metrics() == [
        "exact_match"
    ]