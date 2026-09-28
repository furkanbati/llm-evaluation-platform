from app.evaluators.similarity import SimilarityEvaluator


def test_similarity_returns_perfect_score_for_identical_text() -> None:
    evaluator = SimilarityEvaluator()

    result = evaluator.evaluate(
        expected_output="Paris",
        generated_output="Paris",
    )

    assert result.metric_name == "similarity"
    assert result.score == 1.0
    assert result.passed is True


def test_similarity_returns_lower_score_for_different_text() -> None:
    evaluator = SimilarityEvaluator()

    result = evaluator.evaluate(
        expected_output="Paris",
        generated_output="London",
    )

    assert result.metric_name == "similarity"
    assert result.score < 1.0
    assert result.passed is False


def test_similarity_ignores_surrounding_whitespace() -> None:
    evaluator = SimilarityEvaluator()

    result = evaluator.evaluate(
        expected_output="  Paris  ",
        generated_output="Paris",
    )

    assert result.score == 1.0
    assert result.passed is True