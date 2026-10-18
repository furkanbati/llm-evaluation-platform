import importlib

import app.config as config


def test_ollama_base_url_uses_default_when_environment_variable_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    importlib.reload(config)

    assert config.OLLAMA_BASE_URL == "http://ollama:11434"


def test_ollama_base_url_uses_environment_variable(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    importlib.reload(config)

    assert config.OLLAMA_BASE_URL == "http://localhost:11434"


def test_judge_model_name_uses_default_when_environment_variable_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "JUDGE_MODEL_NAME",
        raising=False,
    )

    importlib.reload(config)

    assert config.JUDGE_MODEL_NAME == "llama3"


def test_judge_model_name_uses_environment_variable(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JUDGE_MODEL_NAME",
        "judge-model",
    )

    importlib.reload(config)

    assert config.JUDGE_MODEL_NAME == "judge-model"


def test_judge_temperature_defaults_to_zero(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "JUDGE_TEMPERATURE",
        raising=False,
    )

    importlib.reload(config)

    assert config.JUDGE_TEMPERATURE == 0.0


def test_judge_temperature_uses_environment_variable(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "JUDGE_TEMPERATURE",
        "0.2",
    )

    importlib.reload(config)

    assert config.JUDGE_TEMPERATURE == 0.2


def test_database_path_uses_default_when_environment_variable_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "DATABASE_PATH",
        raising=False,
    )

    importlib.reload(config)

    assert config.DATABASE_PATH == "data/evaluations.db"


def test_database_path_uses_environment_variable(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "DATABASE_PATH",
        "/tmp/evaluations.db",
    )

    importlib.reload(config)

    assert config.DATABASE_PATH == "/tmp/evaluations.db"


def test_generation_temperature_defaults_to_zero(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "GENERATION_TEMPERATURE",
        raising=False,
    )

    importlib.reload(config)

    assert config.GENERATION_TEMPERATURE == 0.0


def test_generation_temperature_can_be_left_to_provider(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "GENERATION_TEMPERATURE",
        "",
    )

    importlib.reload(config)

    assert config.GENERATION_TEMPERATURE is None


def test_generation_seed_is_unset_by_default(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "GENERATION_SEED",
        raising=False,
    )

    importlib.reload(config)

    assert config.GENERATION_SEED is None


def test_generation_seed_uses_environment_variable(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "GENERATION_SEED",
        "42",
    )

    importlib.reload(config)

    assert config.GENERATION_SEED == 42


def test_evaluation_limits_use_defaults(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MAX_CONCURRENT_EVALUATIONS", raising=False)
    monkeypatch.delenv("MAX_QUEUED_EVALUATIONS", raising=False)

    importlib.reload(config)

    assert config.MAX_CONCURRENT_EVALUATIONS == 2
    assert config.MAX_QUEUED_EVALUATIONS == 100


def test_evaluation_limits_use_environment_variables(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MAX_CONCURRENT_EVALUATIONS", "4")
    monkeypatch.setenv("MAX_QUEUED_EVALUATIONS", "20")

    importlib.reload(config)

    assert config.MAX_CONCURRENT_EVALUATIONS == 4
    assert config.MAX_QUEUED_EVALUATIONS == 20
