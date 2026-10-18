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
