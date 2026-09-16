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