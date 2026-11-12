"""Тесты не должны зависеть от окружения: в Colab ноутбук выставляет LLM_BACKEND=ollama до запуска pytest."""
import pytest

ENV_VARS = ["LLM_BACKEND", "OLLAMA_MODEL", "OLLAMA_URL", "OLLAMA_NUM_CTX", "OLLAMA_TIMEOUT",
            "LOCAL_MODEL_NAME", "ALLOW_NO_STEMMER"]


@pytest.fixture(autouse=True)
def clean_llm_environment(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
