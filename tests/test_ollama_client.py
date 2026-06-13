"""Unit tests for the Ollama client (mocked HTTP, no live calls)."""

from __future__ import annotations

import httpx
import pytest

from src.ollama_client import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    infer_model_family,
    list_available_models,
    query_ollama,
)


# ---------------------------------------------------------------------------
# list_available_models
# ---------------------------------------------------------------------------


def test_list_available_models_returns_names(monkeypatch):
    def fake_get(url, timeout):
        assert url.endswith("/api/tags")
        request = httpx.Request("GET", url)
        return httpx.Response(200, json={"models": [
            {"name": "llama3.2:latest", "size": 1},
            {"name": "qwen2.5:0.5b", "size": 2},
        ]}, request=request)
    monkeypatch.setattr(httpx, "get", fake_get)
    names = list_available_models("http://localhost:11434")
    assert names == ["llama3.2:latest", "qwen2.5:0.5b"]


def test_list_available_models_raises_on_connection_failure(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError("nope", request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(OllamaConnectionError):
        list_available_models("http://localhost:11434")


# ---------------------------------------------------------------------------
# query_ollama
# ---------------------------------------------------------------------------


def test_query_ollama_returns_content(monkeypatch):
    def fake_post(url, json, timeout):
        assert url.endswith("/api/chat")
        assert json["messages"][0]["role"] == "system"
        assert json["options"]["temperature"] == 0.0
        return httpx.Response(200, json={"message": {"content": "B"}}, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", fake_post)
    out = query_ollama("llama3.2", "sys", "user", temperature=0.0, base_url="http://localhost:11434")
    assert out == "B"


def test_query_ollama_model_not_found_raises(monkeypatch):
    def fake_post(url, json, timeout):
        return httpx.Response(404, json={"error": "model not found"}, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(OllamaModelNotFoundError):
        query_ollama("missing", "s", "u", 0.0, "http://localhost:11434")


def test_query_ollama_other_error_raises(monkeypatch):
    def fake_post(url, json, timeout):
        return httpx.Response(500, text="server exploded", request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(OllamaResponseError):
        query_ollama("m", "s", "u", 0.0, "http://localhost:11434")


# ---------------------------------------------------------------------------
# infer_model_family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, family", [
    ("llama3.2:latest", "llama"),
    ("qwen2.5:0.5b", "qwen"),
    ("phi3:mini", "phi"),
    ("tinyllama:latest", "llama"),
    ("gemma2:2b", "gemma"),
    ("mistral:7b", "mistral"),
    ("unknown-model:latest", "other"),
])
def test_infer_model_family(name, family):
    assert infer_model_family(name) == family
