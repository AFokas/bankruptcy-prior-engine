"""Thin synchronous client for a local Ollama server.

Only the API surface needed by Stage 4 is exposed: list installed models
and run a single chat/completion query. All requests are blocking; the
Bradley-Terry experiment loops over models and prompts sequentially so the
per-call latency is the only thing that matters.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class OllamaConnectionError(RuntimeError):
    """Raised when the Ollama server is unreachable."""


class OllamaModelNotFoundError(RuntimeError):
    """Raised when the requested model is not present locally."""


class OllamaResponseError(RuntimeError):
    """Raised on a non-200 response that is not a model-not-found case."""


def list_available_models(base_url: str, timeout: float = 5.0) -> list[str]:
    """Return the names of all models installed on the local Ollama server.

    Args:
        base_url: Ollama API base URL, e.g. ``http://localhost:11434``.
        timeout: HTTP timeout in seconds.

    Returns:
        List of model names (e.g. ``['llama3.2:latest', 'qwen2.5:0.5b']``).

    Raises:
        OllamaConnectionError: If the server is unreachable.
    """
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    except httpx.RequestError as exc:
        raise OllamaConnectionError(f"Cannot reach Ollama at {base_url}: {exc}") from exc
    if resp.status_code != 200:
        raise OllamaResponseError(f"Ollama /api/tags returned {resp.status_code}: {resp.text[:200]}")
    payload = resp.json()
    return [m["name"] for m in payload.get("models", [])]


def query_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    base_url: str,
    timeout: float = 120.0,
    options: Optional[dict] = None,
) -> str:
    """Query a local Ollama model with one system + one user message.

    Args:
        model: Model name as listed by :func:`list_available_models`.
        system_prompt: System message setting the model's role and rules.
        user_prompt: User message body.
        temperature: Sampling temperature (0 = deterministic).
        base_url: Ollama API base URL.
        timeout: Per-request timeout in seconds.
        options: Optional extra ``options`` dict passed to ``/api/chat``
            (e.g. ``{"seed": 42, "num_predict": 64}``).

    Returns:
        Raw text content of the assistant message.

    Raises:
        OllamaConnectionError: Server unreachable.
        OllamaModelNotFoundError: Model not installed locally.
        OllamaResponseError: Any other non-200 response.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": float(temperature), **(options or {})},
    }

    try:
        resp = httpx.post(f"{base_url.rstrip('/')}/api/chat", json=body, timeout=timeout)
    except httpx.RequestError as exc:
        raise OllamaConnectionError(f"Request to {base_url} failed: {exc}") from exc

    if resp.status_code == 404 or (resp.status_code == 400 and "not found" in resp.text.lower()):
        raise OllamaModelNotFoundError(f"Model '{model}' is not installed locally")
    if resp.status_code != 200:
        raise OllamaResponseError(f"Ollama /api/chat returned {resp.status_code}: {resp.text[:300]}")

    payload = resp.json()
    message = payload.get("message", {})
    content = message.get("content", "")
    return content


def infer_model_family(model_name: str) -> str:
    """Return a coarse family tag from a model name (best-effort, lower-cased).

    Used for grouping in the LLN convergence plot. Conservative: when the
    family is ambiguous, returns ``'other'``.
    """
    name = model_name.lower()
    for prefix, family in (
        ("llama", "llama"),
        ("qwen", "qwen"),
        ("phi", "phi"),
        ("gemma", "gemma"),
        ("mistral", "mistral"),
        ("mixtral", "mistral"),
        ("tinyllama", "llama"),
        ("deepseek", "deepseek"),
        ("yi", "yi"),
    ):
        if name.startswith(prefix):
            return family
    return "other"
