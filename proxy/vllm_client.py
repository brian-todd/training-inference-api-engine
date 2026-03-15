import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
from config import settings

_client: httpx.AsyncClient | None = None
_loaded_adapters: set[str] = set()
_load_lock: asyncio.Lock = asyncio.Lock()


async def startup() -> None:
    """
    Initialise the shared HTTP client. Call once from app lifespan.
    """

    global _client  # noqa: PLW0603
    _client = httpx.AsyncClient(base_url=settings.vllm_base_url, timeout=120.0)


async def shutdown() -> None:
    """
    Close the shared HTTP client. Call once from app lifespan.
    """

    if _client is not None:
        await _client.aclose()


def _get_client() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("vLLM HTTP client not initialised — call startup() first")
    return _client


def evict_adapter(adapter_name: str) -> None:
    """
    Remove adapter from loaded cache so the next request triggers a reload.

    Parameters
    ----------
    adapter_name : str
        Name of the adapter to evict from the in-memory loaded set.
    """

    _loaded_adapters.discard(adapter_name)


async def health() -> bool:
    """
    Return True if vLLM health endpoint responds 200.

    Returns
    -------
    bool
        ``True`` if vLLM is healthy, ``False`` otherwise.
    """

    try:
        resp = await _get_client().get("/health")
        healthy = resp.status_code == 200
    except httpx.HTTPError:
        healthy = False

    if not healthy:
        _loaded_adapters.clear()

    return healthy


async def list_models() -> dict[str, Any]:
    """
    Return the raw JSON from vLLM's GET /v1/models.

    Returns
    -------
    dict[str, Any]
        Parsed JSON response from vLLM.
    """

    resp = await _get_client().get("/v1/models")
    resp.raise_for_status()
    return resp.json()


async def ensure_adapter_loaded(adapter_name: str, vllm_adapter_path: str) -> None:
    """
    Load a LoRA adapter into vLLM if not already loaded.

    Parameters
    ----------
    adapter_name : str
        Name to register the adapter under in vLLM (used as ``model`` in completions).
    vllm_adapter_path : str
        Filesystem path as seen by the vLLM container.
    """

    if adapter_name in _loaded_adapters:
        return

    async with _load_lock:
        if adapter_name in _loaded_adapters:  # re-check after acquiring lock
            return

        resp = await _get_client().post(
            "/v1/load_lora_adapter",
            json={"lora_name": adapter_name, "lora_path": vllm_adapter_path},
        )
        resp.raise_for_status()
        _loaded_adapters.add(adapter_name)


async def stream_chat_completions(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> AsyncIterator[str]:
    """
    Stream a chat completion request from vLLM as SSE lines.

    Parameters
    ----------
    model : str
        Model or adapter name to use for generation.
    messages : list[dict[str, str]]
        Chat history in OpenAI message format.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate.

    Yields
    ------
    str
        Individual SSE lines from the vLLM streaming response.
    """

    client = _get_client()
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if line:
                yield line


async def chat_completions(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """
    Forward a chat completion request to vLLM and return the parsed response.

    Parameters
    ----------
    model : str
        Adapter name (as registered via ``ensure_adapter_loaded``).
    messages : list[dict[str, str]]
        Chat history in OpenAI message format.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate.

    Returns
    -------
    dict[str, Any]
        Parsed JSON response from vLLM's /v1/chat/completions endpoint.
    """

    resp = await _get_client().post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    resp.raise_for_status()
    return resp.json()
