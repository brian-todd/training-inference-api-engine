import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import vllm_client
from starlette.testclient import TestClient

# ── /health ───────────────────────────────────────────────────────────────────


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_degraded(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vllm_client, "health", AsyncMock(return_value=False))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded"}


# ── adapter cache staleness ──────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_health_failure_clears_adapter_cache() -> None:
    """When vLLM health check fails, loaded adapter cache should be cleared."""
    vllm_client._loaded_adapters.add("test-adapter")
    assert "test-adapter" in vllm_client._loaded_adapters

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch.object(vllm_client, "_client", mock_client):
        result = await vllm_client.health()

    assert result is False
    assert len(vllm_client._loaded_adapters) == 0


# ── /v1/adapters and /v1/models ───────────────────────────────────────────────


def test_list_adapters(client: TestClient) -> None:
    response = client.get("/v1/adapters")
    assert response.status_code == 200
    assert "data" in response.json()


def test_list_models(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert "data" in response.json()


# ── /v1/load_adapter ─────────────────────────────────────────────────────────


def test_load_adapter_calls_registry_and_vllm(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_resolve = MagicMock(return_value=("/local/path", "/vllm/path", False))
    mock_ensure = AsyncMock()
    monkeypatch.setattr("registry.resolve_adapter_path", mock_resolve)
    monkeypatch.setattr(vllm_client, "ensure_adapter_loaded", mock_ensure)

    response = client.post("/v1/load_adapter", json={"adapter": "my-adapter"})

    assert response.status_code == 200
    assert response.json() == {"adapter": "my-adapter", "status": "loaded"}
    mock_resolve.assert_called_once_with("my-adapter")
    mock_ensure.assert_awaited_once_with("my-adapter", "/vllm/path")


def test_load_adapter_evicts_when_updated(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_evict = MagicMock()
    monkeypatch.setattr("registry.resolve_adapter_path", MagicMock(return_value=("/p", "/v", True)))
    monkeypatch.setattr(vllm_client, "ensure_adapter_loaded", AsyncMock())
    monkeypatch.setattr(vllm_client, "evict_adapter", mock_evict)

    client.post("/v1/load_adapter", json={"adapter": "stale-adapter"})

    mock_evict.assert_called_once_with("stale-adapter")


# ── /v1/complete ─────────────────────────────────────────────────────────────


def test_complete_extracts_thinking(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "registry.resolve_adapter_path", MagicMock(return_value=("/p", "/v", False))
    )
    monkeypatch.setattr(
        vllm_client,
        "chat_completions",
        AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "<think>reasoning</think>SELECT 1",
                            "role": "assistant",
                        }
                    }
                ]
            }
        ),
    )

    response = client.post(
        "/v1/complete",
        json={
            "adapter": "sql-adapter",
            "messages": [{"role": "user", "content": "Get all rows"}],
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert "<think>" not in message["content"]
    assert "SELECT 1" in message["content"]
    assert message["reasoning_content"] == "reasoning"


def test_complete_runs_sqlglot_validation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validates that sqlglot is run — bad SQL still returns 200 (validation is non-blocking)."""
    monkeypatch.setattr(
        "registry.resolve_adapter_path", MagicMock(return_value=("/p", "/v", False))
    )
    monkeypatch.setattr(
        vllm_client,
        "chat_completions",
        AsyncMock(
            return_value={
                "choices": [{"message": {"content": "not valid sql !!!", "role": "assistant"}}]
            }
        ),
    )

    response = client.post(
        "/v1/complete",
        json={
            "adapter": "sql-adapter",
            "messages": [{"role": "user", "content": "Show orders"}],
        },
    )

    assert response.status_code == 200


# ── /v1/chat/completions ──────────────────────────────────────────────────────


def test_chat_completions_non_streaming(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vllm_client,
        "chat_completions",
        AsyncMock(
            return_value={"choices": [{"message": {"content": "Hello!", "role": "assistant"}}]}
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "some-model",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == "Hello!"
    assert "reasoning_content" not in message


def test_chat_completions_extracts_thinking(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        vllm_client,
        "chat_completions",
        AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "<think>internal</think>actual answer",
                            "role": "assistant",
                        }
                    }
                ]
            }
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "q"}]},
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["content"] == "actual answer"
    assert message["reasoning_content"] == "internal"


# ── /v1/chat/completions streaming ──────────────────────────────────────────


def _make_sse_chunk(content: str, model: str = "test-model") -> str:
    """
    Build an SSE data line from a content string.

    Parameters
    ----------
    content : str
        The content to place in ``delta.content``.
    model : str
        Model name for the chunk.

    Returns
    -------
    str
        An SSE-formatted ``data: {...}`` line.
    """
    chunk = {
        "choices": [{"delta": {"content": content}, "index": 0}],
        "model": model,
    }
    return f"data: {json.dumps(chunk)}"


async def _async_iter_lines(lines: list[str]) -> AsyncIterator[str]:
    """
    Yield lines as an async iterator, simulating vllm_client.stream_chat_completions.

    Parameters
    ----------
    lines : list[str]
        SSE lines to yield.

    Yields
    ------
    str
        Individual SSE lines.
    """
    for line in lines:
        yield line


def test_chat_completions_streaming_emits_reasoning_content(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sse_lines = [
        _make_sse_chunk("<think>reasoning</think>answer"),
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        vllm_client,
        "stream_chat_completions",
        lambda **kwargs: _async_iter_lines(sse_lines),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    events = [
        line
        for line in response.text.split("\n\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    # First event should have reasoning_content
    first = json.loads(events[0][len("data: ") :])
    assert first["choices"][0]["delta"]["reasoning_content"] == "reasoning"
    assert "content" not in first["choices"][0]["delta"]
    # Second event should have content
    second = json.loads(events[1][len("data: ") :])
    assert second["choices"][0]["delta"]["content"] == "answer"
    assert "reasoning_content" not in second["choices"][0]["delta"]


def test_chat_completions_streaming_no_thinking(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sse_lines = [
        _make_sse_chunk("plain answer"),
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        vllm_client,
        "stream_chat_completions",
        lambda **kwargs: _async_iter_lines(sse_lines),
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    events = [
        line
        for line in response.text.split("\n\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert len(events) == 1
    parsed = json.loads(events[0][len("data: ") :])
    assert parsed["choices"][0]["delta"]["content"] == "plain answer"
    assert "reasoning_content" not in parsed["choices"][0]["delta"]
