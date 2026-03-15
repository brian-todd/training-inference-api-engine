import os

# Must be set before config.py is imported, as Settings() is called at module level.
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:///tmp/mlflow_proxy_test")

from collections.abc import Iterator  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import pytest  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402


@pytest.fixture()
def mock_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch all vllm_client functions to avoid needing a live vLLM instance.
    """
    import vllm_client

    monkeypatch.setattr(vllm_client, "startup", AsyncMock())
    monkeypatch.setattr(vllm_client, "shutdown", AsyncMock())
    monkeypatch.setattr(vllm_client, "health", AsyncMock(return_value=True))
    monkeypatch.setattr(
        vllm_client, "list_models", AsyncMock(return_value={"object": "list", "data": []})
    )
    monkeypatch.setattr(vllm_client, "ensure_adapter_loaded", AsyncMock())
    monkeypatch.setattr(vllm_client, "evict_adapter", MagicMock())
    monkeypatch.setattr(
        vllm_client,
        "chat_completions",
        AsyncMock(
            return_value={"choices": [{"message": {"content": "SELECT 1", "role": "assistant"}}]}
        ),
    )


@pytest.fixture()
def client(mock_vllm: None) -> Iterator[TestClient]:
    """
    Starlette TestClient wrapping the FastAPI app with vllm mocked out.
    """
    from app import app

    with TestClient(app) as test_client:
        yield test_client
