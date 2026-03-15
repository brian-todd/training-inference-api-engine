import time
from pathlib import Path
from typing import Any

import registry
import vllm_client
from config import settings
from fastapi import APIRouter
from schemas import LoadAdapterRequest

router = APIRouter()


@router.get("/adapters")
async def list_adapters() -> dict[str, Any]:
    """
    List models/adapters currently loaded in vLLM.

    Returns
    -------
    dict[str, Any]
        Raw JSON response from vLLM's model listing endpoint.
    """

    return await vllm_client.list_models()


@router.post("/load_adapter")
async def load_adapter(request: LoadAdapterRequest) -> dict[str, str]:
    """
    Pre-warm an adapter: resolve from MLflow registry and load into vLLM.

    Parameters
    ----------
    request : LoadAdapterRequest
        Request body containing the adapter name.

    Returns
    -------
    dict[str, str]
        Confirmation with adapter name and ``"loaded"`` status.
    """

    _, vllm_path, was_updated = registry.resolve_adapter_path(request.adapter)
    if was_updated:
        vllm_client.evict_adapter(request.adapter)

    await vllm_client.ensure_adapter_loaded(request.adapter, vllm_path)

    return {"adapter": request.adapter, "status": "loaded"}


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """
    OpenAI-compatible model listing.

    Merges vLLM's currently-loaded models with all adapter directories found
    on disk, so adapters appear in Open WebUI before they have been warmed up.

    Returns
    -------
    dict[str, Any]
        OpenAI-compatible ``{"object": "list", "data": [...]}`` response.
    """

    response = await vllm_client.list_models()
    known_ids = {entry["id"] for entry in response.get("data", [])}

    adapters_dir = Path(settings.adapters_dir)
    if adapters_dir.is_dir():
        now = int(time.time())
        for adapter_dir in sorted(adapters_dir.iterdir()):
            if adapter_dir.is_dir() and adapter_dir.name not in known_ids:
                response["data"].append(
                    {
                        "id": adapter_dir.name,
                        "object": "model",
                        "created": now,
                        "owned_by": "local",
                    }
                )

    return response
