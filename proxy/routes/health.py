import vllm_client
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """
    Liveness check — pings vLLM /health internally.

    Returns
    -------
    dict[str, str]
        Status dict with ``"ok"`` or ``"degraded"``.
    """

    ok = await vllm_client.health()
    return {"status": "ok" if ok else "degraded"}
