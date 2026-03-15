from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import mlflow
import vllm_client
from config import settings
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from routes.adapters import router as adapters_router
from routes.chat import router as chat_router
from routes.complete import router as complete_router
from routes.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage vLLM client lifecycle and configure MLflow tracing.
    """

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    await vllm_client.startup()
    yield
    await vllm_client.shutdown()


app = FastAPI(title="LLM Proxy", lifespan=lifespan)


@app.exception_handler(httpx.HTTPStatusError)
async def vllm_error_handler(request: Request, exc: httpx.HTTPStatusError) -> JSONResponse:
    """
    Translate upstream vLLM HTTP errors into structured OpenAI-style JSON responses.
    """

    try:
        detail = exc.response.json()
    except Exception:
        detail = exc.response.text

    return JSONResponse(
        status_code=exc.response.status_code,
        content={
            "error": {
                "message": (
                    detail if isinstance(detail, str) else detail.get("message", str(detail))
                ),
                "type": "upstream_error",
                "code": exc.response.status_code,
            }
        },
    )


app.include_router(health_router)
app.include_router(adapters_router, prefix="/v1")
app.include_router(chat_router, prefix="/v1")
app.include_router(complete_router, prefix="/v1")
