from typing import Any

import mlflow
import processors
import registry
import vllm_client
from config import settings
from fastapi import APIRouter
from mlflow.entities import SpanType
from schemas import CompleteRequest
from thinking import extract_thinking

router = APIRouter()


@router.post("/complete", response_model=None)
async def complete(request: CompleteRequest) -> dict[str, Any]:
    """
    Main inference route.

    Resolves the adapter, loads it into vLLM if needed, runs adapter-specific
    pre-processing, forwards to vLLM, extracts thinking blocks into
    ``reasoning_content``, runs adapter-specific post-processing, and records
    the full request lifecycle as an MLflow trace.

    Parameters
    ----------
    request : CompleteRequest
        Request body with adapter name, messages, context, and sampling params.

    Returns
    -------
    dict[str, Any]
        Parsed JSON response with ``reasoning_content`` added if present.
    """

    tags = {"model": settings.base_model_name, "adapter": request.adapter}

    # Resolve adapter from MLflow; download if new version promoted
    with mlflow.start_span(name="resolve_adapter") as span:
        mlflow.update_current_trace(tags=tags)
        _, vllm_path, was_updated = registry.resolve_adapter_path(request.adapter)
        if was_updated:
            vllm_client.evict_adapter(request.adapter)

        await vllm_client.ensure_adapter_loaded(request.adapter, vllm_path)
        span.set_attributes({"adapter": request.adapter, "was_updated": was_updated})

    processor = processors.get_processor(request.adapter)
    max_tokens = request.sampling.max_tokens or settings.default_max_tokens

    # Pre-process messages (e.g. inject schema for SQL adapters)
    with mlflow.start_span(name="preprocess") as span:
        messages = processor.preprocess(request.messages, request.context)
        span.set_inputs({"context": request.context})

    # Forward to vLLM
    with mlflow.start_span(name="vllm_inference", span_type=SpanType.LLM) as span:
        span.set_inputs({"messages": messages, "temperature": request.sampling.temperature})
        response = await vllm_client.chat_completions(
            model=request.adapter,
            messages=messages,
            temperature=request.sampling.temperature,
            max_tokens=max_tokens,
        )
        span.set_outputs(response)

    # Post-process: extract thinking into reasoning_content, run adapter-specific validation
    with mlflow.start_span(name="post_process") as span:
        raw_content = response["choices"][0]["message"]["content"]
        clean_content, reasoning = extract_thinking(raw_content)

        clean_content, span_attrs = processor.postprocess(clean_content)

        response["choices"][0]["message"]["content"] = clean_content
        if reasoning is not None:
            response["choices"][0]["message"]["reasoning_content"] = reasoning

        span.set_attributes(span_attrs)

    return response
