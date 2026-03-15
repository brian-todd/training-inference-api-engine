import json
from collections.abc import AsyncIterator
from pathlib import Path

import processors
import vllm_client
from config import settings
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from schemas import ChatCompletionRequest
from thinking import ThinkingExtractor, extract_thinking

from tasks.base import Processor

router = APIRouter()


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
) -> StreamingResponse | dict[str, object]:
    """
    OpenAI-compatible chat completions for general-purpose chat.

    If the requested model name matches a local adapter directory, the adapter
    is auto-loaded into vLLM. Supports both streaming and non-streaming responses.
    Adapter-specific pre/post-processing is applied if a processor is registered.

    Parameters
    ----------
    request : ChatCompletionRequest
        OpenAI-compatible chat request body.

    Returns
    -------
    StreamingResponse | dict[str, object]
        SSE stream for streaming requests, or parsed JSON for non-streaming.
    """

    model_name = request.model
    max_tokens = request.max_tokens or settings.default_max_tokens

    # Auto-load adapter if it exists on disk (no MLflow coupling)
    adapter_path = Path(settings.adapters_dir) / model_name
    if adapter_path.exists():
        vllm_path = f"{settings.vllm_adapters_dir}/{model_name}"
        await vllm_client.ensure_adapter_loaded(model_name, vllm_path)

    processor = processors.get_processor(model_name)
    messages = processor.preprocess(request.messages, {})

    if request.stream:
        return StreamingResponse(
            _stream_chat(model_name, messages, request.temperature, max_tokens, processor),
            media_type="text/event-stream",
        )

    response = await vllm_client.chat_completions(
        model=model_name,
        messages=messages,
        temperature=request.temperature,
        max_tokens=max_tokens,
    )

    raw_content = response["choices"][0]["message"]["content"]
    clean_content, reasoning = extract_thinking(raw_content)
    clean_content, _ = processor.postprocess(clean_content)

    response["choices"][0]["message"]["content"] = clean_content
    if reasoning is not None:
        response["choices"][0]["message"]["reasoning_content"] = reasoning

    return response


async def _stream_chat(
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    processor: Processor,
) -> AsyncIterator[str]:
    """
    Yield SSE events for a streaming chat completion, separating thinking tokens.

    Thinking blocks are extracted and emitted as ``delta.reasoning_content`` SSE
    events before the corresponding ``delta.content`` events.

    Parameters
    ----------
    model_name : str
        Model or adapter name.
    messages : list[dict[str, str]]
        Chat messages.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate.
    processor : Processor
        Adapter-specific processor for post-stream validation.

    Yields
    ------
    str
        SSE-formatted lines ready for the client.
    """

    extractor = ThinkingExtractor()
    full_content = ""

    async for line in vllm_client.stream_chat_completions(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        if not line.startswith("data: "):
            continue

        payload = line[len("data: ") :]
        if payload.strip() == "[DONE]":
            break

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        delta = chunk.get("choices", [{}])[0].get("delta", {})
        raw_content = delta.get("content", "")
        if raw_content:
            clean_text, reasoning_text = extractor.process(raw_content)
            if reasoning_text:
                reasoning_chunk = {
                    **chunk,
                    "choices": [
                        {
                            **chunk["choices"][0],
                            "delta": {"reasoning_content": reasoning_text},
                        }
                    ],
                }
                yield f"data: {json.dumps(reasoning_chunk)}\n\n"
            if clean_text:
                full_content += clean_text
                delta["content"] = clean_text
                yield f"data: {json.dumps(chunk)}\n\n"
        else:
            # Pass through non-content deltas (role, finish_reason, etc.)
            yield f"{line}\n\n"

    processor.postprocess_stream(full_content)

    yield "data: [DONE]\n\n"
