from typing import Any

from pydantic import BaseModel, Field


class SamplingParams(BaseModel):
    """
    Sampling parameters for inference.
    """

    temperature: float = 0.7
    max_tokens: int | None = None


class CompleteRequest(BaseModel):
    """
    Request body for the /v1/complete endpoint.
    """

    adapter: str
    messages: list[dict[str, str]]
    context: dict[str, Any] = Field(default_factory=dict)
    sampling: SamplingParams = Field(default_factory=SamplingParams)


class LoadAdapterRequest(BaseModel):
    """
    Request body for the /v1/load_adapter endpoint.
    """

    adapter: str


class ChatCompletionRequest(BaseModel):
    """
    Request body for the OpenAI-compatible /v1/chat/completions endpoint.
    """

    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False

    model_config = {"extra": "ignore"}
