from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes
    ----------
    vllm_base_url : str
        Base URL for the vLLM inference server.
    mlflow_tracking_uri : str
        MLflow tracking server URI. No default — fails loudly on misconfiguration.
    adapters_dir : str
        Adapter directory as seen by this (proxy) container.
    vllm_adapters_dir : str
        Adapter directory as seen by the vLLM container.
    default_max_tokens : int
        Default max tokens when not specified by the request.
    base_model_name : str
        Full HuggingFace model name for the base model served by vLLM.
    thinking_open_tag : str
        Opening tag for chain-of-thought reasoning blocks.
    thinking_close_tag : str
        Closing tag for chain-of-thought reasoning blocks.
    mlflow_experiment_name : str
        MLflow experiment name for proxy request traces.
    """

    vllm_base_url: str = "http://vllm:8000"
    mlflow_tracking_uri: str
    adapters_dir: str = "/app/adapters"  # proxy container mount point
    vllm_adapters_dir: str = "/adapters"  # vLLM container mount point
    default_max_tokens: int = 2048
    base_model_name: str = "Qwen/Qwen3.5-0.8B"
    thinking_open_tag: str = "<think>"
    thinking_close_tag: str = "</think>"
    mlflow_experiment_name: str = "proxy-traces"

    model_config = {"env_file": ".env"}


settings = Settings()
