from pathlib import Path

import mlflow
from config import settings
from mlflow import MlflowClient

_adapter_run_ids: dict[str, str] = {}


def resolve_adapter_path(adapter_name: str) -> tuple[str, str, bool]:
    """
    Resolve production adapter path from MLflow registry.

    Downloads adapter artifacts to ``settings.adapters_dir`` on first call or when
    the production version changes. Returns a flag indicating whether a (re-)download
    occurred so callers can evict the adapter from vLLM's loaded cache.

    Parameters
    ----------
    adapter_name : str
        Registered model name in the MLflow Model Registry.

    Returns
    -------
    tuple[str, str, bool]
        ``(proxy_local_path, vllm_local_path, was_updated)`` where
        ``proxy_local_path`` is the adapter directory as seen by this container,
        ``vllm_local_path`` is the path to pass to vLLM's load endpoint, and
        ``was_updated`` is ``True`` if artifacts were (re-)downloaded.

    Raises
    ------
    ValueError
        If no production alias exists for the adapter in the registry.
    """

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    try:
        model_version = client.get_model_version_by_alias(adapter_name, "production")
    except mlflow.exceptions.MlflowException as exc:
        raise ValueError(f"No 'production' alias found for adapter '{adapter_name}'") from exc

    run_id = model_version.run_id

    # Download_artifacts creates a subdirectory named after the artifact path ("adapter"),
    # so the actual files land at `{download_root}/adapter/`.
    download_root = f"{settings.adapters_dir}/{adapter_name}"
    proxy_path = f"{download_root}/adapter"
    vllm_path = f"{settings.vllm_adapters_dir}/{adapter_name}/adapter"

    cached_run_id = _adapter_run_ids.get(adapter_name)
    needs_download = cached_run_id != run_id or not Path(proxy_path).exists()

    if needs_download:
        artifact_uri = client.get_run(run_id).info.artifact_uri
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"{artifact_uri}/adapter",
            dst_path=download_root,
        )
        _adapter_run_ids[adapter_name] = run_id

    return proxy_path, vllm_path, needs_download
