"""
Register and promote a LoRA adapter from an existing MLflow training run.

Usage
-----
Promote the adapter from a specific training run:

    uv run scripts/promote_adapter.py --run-id <mlflow-run-id>

Use a custom adapter name:

    uv run scripts/promote_adapter.py --run-id <mlflow-run-id> --adapter-name my-adapter

Required environment variable:

    MLFLOW_TRACKING_URI   URI of the MLflow tracking server (e.g. http://localhost:5000)
"""

import argparse
import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for adapter promotion.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with run ID and adapter name.
    """

    parser = argparse.ArgumentParser(
        description="Register a LoRA adapter and set the 'production' alias in MLflow"
    )
    parser.add_argument(
        "--run-id", type=str, required=True, help="MLflow training run ID with adapter artifacts"
    )
    parser.add_argument(
        "--adapter-name",
        type=str,
        default="lora-adapter",
        help="Registered model name in the MLflow registry (default: lora-adapter)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Register the adapter from a training run and set the 'production' alias.
    """

    args = parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    client = MlflowClient()
    try:
        client.create_registered_model(args.adapter_name)
    except MlflowException:
        pass  # model already registered

    source_uri = f"runs:/{args.run_id}/adapter"
    model_version = client.create_model_version(
        name=args.adapter_name,
        source=source_uri,
        run_id=args.run_id,
    )
    client.set_registered_model_alias(
        name=args.adapter_name,
        alias="production",
        version=model_version.version,
    )

    print(
        f"Registered {args.adapter_name} v{model_version.version} "
        f"from run {args.run_id} → production"
    )


if __name__ == "__main__":
    main()
