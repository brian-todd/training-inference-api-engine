"""
Evaluate a fine-tuned LoRA adapter using mlflow.evaluate().

Usage
-----
Basic evaluation (full validation set):

    uv run scripts/eval.py --task sql --adapter-path ./adapters/sql-lora

Evaluate on a random subset of 200 examples:

    uv run scripts/eval.py --task sql --adapter-path ./adapters/sql-lora --num-samples 200

Register and promote the adapter to Production in MLflow:

    uv run scripts/eval.py --task sql --adapter-path ./adapters/sql-lora --promote

Link to the training run for cross-reference:

    uv run scripts/eval.py --task sql --adapter-path ./adapters/sql-lora \\
        --training-run-id <mlflow-run-id>

Required environment variable:

    MLFLOW_TRACKING_URI   URI of the MLflow tracking server (e.g. http://localhost:5000)
"""

import argparse
import importlib
import os
from typing import Any

import mlflow
import torch
from datasets import load_from_disk
from mlflow.client import MlflowClient
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def register_and_promote(run_id: str, adapter_name: str) -> None:
    """
    Register a LoRA adapter in the MLflow Model Registry and promote to Production.

    Parameters
    ----------
    run_id : str
        MLflow run ID that logged the adapter artifacts under the ``adapter`` path.
    adapter_name : str
        Registered model name in the MLflow registry.
    """

    client = MlflowClient()
    try:
        client.create_registered_model(adapter_name)
    except mlflow.exceptions.MlflowException:
        pass  # model already registered

    source_uri = f"runs:/{run_id}/adapter"
    model_version = client.create_model_version(
        name=adapter_name,
        source=source_uri,
        run_id=run_id,
    )
    client.transition_model_version_stage(
        name=adapter_name,
        version=model_version.version,
        stage="Production",
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with evaluation configuration.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned adapter via mlflow.evaluate()"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sql",
        help="Task plugin name under tasks/ (default: sql)",
    )
    parser.add_argument(
        "--adapter-path", type=str, required=True, help="Path to saved adapter directory"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model ID (default: Qwen/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--db-dir", type=str, default=None, help="Task database directory (e.g. Spider)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Evaluate on a random subset of this size (default: full val set)",
    )
    parser.add_argument(
        "--adapter-name",
        type=str,
        default="sql-lora",
        help="Registry model name (default: sql-lora)",
    )
    parser.add_argument(
        "--training-run-id",
        type=str,
        default=None,
        help="Training run ID to store as a param for cross-reference",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Register adapter in MLflow registry and promote to Production",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run evaluation and optionally register and promote the adapter.
    """

    args = parse_args()

    # ── Plugin ─────────────────────────────────────────────────────────────
    module = importlib.import_module(f"tasks.{args.task}.eval")
    evaluator = module.get_evaluator()

    # ── Model ──────────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    config = AutoConfig.from_pretrained(args.base_model)
    text_config = getattr(config, "text_config", config)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=text_config,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # ── Data ───────────────────────────────────────────────────────────────
    val_dataset = load_from_disk(f"data/{args.task}/val")
    eval_df = evaluator.build_eval_df(val_dataset, num_samples=args.num_samples)

    # ── Eval components ────────────────────────────────────────────────────
    predict_fn = evaluator.make_predict_fn(model, tokenizer)
    kwargs: dict[str, Any] = {"db_dir": args.db_dir} if args.db_dir else {}
    extra_metrics = evaluator.get_metrics(eval_df, **kwargs)

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(f"qwen3.5-0.8b-{args.task}")

    params: dict[str, Any] = {
        "adapter_path": args.adapter_path,
        "base_model": args.base_model,
        "num_samples": args.num_samples if args.num_samples is not None else "full",
    }
    if args.training_run_id is not None:
        params["training_run_id"] = args.training_run_id

    with mlflow.start_run(run_name="eval") as run:
        mlflow.log_params(params)
        results = mlflow.models.evaluate(
            model=predict_fn,
            data=eval_df,
            targets="targets",
            extra_metrics=extra_metrics,
        )

        if args.promote:
            source_run_id = args.training_run_id or run.info.run_id
            register_and_promote(source_run_id, args.adapter_name)

    exec_acc_key = next((k for k in results.metrics if "exec_acc" in k), None)
    if exec_acc_key:
        print(f"Execution accuracy: {results.metrics[exec_acc_key]:.4f}")
    else:
        print("Metrics logged:", results.metrics)


if __name__ == "__main__":
    main()
