"""Prepare a fine-tuning dataset for a given task.

Downloads the task's HuggingFace dataset, formats examples as chat messages
via a task-specific plugin, tokenizes and filters by sequence length, creates
a train/val split, logs metadata to MLflow, and saves processed splits to disk.

Usage
-----
Basic run with defaults (task=sql):

    uv run scripts/prepare_data.py

Specify a task explicitly:

    uv run scripts/prepare_data.py --task sql

Custom settings:

    uv run scripts/prepare_data.py --task sql --max-seq-length 4096 --val-fraction 0.1

Required environment variable:

    MLFLOW_TRACKING_URI   URI of the MLflow tracking server (e.g. http://localhost:5000)
"""

import argparse
import importlib
import os
from typing import Any

import mlflow
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for data preparation.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with data preparation configuration.
    """

    parser = argparse.ArgumentParser(description="Prepare a fine-tuning dataset for a given task")
    parser.add_argument(
        "--task",
        type=str,
        default="sql",
        help="Task name matching a tasks/{task}/data.py plugin (default: sql)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model ID for tokenizer (default: Qwen/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of data to hold out for validation (default: 0.05)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Root output directory; splits saved to {output_dir}/{task}/train|val (default: data)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the data preparation pipeline.
    """

    args = parse_args()

    # ── Load task plugin ─────────────────────────────────────────────────────
    module = importlib.import_module(f"tasks.{args.task}.data")
    preparer = module.get_data_preparer()

    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"Loading dataset {preparer.dataset_name}...")
    dataset = load_dataset(preparer.dataset_name, preparer.dataset_config)

    if "validation" in dataset:
        train_split = dataset["train"]
        val_split = dataset["validation"]
    else:
        splits = dataset["train"].train_test_split(test_size=args.val_fraction, seed=42)
        train_split = splits["train"]
        val_split = splits["test"]

    print(f"  Train: {len(train_split)} examples | Val: {len(val_split)} examples")

    # ── Format examples via plugin ────────────────────────────────────────────
    print(f"Formatting examples with {args.task} plugin...")

    def add_messages(example: dict[str, Any]) -> dict[str, Any]:
        """
        Add a ``messages`` field to the example via the task plugin.

        Parameters
        ----------
        example : dict[str, Any]
            A single row from the dataset.

        Returns
        -------
        dict[str, Any]
            Example with ``messages`` field added (None if the plugin drops it).
        """
        return {**example, "messages": preparer.format_example(example)}

    train_with_msgs = train_split.map(add_messages)
    val_with_msgs = val_split.map(add_messages)

    train_valid = train_with_msgs.filter(lambda row: row["messages"] is not None)
    val_valid = val_with_msgs.filter(lambda row: row["messages"] is not None)

    train_dropped = len(train_split) - len(train_valid)
    val_dropped = len(val_split) - len(val_valid)
    print(f"  Plugin dropped: train={train_dropped}, val={val_dropped}")

    # ── Tokenize and filter by length ─────────────────────────────────────────
    print(f"Tokenizing with {args.base_model} (max_seq_length={args.max_seq_length})...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def add_token_count(example: dict[str, Any]) -> dict[str, Any]:
        """
        Compute token count for the chat-formatted example.

        Parameters
        ----------
        example : dict[str, Any]
            A row with a ``messages`` field.

        Returns
        -------
        dict[str, Any]
            Example with ``token_count`` field added.
        """
        token_ids = tokenizer.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=True
        )
        return {**example, "token_count": len(token_ids)}

    train_counted = train_valid.map(add_token_count)
    val_counted = val_valid.map(add_token_count)

    train_filtered = train_counted.filter(lambda row: row["token_count"] <= args.max_seq_length)
    val_filtered = val_counted.filter(lambda row: row["token_count"] <= args.max_seq_length)

    train_len_dropped = len(train_counted) - len(train_filtered)
    val_len_dropped = len(val_counted) - len(val_filtered)
    print(f"  Length filter dropped: train={train_len_dropped}, val={val_len_dropped}")
    print(f"  Final: train={len(train_filtered)}, val={len(val_filtered)}")

    # ── Save to disk ──────────────────────────────────────────────────────────
    train_path = f"{args.output_dir}/{args.task}/train"
    val_path = f"{args.output_dir}/{args.task}/val"
    train_filtered.save_to_disk(train_path)
    val_filtered.save_to_disk(val_path)
    print(f"  Saved to {train_path} and {val_path}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(f"qwen3.5-0.8b-{args.task}")

    with mlflow.start_run(run_name="data-prep"):
        mlflow.log_params(
            {
                "task": args.task,
                "dataset_name": preparer.dataset_name,
                "max_seq_length": args.max_seq_length,
                "val_fraction": args.val_fraction,
                "train_size": len(train_filtered),
                "val_size": len(val_filtered),
                "train_plugin_dropped": train_dropped,
                "val_plugin_dropped": val_dropped,
                "train_length_dropped": train_len_dropped,
                "val_length_dropped": val_len_dropped,
            }
        )

    print("  MLflow run logged.")
    print("Done.")


if __name__ == "__main__":
    main()
