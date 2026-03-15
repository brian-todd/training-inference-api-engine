"""
Fine-tune a model with LoRA (optionally QLoRA) using any TRL trainer.

Usage
-----
SFT (default):

    uv run scripts/train.py --task sql

GRPO (requires tasks/{task}/trainer_extras.py with get_trainer_kwargs):

    uv run scripts/train.py --task sql --trainer grpo

DPO:

    uv run scripts/train.py --task sql --trainer dpo

Disable quantization (full bf16, requires more VRAM):

    uv run scripts/train.py --no-quantize

Custom hyperparameters:

    uv run scripts/train.py --lora-rank 32 --lora-alpha 64 --learning-rate 1e-4 --epochs 5

Save adapter to a different directory:

    uv run scripts/train.py --adapter-name my-adapter

Required environment variable:

    MLFLOW_TRACKING_URI   URI of the MLflow tracking server (e.g. http://localhost:5000)

Supported trainers: sft, grpo, dpo, orpo
Trainer-specific kwargs (e.g. reward_funcs for GRPO) are loaded from
tasks/{task}/trainer_extras.py via get_trainer_kwargs(trainer_name).
If that module does not exist the trainer is called with the common kwargs only.
"""

import argparse
import importlib
import os
from typing import Any

import mlflow
import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import (
    DPOConfig,
    DPOTrainer,
    GRPOConfig,
    GRPOTrainer,
    ORPOConfig,
    ORPOTrainer,
    SFTConfig,
    SFTTrainer,
)

# Maps CLI trainer name → (TrainerClass, ConfigClass).
# All configs inherit from TrainingArguments, so common hyperparameters work for every entry.
# Trainer-specific kwargs (e.g. reward_funcs for GRPO) come from tasks/{task}/trainer_extras.py.
TRAINER_REGISTRY: dict[str, tuple[type, type]] = {
    "sft": (SFTTrainer, SFTConfig),
    "grpo": (GRPOTrainer, GRPOConfig),
    "dpo": (DPOTrainer, DPOConfig),
    "orpo": (ORPOTrainer, ORPOConfig),
}


def _load_trainer_extras(task: str, trainer_name: str) -> dict[str, Any]:
    """
    Load trainer-specific kwargs from ``tasks/{task}/trainer_extras.py``.

    Imports the module and calls ``get_trainer_kwargs(trainer_name)``.  If the
    module does not exist for this task, returns an empty dict so the caller can
    proceed without it.

    Parameters
    ----------
    task : str
        Task name (e.g. ``"sql"``).
    trainer_name : str
        Trainer key from ``TRAINER_REGISTRY`` (e.g. ``"grpo"``).

    Returns
    -------
    dict[str, Any]
        Extra kwargs merged into the trainer constructor call.
    """
    try:
        module = importlib.import_module(f"tasks.{task}.trainer_extras")
    except ModuleNotFoundError:
        return {}
    return module.get_trainer_kwargs(trainer_name)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training hyperparameters.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with training configuration.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune with LoRA using any TRL trainer",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="sft",
        choices=list(TRAINER_REGISTRY),
        help="TRL trainer to use (default: sft)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sql",
        help="Task name; loads data from data/{task}/train and data/{task}/val (default: sql)",
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument(
        "--adapter-name",
        type=str,
        default="sql-lora",
        help="Adapter directory name under ./adapters/ (default: sql-lora)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="HuggingFace model ID (default: Qwen/Qwen3.5-0.8B)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Per-device batch size (default: 2)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable 4-bit quantization and load model in bf16 (requires more VRAM)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit train/val to N samples for quick smoke tests (default: 0 = use all)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the full fine-tuning pipeline.
    """
    args = parse_args()
    output_dir = f"./adapters/{args.adapter_name}"
    trainer_class, config_class = TRAINER_REGISTRY[args.trainer]

    # ── Model ──────────────────────────────────────────────────────────────
    bnb_config = (
        None
        if args.no_quantize
        else BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    )

    # Qwen3.5 is registered as a multimodal model in transformers — the top-level
    # config wraps text_config and lacks vocab_size.  Pass the text sub-config so
    # AutoModelForCausalLM resolves to the text-only CausalLM class.
    config = AutoConfig.from_pretrained(args.base_model)
    text_config = getattr(config, "text_config", config)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=text_config,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if args.no_quantize else None,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # ── LoRA ───────────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Data ───────────────────────────────────────────────────────────────
    train_dataset = load_from_disk(f"data/{args.task}/train")
    val_dataset = load_from_disk(f"data/{args.task}/val")

    if args.max_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(args.max_samples, len(val_dataset))))

    # ── Training config ─────────────────────────────────────────────────────
    # All config classes in TRAINER_REGISTRY inherit TrainingArguments, so common
    # hyperparameters work regardless of which trainer is selected.
    training_config = config_class(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="mlflow",
        gradient_checkpointing=True,
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    # Start with the kwargs every TRL trainer accepts, then merge in anything
    # trainer-specific (e.g. reward_funcs for GRPO) from the task module.
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "processing_class": tokenizer,
        "args": training_config,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "peft_config": peft_config,
    }
    trainer_kwargs.update(_load_trainer_extras(args.task, args.trainer))

    trainer = trainer_class(**trainer_kwargs)

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(f"qwen3.5-0.8b-{args.task}")

    run_name = f"{args.adapter_name}-{args.trainer}-r{args.lora_rank}-lr{args.learning_rate}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "trainer": args.trainer,
                "task": args.task,
                "base_model": args.base_model,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.grad_accum,
                "quantized": not args.no_quantize,
                "max_samples": args.max_samples or "all",
            }
        )

        trainer.train()
        trainer.save_model(output_dir)

        mlflow.log_artifacts(output_dir, artifact_path="adapter")


if __name__ == "__main__":
    main()
