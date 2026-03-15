"""
Register task prompts in the MLflow Prompt Registry and set aliases.

Walks ``tasks/*/prompts/*.yaml`` and registers each prompt with MLflow,
then sets the configured alias on the registered version.

Usage
-----
Register all task prompts:

    uv run scripts/register_prompts.py

Register prompts for a single task:

    uv run scripts/register_prompts.py --task sql

Required environment variable:

    MLFLOW_TRACKING_URI   URI of the MLflow tracking server (e.g. http://localhost:5000)
"""

import argparse
import os
from pathlib import Path

import mlflow
import yaml
from mlflow.genai import register_prompt, set_prompt_alias


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for prompt registration.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with optional task filter.
    """

    parser = argparse.ArgumentParser(
        description="Register task prompts in the MLflow Prompt Registry"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Register prompts for a single task only (default: all tasks)",
    )
    return parser.parse_args()


def register_task_prompts(prompts_dir: Path) -> None:
    """
    Register all prompts found under a task's prompts directory.

    Parameters
    ----------
    prompts_dir : Path
        Path to a ``tasks/{task}/prompts/`` directory containing YAML files.
    """

    for yaml_path in sorted(prompts_dir.glob("*.yaml")):
        with yaml_path.open() as file_handle:
            config = yaml.safe_load(file_handle)

        name = config["name"]
        template = config["template"]
        commit_message = config.get("commit_message", "")
        alias = config.get("alias")

        prompt = register_prompt(
            name=name,
            template=template,
            commit_message=commit_message,
        )

        if alias:
            set_prompt_alias(name=name, alias=alias, version=prompt.version)

        alias_suffix = f" @{alias}" if alias else ""
        print(f"Registered '{name}' v{prompt.version}{alias_suffix}  ({yaml_path})")


def main() -> None:
    """
    Walk task prompt directories and register each prompt with MLflow.
    """

    args = parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    tasks_root = Path(__file__).parent.parent / "tasks"

    if args.task:
        task_dirs = [tasks_root / args.task]

    else:
        task_dirs = sorted(tasks_root.iterdir()) if tasks_root.exists() else []

    for task_dir in task_dirs:
        prompts_dir = task_dir / "prompts"
        if prompts_dir.is_dir():
            register_task_prompts(prompts_dir)


if __name__ == "__main__":
    main()
