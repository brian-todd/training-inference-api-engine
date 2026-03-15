"""Trainer-specific kwargs for the SQL task."""

from collections.abc import Sequence


def get_trainer_kwargs(trainer_name: str) -> dict[str, object]:
    """
    Return extra kwargs to merge into the trainer constructor for the SQL task.

    Parameters
    ----------
    trainer_name : str
        One of the registered trainer keys (e.g. ``"sft"``, ``"grpo"``).

    Returns
    -------
    dict[str, object]
        Kwargs merged into the trainer constructor call.
        For ``"grpo"``, includes ``reward_funcs``.
    """
    if trainer_name == "grpo":
        return {"reward_funcs": _get_reward_funcs()}
    return {}


def _sql_format_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    **kwargs: object,
) -> list[float]:
    """
    Reward function: +1 if the completion looks like a valid SQL SELECT statement.

    Parameters
    ----------
    prompts : Sequence[str]
        Input prompts (unused here, but required by the GRPO interface).
    completions : Sequence[str]
        Model completions to score.
    **kwargs : object
        Additional keyword arguments passed by GRPOTrainer (e.g. ground-truth answers).

    Returns
    -------
    list[float]
        Per-completion reward scores in [0.0, 1.0].
    """
    rewards = []
    for completion in completions:
        cleaned = completion.strip().upper()
        rewards.append(1.0 if cleaned.startswith("SELECT") else 0.0)
    return rewards


def _get_reward_funcs() -> list:
    """
    Return the ordered list of reward functions for GRPO training on the SQL task.

    Returns
    -------
    list
        Reward function callables passed to ``GRPOTrainer(reward_funcs=...)``.
    """
    return [_sql_format_reward]
