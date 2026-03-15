"""SQL-specific evaluation plugin for mlflow.evaluate()."""

import re
import sqlite3
from collections.abc import Callable
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from mlflow.metrics import EvaluationMetric, MetricValue, make_metric
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from tasks.base import Evaluator
from tasks.sql.data import SQL_SYSTEM_PROMPT_TEMPLATE


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _execute_sql_inmemory(ddl: str, sql: str) -> frozenset[tuple[Any, ...]] | None:
    """
    Execute a SQL query against an in-memory SQLite database seeded with DDL.

    Parameters
    ----------
    ddl : str
        CREATE TABLE statements to initialise the database.
    sql : str
        SQL query to execute.

    Returns
    -------
    frozenset[tuple[Any, ...]] | None
        Unordered result set, or ``None`` if execution fails.
    """
    try:
        conn = sqlite3.connect(":memory:")
        conn.executescript(ddl)
        cursor = conn.execute(sql)
        results = frozenset(tuple(row) for row in cursor.fetchall())
        conn.close()
        return results
    except Exception:
        return None


class SQLEvaluator(Evaluator):
    """
    Evaluation plugin for sql-create-context text-to-SQL using execution accuracy.
    """

    def build_eval_df(self, val_dataset: Dataset, num_samples: int | None = None) -> pd.DataFrame:
        """
        Convert sql-create-context HF val dataset to an eval DataFrame.

        Parameters
        ----------
        val_dataset : Dataset
            HuggingFace dataset with ``question``, ``context``, and ``answer`` columns.
        num_samples : int | None
            If set, shuffle with seed=42 and select this many examples.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ``inputs``, ``schema``, ``targets``.
        """
        examples = val_dataset
        if num_samples is not None:
            examples = val_dataset.shuffle(seed=42).select(range(num_samples))

        rows = []
        for example in examples:
            rows.append(
                {
                    "inputs": example["question"],
                    "schema": example["context"],
                    "targets": example["answer"],
                }
            )
        return pd.DataFrame(rows)

    def make_predict_fn(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
    ) -> Callable[[pd.DataFrame], pd.Series]:
        """
        Return a predict function closing over model and tokenizer.

        Parameters
        ----------
        model : PreTrainedModel
            Fine-tuned model.
        tokenizer : PreTrainedTokenizerBase
            Corresponding tokenizer.

        Returns
        -------
        Callable[[pd.DataFrame], pd.Series]
            Predict function for ``mlflow.evaluate(model=...)``.
        """

        def predict_fn(eval_df: pd.DataFrame) -> pd.Series:
            predictions = []
            for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Generating"):
                system_prompt = SQL_SYSTEM_PROMPT_TEMPLATE.format(schema=row["schema"])
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["inputs"]},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                generated = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                predictions.append(_strip_thinking(generated))
            return pd.Series(predictions)

        return predict_fn

    def get_metrics(self, eval_df: pd.DataFrame, **kwargs: object) -> list[EvaluationMetric]:
        """
        Return execution accuracy as an MLflow EvaluationMetric.

        Executes both gold and predicted SQL against an in-memory SQLite database
        seeded from the ``schema`` column, so no external database files are needed.

        Parameters
        ----------
        eval_df : pd.DataFrame
            Eval DataFrame; closed over to access ``schema`` by row index.
        **kwargs : object
            Accepted but unused — kept for interface compatibility.

        Returns
        -------
        list[EvaluationMetric]
            Single-element list with ``exec_acc`` metric.
        """

        def exec_acc_fn(
            predictions: pd.Series, targets: pd.Series, metrics: dict[str, Any]
        ) -> MetricValue:
            scores: list[float | None] = []
            for idx, pred_sql in enumerate(predictions):
                ddl = eval_df["schema"].iloc[idx]
                gold_sql = targets.iloc[idx]
                gold_result = _execute_sql_inmemory(ddl, gold_sql)
                if gold_result is None:
                    scores.append(None)
                    continue
                pred_result = _execute_sql_inmemory(ddl, pred_sql)
                scores.append(1.0 if pred_result == gold_result else 0.0)
            valid = [s for s in scores if s is not None]
            mean = sum(valid) / len(valid) if valid else 0.0
            return MetricValue(scores=scores, aggregate_results={"mean": mean})

        return [make_metric(eval_fn=exec_acc_fn, greater_is_better=True, name="exec_acc")]


def get_evaluator() -> Evaluator:
    """
    Return the SQL evaluator instance.

    Returns
    -------
    Evaluator
        A ``SQLEvaluator`` instance.
    """
    return SQLEvaluator()
