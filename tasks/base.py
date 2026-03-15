from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from datasets import Dataset
    from mlflow.metrics import EvaluationMetric
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class DataPreparer:
    """
    Base class for task-specific data preparation plugins.

    Subclasses set ``dataset_name`` (and optionally ``dataset_config``) and
    override ``format_example`` to produce chat messages for each row.
    """

    dataset_name: str
    dataset_config: str | None = None

    def format_example(self, example: dict[str, Any]) -> list[dict[str, str]] | None:
        """
        Convert a raw dataset row into chat messages.

        Parameters
        ----------
        example : dict[str, Any]
            A single row from the HuggingFace dataset.

        Returns
        -------
        list[dict[str, str]] | None
            Chat message list (system / user / assistant), or ``None`` to drop
            the example.

        Raises
        ------
        NotImplementedError
            Subclasses must override this method.
        """
        raise NotImplementedError


class Processor:
    """
    Base processor with no-op defaults.

    Subclasses override methods to implement adapter-specific pre/post-processing.
    Unregistered adapters get this passthrough behavior.
    """

    def preprocess(
        self, messages: list[dict[str, str]], context: dict[str, Any]
    ) -> list[dict[str, str]]:
        """
        Transform messages before the vLLM call.

        Parameters
        ----------
        messages : list[dict[str, str]]
            Incoming chat messages.
        context : dict[str, Any]
            Arbitrary context from the request.

        Returns
        -------
        list[dict[str, str]]
            Messages, possibly transformed.
        """
        return messages

    def postprocess(self, content: str) -> tuple[str, dict[str, Any]]:
        """
        Process content after thinking extraction (non-streaming).

        Parameters
        ----------
        content : str
            Clean content with thinking blocks already removed.

        Returns
        -------
        tuple[str, dict[str, Any]]
            Possibly modified content and span attributes.
        """
        return content, {}

    def postprocess_stream(self, full_content: str) -> dict[str, Any]:
        """
        Process accumulated content after stream ends.

        Parameters
        ----------
        full_content : str
            Full accumulated clean content from the stream.

        Returns
        -------
        dict[str, Any]
            Span attributes to record.
        """
        return {}


class Evaluator:
    """Base class for task-specific evaluation plugins."""

    def build_eval_df(self, val_dataset: Dataset, num_samples: int | None = None) -> pd.DataFrame:
        """
        Convert HF val dataset to eval DataFrame with inputs/targets/... columns.

        Parameters
        ----------
        val_dataset : Dataset
            HuggingFace validation dataset.
        num_samples : int | None
            Optional number of examples to sample.

        Returns
        -------
        pd.DataFrame
            DataFrame with at least ``inputs`` and ``targets`` columns.
        """
        raise NotImplementedError

    def make_predict_fn(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
    ) -> Callable[[pd.DataFrame], pd.Series]:
        """
        Return predict callable for mlflow.evaluate(model=...).

        Parameters
        ----------
        model : PreTrainedModel
            Fine-tuned model.
        tokenizer : PreTrainedTokenizerBase
            Corresponding tokenizer.

        Returns
        -------
        Callable[[pd.DataFrame], pd.Series]
            Function mapping eval DataFrame rows to predicted strings.
        """
        raise NotImplementedError

    def get_metrics(self, eval_df: pd.DataFrame, **kwargs: object) -> list[EvaluationMetric]:
        """
        Return custom EvaluationMetric list.

        Parameters
        ----------
        eval_df : pd.DataFrame
            Eval DataFrame; metrics may close over it for column access.
        **kwargs : object
            Additional keyword arguments (e.g. ``db_dir`` for SQL).

        Returns
        -------
        list[EvaluationMetric]
            Custom metrics to pass to ``mlflow.evaluate``.
        """
        raise NotImplementedError
