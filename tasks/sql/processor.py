from collections.abc import Callable
from typing import Any

import mlflow
import sqlglot

from tasks.base import Processor


class SQLProcessor(Processor):
    """
    Processor for the SQL task.

    Pre-processing injects the SQL schema as a system message using the MLflow
    Prompt Registry. Post-processing validates the output with sqlglot.
    """

    prompt_name: str = "sql-system-prompt"

    def preprocess(
        self, messages: list[dict[str, str]], context: dict[str, Any]
    ) -> list[dict[str, str]]:
        """
        Inject SQL schema context as a system message.

        Parameters
        ----------
        messages : list[dict[str, str]]
            Incoming chat messages.
        context : dict[str, Any]
            Must contain ``"schema"`` key with the SQL schema string.

        Returns
        -------
        list[dict[str, str]]
            Messages with schema injected as (or prepended to) the system message.
        """
        schema = context.get("schema", "")
        if not schema:
            return messages

        prompt = mlflow.genai.load_prompt(f"prompts:/{self.prompt_name}@production")
        system_content = prompt.format(schema=schema)

        if messages and messages[0]["role"] == "system":
            augmented = {**messages[0], "content": system_content + "\n\n" + messages[0]["content"]}
            return [augmented, *messages[1:]]

        return [{"role": "system", "content": system_content}, *messages]

    def postprocess(self, content: str) -> tuple[str, dict[str, Any]]:
        """
        Validate SQL output with sqlglot.

        Parameters
        ----------
        content : str
            Clean content after thinking extraction.

        Returns
        -------
        tuple[str, dict[str, Any]]
            Unmodified content and ``{"sql_valid": bool}`` span attribute.
        """
        sql_valid = True
        try:
            sqlglot.parse_one(content)

        except sqlglot.errors.ParseError:
            sql_valid = False

        return content, {"sql_valid": sql_valid}

    def postprocess_stream(self, full_content: str) -> dict[str, Any]:
        """
        Validate accumulated SQL after stream ends.

        Parameters
        ----------
        full_content : str
            Full accumulated clean content from the stream.

        Returns
        -------
        dict[str, Any]
            ``{"sql_valid": bool}`` span attribute.
        """
        sql_valid = True
        try:
            sqlglot.parse_one(full_content)

        except sqlglot.errors.ParseError:
            sql_valid = False

        return {"sql_valid": sql_valid}


def register_processor(register_fn: Callable[[str, Processor], None]) -> None:
    """
    Register this task's processor with the proxy registry.

    Parameters
    ----------
    register_fn : Callable[[str, Processor], None]
        The registry's ``register`` function, called with the task name and
        processor instance.
    """
    register_fn("sql", SQLProcessor())
