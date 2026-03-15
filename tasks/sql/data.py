"""
SQL data preparation plugin.
"""

from typing import Any

import sqlglot
from sqlglot.errors import ParseError

from tasks.base import DataPreparer

SQL_SYSTEM_PROMPT_TEMPLATE = (
    "You are a SQL expert. Given a database schema and a natural language "
    "question, write the correct SQL query.\n\n"
    "Schema:\n{schema}"
)


class SQLDataPreparer(DataPreparer):
    """
    Data preparer for the ``b-mc2/sql-create-context`` dataset.

    Validates each gold SQL answer with sqlglot and formats examples as
    three-turn chat messages (system with schema, user question, assistant SQL).
    """

    dataset_name = "b-mc2/sql-create-context"

    def format_example(self, example: dict[str, Any]) -> list[dict[str, str]] | None:
        """
        Format a sql-create-context row as chat messages.

        Parameters
        ----------
        example : dict[str, Any]
            Row with ``question``, ``context`` (CREATE TABLE DDL), and
            ``answer`` (gold SQL).

        Returns
        -------
        list[dict[str, str]] | None
            Three-message list, or ``None`` if the SQL fails validation.
        """
        try:
            sqlglot.parse_one(example["answer"])

        except ParseError:
            return None

        system_prompt = SQL_SYSTEM_PROMPT_TEMPLATE.format(schema=example["context"])

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]


def get_data_preparer() -> DataPreparer:
    """
    Return an ``SQLDataPreparer`` instance.

    Returns
    -------
    DataPreparer
        The SQL task data preparer.
    """
    return SQLDataPreparer()
