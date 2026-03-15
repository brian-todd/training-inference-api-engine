#!/usr/bin/env python3
"""
Smoke-test the SQL adapter via /v1/complete.

Usage
-----
    python tasks/sql/smoke_test.py [--url URL] [--adapter ADAPTER]

Defaults to http://localhost:8080 and adapter name "sql".
"""

import argparse
import sys

import httpx

EXAMPLES = [
    {
        "name": "simple select",
        "schema": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);",
        "question": "Get all users.",
    },
    {
        "name": "filtered query",
        "schema": (
            "CREATE TABLE orders ("
            "id INTEGER PRIMARY KEY, "
            "customer_id INTEGER, "
            "total REAL, "
            "status TEXT"
            ");"
        ),
        "question": "Find all orders with a total greater than 100 that are still pending.",
    },
    {
        "name": "join query",
        "schema": (
            "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER);\n"
            "CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT);"
        ),
        "question": "List each employee's name alongside their department name.",
    },
    {
        "name": "aggregation",
        "schema": (
            "CREATE TABLE sales (id INTEGER PRIMARY KEY, product TEXT, amount REAL, region TEXT);"
        ),
        "question": "What is the total sales amount per region, ordered from highest to lowest?",
    },
]


def run_example(
    client: httpx.Client,
    base_url: str,
    adapter: str,
    example: dict[str, str],
) -> None:
    """
    POST one example to /v1/complete and print the result.

    Parameters
    ----------
    client : httpx.Client
        Shared HTTP client.
    base_url : str
        Proxy base URL (e.g. ``http://localhost:8080``).
    adapter : str
        Adapter name to use.
    example : dict[str, str]
        Dict with ``name``, ``schema``, and ``question`` keys.
    """
    payload = {
        "adapter": adapter,
        "messages": [{"role": "user", "content": example["question"]}],
        "context": {"schema": example["schema"]},
    }

    print(f"\n{'─' * 60}")
    print(f"  {example['name']}")
    print(f"{'─' * 60}")
    print(f"Schema : {example['schema']}")
    print(f"Question: {example['question']}")

    try:
        response = client.post(f"{base_url}/v1/complete", json=payload)
        response.raise_for_status()
        data = response.json()
        sql = data["choices"][0]["message"]["content"]
        reasoning = data["choices"][0]["message"].get("reasoning_content")
        print(f"SQL     : {sql}")
        if reasoning:
            print(f"Thinking: {reasoning[:120]}{'...' if len(reasoning) > 120 else ''}")

    except httpx.HTTPStatusError as exc:
        print(f"ERROR {exc.response.status_code}: {exc.response.text}", file=sys.stderr)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)


def main() -> None:
    """Parse args and run all smoke-test examples."""
    parser = argparse.ArgumentParser(description="SQL adapter smoke test")
    parser.add_argument("--url", default="http://localhost:8080", help="Proxy base URL")
    parser.add_argument("--adapter", default="sql-lora", help="Adapter name")
    args = parser.parse_args()

    print(f"Proxy : {args.url}")
    print(f"Adapter: {args.adapter}")

    with httpx.Client(timeout=120.0) as client:
        for example in EXAMPLES:
            run_example(client, args.url, args.adapter, example)

    print(f"\n{'─' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
