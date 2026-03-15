# CLAUDE.md — training-inference-api-engine

This repo implements a full fine-tuning and inference pipeline: a training script that fine-tunes
a base LLM with LoRA adapters and logs everything to MLflow, a vLLM inference server that hot-swaps
adapters, and an OpenAI-compatible proxy that routes requests, strips chain-of-thought thinking
tokens, and records per-request metrics back to MLflow.

---

## Development environment

- **Python:** 3.11, managed via `uv`
- **Install deps:** `uv sync`
- **Run scripts:** `uv run <script>`
- **Notebooks:** marimo (`uv run marimo edit notebooks/<name>.py`)
- **Services:** Docker Compose manages vLLM, the proxy, and MLflow

---

## Common commands

```bash
uv sync                          # install / sync dependencies
uv run scripts/train.py          # run training
docker compose up -d             # start all services
docker compose logs -f mlflow    # tail MLflow logs
uv run ruff format .             # format code
uv run ruff check .              # lint code
```

---

## Code style

### Formatter and linter: Ruff

Ruff is the sole formatter and linter. Config lives in root `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "ANN"]

[tool.ruff.lint.pydocstyle]
convention = "numpy"
```

### Type hints

Required on all function signatures — every parameter and the return type.
Prefer `collections.abc` over `typing` for `Callable`, `Sequence`, `Iterator`, etc.

```python
from collections.abc import Sequence


def select_top_adapters(scores: Sequence[float], top_k: int) -> list[str]:
    ...
```

### Docstrings

NumPy style on all public functions, classes, and methods. The summary goes on its own
line after the opening `"""`, with a blank line before sections. Add `Parameters`,
`Returns`, and `Raises` sections as needed. Omit docstrings on private helpers
(`_prefixed`) unless the logic is non-obvious.

```python
def load_adapter(adapter_name: str, base_model_path: str) -> dict[str, object]:
    """
    Load a LoRA adapter and merge it with the base model config.

    Parameters
    ----------
    adapter_name : str
        Name of the adapter directory under ``./adapters/``.
    base_model_path : str
        Filesystem path to the base model weights.

    Returns
    -------
    dict[str, object]
        Merged config dict ready to pass to the vLLM client.

    Raises
    ------
    FileNotFoundError
        If the adapter directory does not exist.
    """
```

### Variable naming

**No single-letter variable names — ever.** This applies to loop counters, comprehensions,
lambdas, and anywhere else. Examples:

| Instead of | Use |
|------------|-----|
| `i` | `idx` |
| `j` | `jdx` |
| `r` | `row` |
| `c` | `col_name` |
| `x` | `value` or a domain-specific name |
| `n` | `count` or `num_samples` |
| `f` | `file_handle` or `func` |

```python
# Wrong
for i, r in enumerate(rows):
    ...

# Right
for idx, row in enumerate(rows):
    ...
```

---

## Project-specific conventions

- **MLflow tracking URI** always comes from an environment variable or config object.
  Never hardcode a tracking URI in library code.
- **Adapter paths** follow the `./adapters/{adapter_name}/` convention. The `./adapters/`
  directory is bind-mounted into both the vLLM container (`/adapters`) and the proxy
  container (`/app/adapters`).
- **vLLM interactions** must go through `proxy/vllm_client.py`. Do not call vLLM endpoints
  directly from any other module.
- **Adapter-specific processing:** use the processor system in `proxy/processors/`.
  `processors.get_processor(adapter_name)` returns the registered processor (or a no-op
  default). Pre/post-processing logic (e.g. schema injection, SQL validation) belongs in
  processor subclasses, not in route handlers.
- **Thinking-mode stripping:** use `extract_thinking()` from `proxy/thinking.py` to remove
  chain-of-thought tokens before returning responses.

---

## Notebook conventions

Notebooks are marimo `.py` files. Open them with `uv run marimo edit notebooks/<name>.py`
and run headless with `uv run marimo run notebooks/<name>.py`.

- Marimo cells are reactive functions — each cell is a `def` decorated by the app object.
  Do not write imperative top-level code outside of cells.
- **First cell:** imports only.
- **Second cell:** config and constants (MLflow URI, model name, adapter name, etc.).
- Log everything to MLflow: params, metrics, and artifacts. No silent runs.

---

## Architecture reminders

Key constraints:

- `./adapters/` is the single source of truth for adapter weights; both containers mount it.
- Keep `docker-compose.yml` infra-only — no application logic, no environment-specific
  secrets baked in.
- The proxy is the **only** service that writes to MLflow at request time. Training scripts
  also write to MLflow, but vLLM does not.
