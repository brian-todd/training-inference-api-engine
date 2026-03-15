# Training Inference API Engine

Fine-tuning and inference pipeline for Qwen3.5-0.8B with LoRA adapters. Trains with `peft`/`trl`,
tracks experiments in MLflow, serves with vLLM, and exposes an OpenAI-compatible proxy that
hot-swaps adapters and strips chain-of-thought tokens.

## Prerequisites

- Python 3.11 via `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker with the NVIDIA container runtime (`--runtime nvidia` must work)
- A HuggingFace account with access to `Qwen/Qwen3.5-0.8B`

## Setup

```bash
cp .env.example .env
# Fill in HF_TOKEN in .env (all other values have working defaults)

uv sync
make download-model   # pre-cache the base model (~1.6 GB)
```

## Starting the services

`make start` brings up Postgres, MinIO, MLflow, vLLM, the proxy, and Open WebUI in order,
waiting for each health check before proceeding. Times out after 600s by default.

```bash
make start                      # bring everything up in order, wait for health checks
make stop                       # graceful shutdown, preserve data
make clean                      # shutdown and destroy all volumes (full reset)
make download-model             # pre-cache base model (avoids slow first start)
make download-model MODEL=...   # cache a different model
```

| Service    | URL                   |
|------------|-----------------------|
| MLflow     | http://localhost:5000 |
| vLLM       | http://localhost:8000 |
| MinIO      | http://localhost:9001 |
| Proxy      | http://localhost:8080 |
| Open WebUI | http://localhost:3000 |

## Training pipeline

The full flow is: **prepare data → train → eval → promote**. Each step is a `make` target
that reads credentials from `.env` automatically.

### 1. Prepare data

Downloads and formats the task dataset, then saves train/val splits to `data/<task>/`.

```bash
make prepare-data TASK=sql
```

The SQL task uses [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context).
Each example is formatted as a three-turn chat (system with schema, user question, assistant SQL)
and validated with sqlglot. Progress and dataset stats are logged to MLflow.

### 2. Train

Fine-tunes `Qwen/Qwen3.5-0.8B` with QLoRA (4-bit NF4) and saves the adapter to
`./adapters/<adapter-name>/`.

```bash
make train TASK=sql
```

Key options (pass as `make` variables):

| Variable | Default | Description |
|---|---|---|
| `TASK` | `sql` | Task name — loads `data/<task>/train` and `data/<task>/val` |
| `EPOCHS` | `3` | Number of training epochs |

The run is tracked in MLflow under the experiment `qwen3.5-0.8b-<task>`. Note the **run ID**
shown at the end — you'll need it for the promote step.

For all training hyperparameters (`--lora-rank`, `--learning-rate`, `--epochs`, etc.) run:

```bash
uv run scripts/train.py --help
```

### 3. Evaluate

Runs the adapter against the validation set and logs a per-example results table
(inputs / predictions / gold SQL / exec_acc score) to MLflow via `mlflow.models.evaluate`.

```bash
make eval ADAPTER_PATH=./adapters/sql-lora
```

Key options:

| Variable | Default | Description |
|---|---|---|
| `ADAPTER_PATH` | *(required)* | Path to the saved adapter directory |
| `TASK` | `sql` | Task plugin to use for eval |
| `NUM_SAMPLES` | *(full val set)* | Evaluate on a random subset (faster for iteration) |
| `TRAINING_RUN_ID` | *(none)* | Links the eval run to the training run in MLflow |

Example with a sample limit for quick iteration:

```bash
make eval ADAPTER_PATH=./adapters/sql-lora NUM_SAMPLES=50
```

After eval completes, check `exec_acc/mean` in the MLflow UI at http://localhost:5000
under **Experiments → qwen3.5-0.8b-sql**. The per-example table is on the **Evaluation** tab.

### 4. Promote

Registers the adapter in the MLflow Model Registry and sets the **production** alias,
making it available for the proxy to serve.

```bash
make promote RUN_ID=<training-run-id> ADAPTER_NAME=sql-lora
```

The `RUN_ID` should be the **training** run ID (not the eval run ID), so the registered
model version links back to the adapter artifact logged during training.

#### Eval + promote in one step

Once you're confident in the process, you can combine eval and promote:

```bash
make eval ADAPTER_PATH=./adapters/sql-lora TRAINING_RUN_ID=<run-id> PROMOTE=1
```

## Using a promoted adapter

Once an adapter has the production alias, there are two ways to use it.

### Open WebUI (browser chat)

```bash
make start   # if not already running
```

Open http://localhost:3000, create an account, and start chatting. Open WebUI talks to the
proxy at `/v1/chat/completions` using the base model — no adapter selection needed here.

### Adapter-specific inference (proxy API)

Use the `/v1/complete` endpoint to query a specific adapter directly. The proxy resolves the
adapter from the MLflow registry, loads it into vLLM if needed, and applies any registered
pre/post-processing (schema injection, SQL validation, etc.).

```bash
curl http://localhost:8080/v1/complete \
  -H "Content-Type: application/json" \
  -d '{
    "adapter": "sql-lora",
    "messages": [{"role": "user", "content": "List all customers from New York"}]
  }'
```

To pre-warm the adapter before your first request (avoids cold-start latency):

```bash
curl -X POST http://localhost:8080/v1/load_adapter \
  -H "Content-Type: application/json" \
  -d '{"adapter": "sql-lora"}'
```

## Common commands

```bash
uv sync                                        # install / sync dependencies
make start                                     # start all services
make stop                                      # stop all services
make fmt                                       # format with ruff
make lint                                      # lint with ruff
uv run scripts/train.py --help                 # all training options
uv run scripts/eval.py --help                  # all eval options
docker compose logs -f <service>               # tail logs
```

## Development

```bash
cd proxy && uv run pytest     # run proxy tests
make lint                      # lint with ruff
make fmt                       # format with ruff
```

Code conventions (style, naming, docstrings) are documented in `CLAUDE.md`.

### Task plugins

The training pipeline is organized around task plugins in `tasks/`. Each task provides three
components by subclassing the bases in `tasks/base.py`:

- **`DataPreparer`** — downloads a HuggingFace dataset and formats each row into chat messages
- **`Processor`** — pre/post-processing for inference (e.g. schema injection, SQL validation)
- **`Evaluator`** — builds an eval DataFrame and custom metrics for `mlflow.evaluate`

To add a new task, create a package under `tasks/<task_name>/` that implements these three
classes, then register it in `tasks/__init__.py`. See `tasks/sql/` for a working example.

## Architecture

```
┌────────────┐                   ┌──────────┐
│ Open WebUI │ ── /v1/chat ────► │  Proxy   │
│   :3000    │   /completions    │  :8080   │
└────────────┘                   └────┬─────┘
                    ┌─────────────────┤
                    ▼                 ▼
              ┌──────────┐    ┌──────────┐
              │   vLLM   │    │  MLflow  │
              │  :8000   │    │  :5000   │
              └──────────┘    └──────────┘
                                    │
                              ┌─────┴──────┐
                              │  Postgres  │
                              │  + MinIO   │
                              └────────────┘
```

The proxy is the single entry point. It exposes two API families:

- **`/v1/chat/completions`** — OpenAI-compatible general-purpose chat (streaming and
  non-streaming). Open WebUI uses this for the browser-based chat interface.
- **`/v1/complete`** — Adapter-specific endpoint that resolves adapters from MLflow and
  applies registered pre/post-processing via the processor system (e.g. the SQL processor
  injects schema context and validates output with sqlglot).

The `/v1/chat/completions` path strips chain-of-thought thinking tokens (configurable
tags, defaulting to `<think>`/`</think>`) before returning responses. The `/v1/complete`
path additionally logs latency to MLflow.
Adapters are stored in `./adapters/` (shared bind mount with vLLM) and loaded via vLLM's
runtime LoRA API.

### API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness check (pings vLLM) |
| `GET` | `/v1/models` | OpenAI-compatible model listing |
| `GET` | `/v1/adapters` | List loaded adapters in vLLM |
| `POST` | `/v1/chat/completions` | General-purpose chat (streaming supported) |
| `POST` | `/v1/complete` | Adapter-specific inference with pre/post-processing |
| `POST` | `/v1/load_adapter` | Pre-warm an adapter from MLflow |
