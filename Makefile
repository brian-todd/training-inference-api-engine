.PHONY: start start-mlflow stop clean download-model prepare-data train eval promote register-prompts lint fmt

ifneq (,$(wildcard .env))
  include .env
  export
endif

MODEL ?= Qwen/Qwen3.5-0.8B
TASK  ?= sql

# timeout for health-check loops (seconds)
HEALTH_TIMEOUT ?= 600

download-model:
	@echo "Downloading $(MODEL) to ~/.cache/huggingface..."
	uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$(MODEL)')"
	@echo "Model cached."

define wait_for_healthy
	@elapsed=0; \
	while ! curl -sf $(1) > /dev/null 2>&1; do \
		elapsed=$$((elapsed + $(2))); \
		if [ $$elapsed -ge $(HEALTH_TIMEOUT) ]; then \
			echo "ERROR: $(3) did not become healthy within $(HEALTH_TIMEOUT)s"; \
			exit 1; \
		fi; \
		sleep $(2); \
	done; \
	echo "$(3) ready."
endef

start:
	@test -f .env || { echo "ERROR: .env file not found. Copy .env.example and fill in values."; exit 1; }
	docker compose --env-file .env up -d postgres minio mc-init mlflow
	@echo "Waiting for MLflow..."
	$(call wait_for_healthy,http://localhost:5000/health,2,MLflow)
	docker compose --env-file .env up -d vllm
	@echo "Waiting for vLLM (up to $(HEALTH_TIMEOUT)s)..."
	$(call wait_for_healthy,http://localhost:8000/health,5,vLLM)
	docker compose --env-file .env up -d --build proxy
	@echo "Waiting for proxy..."
	$(call wait_for_healthy,http://localhost:8080/health,2,Proxy)
	docker compose --env-file .env up -d open-webui
	@echo "Waiting for Open WebUI..."
	$(call wait_for_healthy,http://localhost:3000,3,Open WebUI)
	@echo ""
	@echo "All services up:"
	@echo "  MLflow     → http://localhost:5000"
	@echo "  vLLM       → http://localhost:8000"
	@echo "  MinIO      → http://localhost:9001"
	@echo "  Proxy      → http://localhost:8080"
	@echo "  Open WebUI → http://localhost:3000"

start-mlflow:
	@test -f .env || { echo "ERROR: .env file not found. Copy .env.example and fill in values."; exit 1; }
	docker compose --env-file .env up -d postgres minio mc-init mlflow
	@echo "Waiting for MLflow..."
	$(call wait_for_healthy,http://localhost:5000/health,2,MLflow)
	@echo ""
	@echo "MLflow stack up:"
	@echo "  MLflow → http://localhost:5000"
	@echo "  MinIO  → http://localhost:9001"

stop:
	docker compose down

clean:
	docker compose down -v

ENV := MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	MLFLOW_S3_ENDPOINT_URL=$(MLFLOW_S3_ENDPOINT_URL) \
	AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
	AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
	HF_TOKEN=$(HF_TOKEN)

prepare-data:
	$(ENV) uv run scripts/prepare_data.py --task $(TASK)

train:
	$(ENV) uv run scripts/train.py --task $(TASK) \
	  $(if $(EPOCHS),--epochs $(EPOCHS),) \
	  $(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES),)

eval:
	@test -n "$(ADAPTER_PATH)" || { echo "Usage: make eval ADAPTER_PATH=<path> [NUM_SAMPLES=50] [TRAINING_RUN_ID=<id>] [PROMOTE=1]"; exit 1; }
	$(ENV) uv run scripts/eval.py --task $(TASK) \
	  --adapter-path $(ADAPTER_PATH) \
	  $(if $(DB_DIR),--db-dir $(DB_DIR),) \
	  $(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),) \
	  $(if $(TRAINING_RUN_ID),--training-run-id $(TRAINING_RUN_ID),) \
	  $(if $(PROMOTE),--promote,)

promote:
	@test -n "$(RUN_ID)" || { echo "Usage: make promote RUN_ID=<mlflow-run-id> [ADAPTER_NAME=sql-lora]"; exit 1; }
	$(ENV) uv run scripts/promote_adapter.py --run-id $(RUN_ID) $(if $(ADAPTER_NAME),--adapter-name $(ADAPTER_NAME),)

register-prompts:
	$(ENV) uv run scripts/register_prompts.py $(if $(TASK),--task $(TASK),)

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
