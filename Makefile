.DEFAULT_GOAL := help

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

SHELL := /usr/bin/env bash

PYTHON ?= python3
VENV ?= .venv
CONTAINER_COMMAND ?= podman

PREFIX ?= ffreis
IMAGE_PROVIDER ?=
IMAGE_TAG ?= api-grpc-smoke
SMOKE_TIMEOUT ?= 20m
BASE_DIR ?= .
CONTAINER_DIR ?= container

IMAGE_PREFIX := $(if $(IMAGE_PROVIDER),$(IMAGE_PROVIDER)/,)$(PREFIX)
IMAGE_ROOT := $(IMAGE_PREFIX)

# ------------------------------------------------------------------------------
# Image names
# ------------------------------------------------------------------------------

BASE_IMAGE := $(IMAGE_PREFIX)/base
BASE_BUILDER_IMAGE := $(IMAGE_PREFIX)/base-builder
UV_VENV_IMAGE := $(IMAGE_PREFIX)/uv-venv
BUILDER_IMAGE := $(IMAGE_PREFIX)/builder
BASE_RUNNER_IMAGE := $(IMAGE_PREFIX)/base-runner
RUNNER_IMAGE := $(IMAGE_PREFIX)/runner

# ------------------------------------------------------------------------------
# Derived values
# ------------------------------------------------------------------------------

# Extract digests from digests.env (computed once)
BASE_IMAGE_VALUE := $(shell grep '^BASE_IMAGE=' $(CONTAINER_DIR)/digests.env | cut -d= -f2)
BASE_DIGEST_VALUE := $(shell grep '^BASE_DIGEST=' $(CONTAINER_DIR)/digests.env | cut -d= -f2)

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------

.PHONY: help
help: ## Show help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ------------------------------------------------------------------------------
# Meta targets
# ------------------------------------------------------------------------------

.PHONY: all
all: lint build run ## Lint, build, and run

# ------------------------------------------------------------------------------
# Tooling / setup
# ------------------------------------------------------------------------------

.PHONY: install-python-local
install-python-local: ## Install Python locally if missing
	@if command -v python3 >/dev/null 2>&1; then \
		echo "python3 already installed: $$(command -v python3)"; \
		exit 0; \
	fi
	sudo apt-get update
	sudo apt-get install -y python3 python3-pip

.PHONY: install-uv-local
install-uv-local: ## Install uv locally if missing
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv already installed: $$(command -v uv)"; \
		exit 0; \
	fi
	$(PYTHON) -m pip install --user --upgrade uv

.PHONY: install-podman-local
install-podman-local: ## Install Podman locally if missing
	@if command -v podman >/dev/null 2>&1; then \
		echo "podman already installed: $$(command -v podman)"; \
		exit 0; \
	fi
	sudo apt-get update
	sudo apt-get install -y podman

.PHONY: local-setup
local-setup: install-python-local install-uv-local install-podman-local ## Install local dev prerequisites

# ------------------------------------------------------------------------------
# Container builds
# ------------------------------------------------------------------------------

.PHONY: build-base
build-base: ## Build base image (pinned by digest env)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base -t $(BASE_IMAGE) $(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE_VALUE)" \
		--build-arg BASE_DIGEST="$(BASE_DIGEST_VALUE)"

.PHONY: build-base-builder
build-base-builder: build-base ## Build base-builder image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-builder -t $(BASE_BUILDER_IMAGE) $(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE)"

.PHONY: build-uv-venv
build-uv-venv: build-base build-base-builder ## Build shared uv-based venv image from uv.lock
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.uv-builder -t $(UV_VENV_IMAGE) $(BASE_DIR) \
		--build-arg BASE_BUILDER_IMAGE="$(BASE_BUILDER_IMAGE)"

.PHONY: build-builder
build-builder: build-uv-venv ## Build builder image (reuses venv image and runs tests)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.builder -t $(BUILDER_IMAGE) $(BASE_DIR) \
		--build-arg BASE_BUILDER_IMAGE="$(BASE_BUILDER_IMAGE)" \
		--build-arg UV_VENV_IMAGE="$(UV_VENV_IMAGE)"

.PHONY: build-base-runner
build-base-runner: build-base ## Build base-runner image
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.base-runner -t $(BASE_RUNNER_IMAGE) $(BASE_DIR) \
		--build-arg BASE_IMAGE="$(BASE_IMAGE)"

.PHONY: build-runner
build-runner: build-base-runner build-builder ## Build runner image (minimal Python runtime)
	$(CONTAINER_COMMAND) build -f $(CONTAINER_DIR)/Dockerfile.runner -t $(RUNNER_IMAGE) $(BASE_DIR) \
		--build-arg BASE_RUNNER_IMAGE="$(BASE_RUNNER_IMAGE)" \
		--build-arg BUILDER_IMAGE="$(BUILDER_IMAGE)" \
		--build-arg UV_VENV_IMAGE="$(UV_VENV_IMAGE)"

.PHONY: build-images
build-images: build-base build-base-builder build-uv-venv build-builder build-base-runner build-runner ## Build all images (may be slow)

.PHONY: build
build: build-images ## Build all container images

# ------------------------------------------------------------------------------
# Python (local) targets
# ------------------------------------------------------------------------------

.PHONY: env
env: ## Create local virtual environment
	@if [ -d "$(VENV)" ]; then \
		echo "Virtual environment already exists at $(VENV)"; \
	else \
		uv venv $(VENV); \
	fi
	@echo "Activate with: . $(VENV)/bin/activate"

.PHONY: build-local
build-local: env ## Install project and dev dependencies
	. $(VENV)/bin/activate && uv sync --active --frozen --extra dev

.PHONY: grpc-generate
grpc-generate: ## Regenerate protobuf/gRPC stubs
	./scripts/generate_grpc_stubs.sh

.PHONY: grpc-check
grpc-check: ## Verify protobuf/gRPC stubs are up to date
	./scripts/check_grpc_stubs.sh

.PHONY: openapi-check
openapi-check: ## Validate OpenAPI contract and verify runtime drift
	PYTHONPATH=src env -u VIRTUAL_ENV uv run --project . --with openapi-spec-validator --with pyyaml python scripts/check_openapi.py

.PHONY: openapi-drift-check
openapi-drift-check: ## Ensure API changes are accompanied by OpenAPI updates
	@test -n "$(BASE_SHA)" || (echo "BASE_SHA is required" && exit 1)
	@test -n "$(HEAD_SHA)" || (echo "HEAD_SHA is required" && exit 1)
	python3 scripts/check_openapi_drift.py --base "$(BASE_SHA)" --head "$(HEAD_SHA)"

.PHONY: grpc-clean
grpc-clean: ## Remove generated protobuf/gRPC stubs
	rm -f src/onnx_serving_grpc/inference_pb2.py src/onnx_serving_grpc/inference_pb2_grpc.py

.PHONY: run-app
run-app: ## Run the runner container
	$(CONTAINER_COMMAND) run $(RUNNER_IMAGE)

.PHONY: run
run: ## Run app locally
	uv run --active python main.py

.PHONY: run-container
run-container: run-app ## Alias: run the app in container

.PHONY: fmt
fmt: ## Format Python code
	uv run --with black python -m black .
	uv run --with ruff python -m ruff format .

.PHONY: fmt-check
fmt-check: ## Check Python formatting
	uv run --with black python -m black --check .
	uv run --with ruff python -m ruff format --check .

.PHONY: lint
lint: fmt-check ## Run linting + static typing
	uv run --with ruff python -m ruff check .
	uv run --with mypy python -m mypy src

.PHONY: test
test: ## Run all tests
	uv run --extra dev python -m pytest -q

.PHONY: test-unit
test-unit: ## Run unit tests
	uv run --extra dev python -m pytest -q tests/unit_tests

.PHONY: test-integration
test-integration: ## Run integration tests
	uv run --extra dev python -m pytest -q tests/integration_tests

.PHONY: test-e2e
test-e2e: ## Run e2e tests
	uv run --with pytest python -m pytest -q tests/e2e_tests

.PHONY: coverage
coverage: ## Run tests with coverage output
	uv run --with pytest python -m pytest \
		-q \
		--cov=src \
		--cov-report=term \
		--cov-report=xml:coverage.xml

.PHONY: test-grpc-parity
test-grpc-parity: ## Run gRPC/API parity tests
	uv run --with pytest python -m pytest -q tests/integration_tests/test_grpc_parity.py

.PHONY: test-grpc-parity-property
test-grpc-parity-property: ## Run gRPC/API parity property tests (Hypothesis)
	uv run --with pytest python -m pytest -q tests/integration_tests/test_grpc_parity.py -m property

.PHONY: smoke-api-grpc
smoke-api-grpc: ## Run docker-compose HTTP + gRPC smoke test
	@set -euo pipefail; \
	cleanup() { \
		IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" docker compose -f examples/docker-compose.api-grpc.yml down --remove-orphans || true; \
	}; \
	trap cleanup EXIT; \
	IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" docker compose -f examples/docker-compose.api-grpc.yml run --rm model-init; \
	IMAGE_ROOT="$(IMAGE_ROOT)" IMAGE_TAG="$(IMAGE_TAG)" timeout --foreground "$(SMOKE_TIMEOUT)" docker compose -f examples/docker-compose.api-grpc.yml up --build --exit-code-from smoke serving-api serving-grpc smoke

# ------------------------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------------------------

.PHONY: clean-repo
clean-repo: ## Clean repo build outputs
	rm -rf $(VENV) build __pycache__ .pytest_cache .coverage htmlcov *.pyc coverage.xml

.PHONY: clean-base
clean-base: ## Remove base image
	$(CONTAINER_COMMAND) rmi $(BASE_IMAGE) || true

.PHONY: clean-base-builder
clean-base-builder: ## Remove base-builder image
	$(CONTAINER_COMMAND) rmi $(BASE_BUILDER_IMAGE) || true

.PHONY: clean-builder
clean-builder: ## Remove builder image
	$(CONTAINER_COMMAND) rmi $(BUILDER_IMAGE) || true

.PHONY: clean-uv-venv
clean-uv-venv: ## Remove uv-venv image
	$(CONTAINER_COMMAND) rmi $(UV_VENV_IMAGE) || true

.PHONY: clean-base-runner
clean-base-runner: ## Remove base-runner image
	$(CONTAINER_COMMAND) rmi $(BASE_RUNNER_IMAGE) || true

.PHONY: clean-runner
clean-runner: ## Remove runner image
	$(CONTAINER_COMMAND) rmi $(RUNNER_IMAGE) || true

.PHONY: clean-all
clean-all: clean-repo clean-base clean-base-builder clean-uv-venv clean-builder clean-base-runner clean-runner ## Clean everything

.PHONY: ci-grpc
ci-grpc: grpc-check openapi-check lint test-grpc-parity ## Run gRPC sync + parity quality gate
