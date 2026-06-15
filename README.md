# ffreis-python-model-serving

<!-- ffreis-badges:start -->
[![CI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/FelipeFuhr/ffreis-badges/main/badges/ffreis-python-model-serving/ci.json)](https://github.com/FelipeFuhr/ffreis-python-model-serving/actions)
<!-- ffreis-badges:end -->

A universal Python model-serving microservice. It loads a trained model
artifact from a directory and exposes it for online inference over an HTTP API
(FastAPI behind Gunicorn) and an optional gRPC API. Inference backends are
pluggable adapters — ONNX is the default production path, with optional
scikit-learn, PyTorch, and TensorFlow runtimes. The HTTP surface is
SageMaker BYOC-compatible (`/ping`, `/invocations`, `SM_MODEL_DIR`) and the
service is shipped as a minimal, non-root, multi-stage container image.

## What it does

- Loads a single model artifact from `SM_MODEL_DIR` and serves predictions.
- Selects an inference adapter from `MODEL_TYPE` (`onnx`, `sklearn`, `pytorch`,
  `tensorflow`); when unset, the model type is inferred from `MODEL_FILENAME`
  or from a known default filename in the model dir
  (`model.onnx`, `model.joblib`/`model.pkl`, `model.pt`, `model.keras` /
  `saved_model/`).
- Parses tabular inference inputs from JSON, JSON Lines (NDJSON), and CSV; casts
  to a configurable dtype; optionally splits ID columns from feature columns;
  validates the feature count and batch size.
- Supports ONNX multi-input models via a JSON request-key → tensor-name map
  (`ONNX_INPUT_MAP_JSON`, with optional per-input dtype map).
- Formats predictions as JSON (predictions-only or wrapped under a configurable
  key) or CSV, honouring the `Accept` header.
- Bounds load: an in-flight concurrency semaphore (returns `429` when saturated)
  and a request body size limit (returns `413` when exceeded).
- Emits OpenTelemetry traces (OTLP export, W3C `traceparent`/`tracestate`
  propagation) and Prometheus metrics, both toggleable.

Core logic (parsing, formatting, adapters) is IO-isolated behind a `BaseAdapter`
contract and deterministic; the HTTP and gRPC layers are thin transports over
the same parse → predict → format pipeline.

## HTTP surface

The FastAPI application (`src/application.py`, ASGI app `serving:application`)
exposes:

| Method | Path | Purpose |
|---|---|---|
| GET | `/live`, `/healthz` | Liveness — `200` while the process is up. |
| GET | `/ready`, `/readyz`, `/ping` | Readiness — `200` when the adapter loads and reports ready, else `500`. (`/ping` is the SageMaker probe.) |
| POST | `/invocations` | Run inference. `200` with the formatted predictions; `400` on bad input; `413` over body limit; `429` when in-flight slots are exhausted; `500` on internal error. |
| GET | `/metrics` | Prometheus metrics (when `PROMETHEUS_ENABLED`). |
| GET | `/openapi.yaml`, `/docs` | Served only when `SWAGGER_ENABLED=true`. |

Content negotiation uses `Content-Type` / `Accept` (or the
`x-amzn-sagemaker-content-type` / `-accept` equivalents). Request bodies may be
`application/json`, JSON Lines (`application/jsonl`, `application/x-ndjson`,
etc.), or `text/csv`. Responses are JSON or `text/csv`. On a successful response,
`x-trace-id` / `x-span-id` headers are attached when an active span exists.

The transport contract is committed at `docs/openapi.yaml`; tensor semantics are
expected from a model manifest shipped with the artifact, not from OpenAPI.

## gRPC surface

An optional asyncio gRPC server (`src/onnx_model_serving/grpc/server.py`,
console script `onnx-model-serving-grpc`) implements the `InferenceService`
defined in `proto/onnx_serving_grpc/inference.proto`:

- `Live(LiveRequest) -> StatusReply`
- `Ready(ReadyRequest) -> StatusReply`
- `Predict(PredictRequest{payload, content_type, accept}) -> PredictReply{body, content_type, metadata}`

`Predict` runs the same parse/predict/format pipeline as `/invocations`,
returning `INVALID_ARGUMENT` on bad input and `INTERNAL` on runtime errors.
Generated stubs live under `src/onnx_serving_grpc/`; regenerate with
`make grpc-generate` and verify they are current with `make grpc-check`. Server
reflection is intentionally not enabled.

## Configuration

All configuration is via environment variables, read once at startup into a
frozen `Settings` model (`src/config.py`). Notable variables:

- Model: `SM_MODEL_DIR` (default `/opt/ml/model`), `MODEL_TYPE`, `MODEL_FILENAME`.
- Server: `PORT` (8080), `LOG_LEVEL`, `SERVICE_NAME`, `SERVICE_VERSION`,
  Gunicorn tuning (`GUNICORN_WORKERS`/`_THREADS`/`_TIMEOUT`/`_GRACEFUL_TIMEOUT`/`_KEEPALIVE`).
- Limits: `MAX_BODY_BYTES` (6 MiB), `MAX_RECORDS` (5000), `MAX_INFLIGHT` (16),
  `ACQUIRE_TIMEOUT_S` (0.25).
- IO: `INPUT_MODE` (tabular), `DEFAULT_CONTENT_TYPE`, `DEFAULT_ACCEPT`,
  `TABULAR_DTYPE`, `TABULAR_NUM_FEATURES`, `TABULAR_ID_COLUMNS`/`_FEATURE_COLUMNS`,
  `CSV_DELIMITER`/`_HAS_HEADER`/`_SKIP_BLANK_LINES`, `JSON_KEY_INSTANCES`,
  `JSONL_FEATURES_KEY`, `RETURN_PREDICTIONS_ONLY`, `JSON_OUTPUT_KEY`.
- ONNX: `ONNX_PROVIDERS`, `ONNX_INTRA_OP_THREADS`/`_INTER_OP_THREADS`,
  `ONNX_GRAPH_OPT_LEVEL`, `ONNX_INPUT_NAME`/`_OUTPUT_NAME`/`_OUTPUT_INDEX`,
  `ONNX_INPUT_MAP_JSON`, `ONNX_OUTPUT_MAP_JSON`, `ONNX_INPUT_DTYPE_MAP_JSON`,
  `ONNX_DYNAMIC_BATCH`.
- Observability: `PROMETHEUS_ENABLED`/`PROMETHEUS_PATH`, `SWAGGER_ENABLED`,
  `OTEL_ENABLED`, `OTEL_EXPORTER_OTLP_ENDPOINT`/`_HEADERS`/`_TIMEOUT`.

## Running locally

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
make env            # create .venv
make build-local    # uv sync --frozen --extra dev
```

Run the HTTP server (the default ONNX path; point `SM_MODEL_DIR` at a dir
containing `model.onnx`):

```bash
export SM_MODEL_DIR=/path/to/model-dir
export MODEL_TYPE=onnx
uv run --extra onnx python -m uvicorn serving:application --host 0.0.0.0 --port 8080
```

`serving.main()` (invoked by the container entrypoint) instead re-execs into
Gunicorn using `gunicorn_configuration`. Optional native backends require their
extra and a matching artifact:

```bash
export MODEL_TYPE=sklearn   MODEL_FILENAME=model.joblib   # uv sync --extra sklearn
export MODEL_TYPE=pytorch   MODEL_FILENAME=model.pt       # uv sync --extra torch
export MODEL_TYPE=tensorflow MODEL_FILENAME=model.keras   # uv sync --extra tensorflow
```

Run the gRPC server:

```bash
uv run --extra grpc onnx-model-serving-grpc --host 0.0.0.0 --port 50052
```

### Examples

`examples/` contains end-to-end demos that train a model, export it (to ONNX
where applicable), launch the real server in a subprocess, and validate
prediction parity through `/ready` and `/invocations`:

```bash
uv run --extra examples python -m examples.train_and_serve_logistic_regression
uv run --extra examples python -m examples.train_and_serve_random_forest
uv run --extra examples python -m examples.train_and_serve_neural_network
```

`examples/docker-compose.api-grpc.yml` brings up the HTTP and gRPC servers
together with a smoke client; `make smoke-api-grpc` drives the full compose
smoke test.

## Container

The service ships as a chain of incremental, layer-cached multi-stage images
(definitions in `container/`, orchestrated by the `Makefile`), so build tooling
and tests stay out of the runtime image:

1. `base` — Ubuntu base pinned by digest (`container/digests.env`), unprivileged user.
2. `base-builder` — adds Python/virtualenv tooling.
3. `uv-venv` — builds `/opt/venv` from `uv.lock` (`--frozen`, reproducible).
4. `builder` — reuses the venv and runs the test suite (build fails if tests fail).
5. `base-runner` — minimal runtime base with the entrypoint.
6. `runner` — final image: app code + Python runtime + copied `/opt/venv`, runs
   as `appuser` via `scripts/entrypoint.sh` (which execs `python /run/main.py`).

```bash
make build-images     # build the full chain (slow)
make build-runner     # build just the final runner image
make run-app          # run the runner container
```

`make` defaults to `podman` (`CONTAINER_COMMAND`); set it to `docker` if needed.
Example-model container images live under `container/examples/`.

## Development

```bash
make fmt            # black + ruff format
make lint           # fmt-check + ruff check + mypy (strict)
make test           # pytest (unit + integration + e2e)
make test-unit
make test-integration
make test-e2e
make coverage       # pytest with coverage (branch coverage; fail_under = 85)
make ci-grpc        # grpc-check + openapi-check + lint + gRPC/API parity tests
make openapi-check  # validate docs/openapi.yaml and check runtime drift
```

Tests are organised by scope under `tests/{unit_tests,integration_tests,e2e_tests}`
and include HTTP↔gRPC parity tests and Hypothesis property tests. Branch
coverage is enforced and must not regress. See `agents.md` for the full
engineering contract (determinism, IO isolation, contract stability, invariant-
driven testing).

## License

No `LICENSE` file is present at the repository root. Per workspace policy this
repo is **proprietary — All Rights Reserved** by default (it is internal ML
serving tooling, not a product or a public shared library). Add an explicit
`LICENSE` to confirm.
