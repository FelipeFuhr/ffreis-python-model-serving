# Agent Context

**This repo:** `ffreis-python-model-serving` — Python FastAPI + gRPC server for ONNX
model inference. Serves the exact same contract as `ffreis-rust-onnx-model-serving`.

## Non-obvious facts

- **Proto files are canonical; stubs are checked in.** `src/onnx_serving_grpc/` contains
  generated gRPC stubs — do not edit them directly. Regenerate from
  `proto/onnx_serving_grpc/inference.proto`. Stubs are checked in so CI doesn't need
  protoc plugins.

- **HTTP and gRPC serve identical inference semantics.** One is not faster or more
  authoritative than the other. Both must remain in sync.

- **Contract must stay identical to the Rust serving repo.** If you change
  `proto/onnx_serving_grpc/inference.proto` or `docs/openapi.yaml`, the same change
  must be applied to `ml/ffreis-rust-onnx-model-serving`. The integration hub will
  detect divergence.

- **gRPC reflection is intentionally disabled.** Do not enable it.

- **Coverage minimum is 80%.** Enforced by pytest config.

- **OpenTelemetry W3C traceparent** — stay backend-agnostic. Do not add
  vendor-specific telemetry libraries.

- **`uv.lock` is required** — `uv sync --frozen` is used in CI and Docker builds.

## Structure

```
src/onnx_model_serving/     ← FastAPI app + HTTP handlers
src/onnx_serving_grpc/      ← generated gRPC stubs (DO NOT EDIT)
proto/                      ← canonical proto files
docs/openapi.yaml           ← HTTP contract
tests/                      ← unit, integration, e2e
```

## Build/test

```bash
uv sync && make test
make test-unit / test-integration / test-e2e
make test-grpc-parity          # called by integration-hub
make build-images
```

## Keeping this file current

- **If you discover a fact not reflected here:** add it before finishing your task.
- **If something here is wrong or outdated:** correct it in the same commit as the code change.
- **If you rename a file, command, or concept referenced here:** update the reference.
