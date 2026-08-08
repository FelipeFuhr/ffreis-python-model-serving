# agents.md
Engineering Contract for Automated Agents

This repository is treated as production-grade infrastructure by default.

Agents must optimize for:
- Determinism
- Reproducibility
- Strong invariants
- Backward compatibility
- Explicit configuration
- Minimal architectural drift
- Long-term maintainability

If tradeoffs are required:

Correctness > Explicitness > Stability > Performance > Cleverness

---

# 0. Prime Directive

Agents must:
1. Follow existing toolchain and patterns.
2. Avoid introducing architectural drift.
3. Add tests for all new behavior.
4. Not reduce coverage.
5. Preserve backward compatibility unless explicitly instructed.
6. Provide verification steps with every change.

If unsure, choose the safest, most explicit, most testable option.

---

# 1. Architecture Rules

## 1.1 No Drift

Do not:
- Introduce new architectural patterns.
- Collapse separation between core logic and IO.
- Introduce global state.
- Add hidden side effects.

Core logic must:
- Be deterministic.
- Be testable in isolation.
- Not perform IO directly.

IO must be isolated behind adapters.

---

## 2. Tooling Discipline

Respect existing tooling:
- If repo uses uv -> continue using uv.
- If repo uses ruff -> continue using ruff.
- If repo uses betterproto2 -> do not replace schema layer.
- If repo uses Typer -> do not introduce Click/Argparse.
- If Rust lives in /app, respect workspace boundaries.

Do not introduce new toolchains without explicit instruction.

---

# 3. API & Contract Stability

Public APIs are stable by default:
- CLI flags
- Protobuf schemas
- ONNX export interfaces
- Config formats
- Batch input/output contracts

Agents must:
- Preserve compatibility.
- Avoid breaking changes.
- Update documentation when behavior changes.

If breaking change is unavoidable:
- State explicitly.
- Provide migration guidance.
- Add compatibility tests.

---

# 4. Testing Requirements (Strict)

All new behavior must include tests.

## 4.1 Testing Pyramid

### Unit Tests
- Deterministic.
- No network.
- No real external services.
- No reliance on execution order.

### Contract Tests
Required when applicable:
- Protobuf serialization round-trip.
- ONNX input/output shape validation.
- Dataset schema validation.
- Env var validation.

### Integration Tests
- Filesystem boundaries.
- Spark local session (if applicable).
- CLI invocation.
- Adapter boundaries.

### Optional Smoke Tests
- Minimal fixture dataset through full path.

---

## 4.2 Coverage Policy

- Coverage must be measured.
- Branch coverage must be enabled.
- Coverage must not decrease.
- New/changed lines must be covered.
- Error paths must be tested.

Example:

pytest --cov --cov-branch

---

# 5. Invariant-Driven Development

For transformation or validation logic, agents must prefer invariant tests over example-only tests.

Examples of invariants:
- Filtering never increases dataset size.
- Sampling never exceeds original size.
- Serialization round-trip preserves equality.
- ONNX exported model preserves input/output shapes.
- Evaluation does not mutate inputs.

Use property-based testing (e.g., Hypothesis) when appropriate.

Critical modules (metrics, drift, validation, dataset resolution) should be suitable for mutation testing.

---

# 6. Determinism Rules

- Seed randomness explicitly.
- Avoid timezone-dependent behavior.
- Avoid locale dependence.
- Validate required environment variables at startup.
- Do not rely on implicit environment state.

Tests must:
- Be deterministic.
- Avoid flaky behavior.
- Not depend on execution order.

---

# 7. Performance Discipline

Agents must not:
- Introduce quadratic behavior accidentally.
- Load entire datasets unnecessarily.
- Use Spark .collect() in core paths without justification.

If modifying critical logic:
- Consider performance implications.
- Avoid silent memory expansion.

---

# 8. Data & ML Guardrails (When Applicable)

## Spark / Pandas Dual Support
- Maintain consistent interface.
- Hide engine-specific logic behind adapters.
- Do not mix Spark and pandas logic inside core business functions.

## Protobuf
- .proto files are canonical.
- No duplicate schema definitions.
- Preserve backward compatibility.

## ONNX / Model Export
- Validate input shape and dtype.
- Validate output shape.
- Add inference test for exported model.

---

# 9. Containers & Supply Chain

- Prefer pinned base images (by digest).
- Prefer non-root runtime.
- Avoid unverified install scripts.
- Ensure proper SIGTERM/SIGINT handling.
- Fail fast on misconfiguration.

CI must:
- Run lint.
- Run tests.
- Enforce coverage.
- Run security scans if configured.
- Avoid excessive permissions.

---

# 10. Error Handling

- Do not swallow exceptions silently.
- Use explicit exception types.
- Provide actionable error messages.
- Library code must not call sys.exit().

---

# 11. Documentation

When adding features:
- Update README.
- Document new Make targets.
- Document new env vars.
- Provide usage examples.

---

# 12. Agent Output Format

When generating changes, agents must:

1. Provide modified file tree.
2. Provide full file contents (not partial diffs unless requested).
3. Provide verification steps.
4. Provide short explanation:
   - What changed
   - Why
   - How to verify
   - Backward compatibility impact

---

# 13. Definition of Done

Before finalizing changes:

- [ ] Tests added
- [ ] Coverage not reduced
- [ ] Branch coverage enabled
- [ ] Invariants considered
- [ ] Lint passes
- [ ] Types pass (if configured)
- [ ] No secrets introduced
- [ ] Dependencies justified
- [ ] Docs updated
- [ ] Backward compatibility considered
- [ ] Determinism preserved
- [ ] Performance impact considered

---

# 14. Registry-backed Startup

The serving process supports resolving model artifacts from `ffreis-ml-registry`
at startup. This is an optional extension — the legacy `SM_MODEL_DIR` path is
unchanged when the registry env vars are absent.

## Env vars

| Variable                  | Required | Description                                                          |
|---------------------------|----------|----------------------------------------------------------------------|
| `MODEL_REGISTRY_BACKEND`  | together | Backend key: `sqlite`, `local` (alias), `dynamodb`, `postgres`.     |
| `MODEL_REGISTRY_URI`      | together | Connection string/path for the backend (see table below).            |
| `MODEL_NAME`              | yes      | Logical model name to resolve (must have a `production`-stage version). |

`MODEL_REGISTRY_BACKEND` and `MODEL_REGISTRY_URI` must be set together or not at all.

## URI semantics per backend

| Backend     | URI format                                      |
|-------------|------------------------------------------------|
| `sqlite`    | Absolute path to SQLite file, e.g. `/models/registry.db` |
| `dynamodb`  | DynamoDB table name, e.g. `ml-model-registry`  |
| `postgres`  | DSN string, e.g. `postgresql://user:pw@host/db` |

## `onnx_uri` artifact semantics

The `onnx_uri` field on the registered `ModelVersion` is the artifact address:

- Bare local path (`/opt/ml/models/iris.onnx`) — copied into a temp dir.
- `file://` URI — stripped and treated as a local path.
- `s3://` — reserved; raises `NotImplementedError` (future extension).

## Optional dependency

Install the registry extra before using registry-backed startup:

```bash
uv sync --extra registry
```

The `ml_registry` import is guarded behind `try/except ImportError` with a
clear error message if the package is not installed.

## Example

```bash
MODEL_REGISTRY_BACKEND=sqlite \
MODEL_REGISTRY_URI=/opt/ml/registry.db \
MODEL_NAME=my-classifier \
MODEL_TYPE=onnx \
uv run python main.py
```

## Implementation files

- `src/registry_pull.py` — resolution + materialisation logic (no IO in core).
- `src/config.py` — `model_registry_backend`, `model_registry_uri`, `model_name` fields.
- `src/base_adapter.py` — `load_adapter` calls `pull_model_from_registry` before
  adapter auto-detection; shadows `model_dir` with the registry-pulled temp dir.
- `tests/integration_tests/test_registry_pull.py` — integration + error-path tests.

---

# 15. Mutation Testing (mutmut) Gotchas

`pyproject.toml` pins `mutmut>=2.4,<3` (v3 dropped `--paths-to-mutate`, which
`.github/workflows/mutation.yml` and `make mutation` both invoke; an unpinned
`>=2.0.0`-style range silently resolves to 3.x). Two Python-3.13-specific traps
found while wiring this up — both reproduced locally, neither is code-quality
work needed on this repo, just tool-compat awareness:

- **`src/value_types.py` is excluded from mutation scanning** via
  `[tool.mutmut] paths_to_exclude` in `pyproject.toml`. It uses PEP 695
  `type X = ...` alias statements; mutmut's parser (`parso` 0.8.7) cannot parse
  that syntax yet and aborts the *entire* run with "Failed while creating
  mutations" if the file is in scope. Pure type aliases have no runtime
  behavior to mutate anyway, so excluding it costs nothing. If a future
  `parso`/`mutmut` release adds PEP 695 support, this exclusion can be dropped.
- **`mutmut results` and `mutmut html` crash** under Python 3.13 + mutmut 2.5.1
  (`TypeError: 'QueryResultIterator' object is not iterable` — a Pony ORM
  0.7.19 incompatibility in `Mutant.select()`). `mutmut run` itself is
  unaffected (writes results straight to the `.mutmut-cache` SQLite file). Both
  `make mutation` and the CI gate (`python-mutation.yml` in
  `ffreis-workflows-python`) read the score directly from that SQLite file
  instead of calling the broken CLI commands — do not "fix" this by switching
  back to `mutmut results` without re-testing on the pinned Python version.
- **If you find `.mutmut-cache` or a `*.bak` file next to a source file with an
  unexplained diff**, it's a mutation left applied from an interrupted
  `mutmut run` (e.g. a killed/timed-out process caught mid-mutant, before it
  restored the original). Restore from `.bak`, delete both, and diff against
  `git show HEAD:<path>` to confirm a clean restore before trusting local test
  results.

---

# 16. Philosophy

This repository values:

- Strong invariants over superficial coverage.
- Stability over novelty.
- Determinism over convenience.
- Explicit contracts over implicit behavior.
- Long-term maintainability over short-term speed.

Agents must optimize for code that remains correct and evolvable over years.
