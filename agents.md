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
- Do not silence errors with blanket ignores (`_ = err`, `//nolint`, `#nosec`, `@SuppressWarnings`, etc.) to work around a problem. Fix the root cause instead. Suppression is only acceptable when the error is genuinely irrelevant (e.g. a best-effort cleanup) and must include an inline comment explaining why.
- When encountering existing error-ignore patterns during any edit, remove the suppression and fix the underlying issue unless there is a clear documented reason for it.

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

# 14. Philosophy

This repository values:

- Strong invariants over superficial coverage.
- Stability over novelty.
- Determinism over convenience.
- Explicit contracts over implicit behavior.
- Long-term maintainability over short-term speed.

Agents must optimize for code that remains correct and evolvable over years.
