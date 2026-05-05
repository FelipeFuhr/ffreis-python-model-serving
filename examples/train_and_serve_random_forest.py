"""Train and serve a random forest ONNX model."""

from __future__ import annotations


def main() -> None:
    """Run random forest training and serving demo."""
    from examples.common import run_train_and_serve_demo

    result = run_train_and_serve_demo("random_forest")
    print(
        "PASS: random_forest example completed "
        f"(rows={result.request_row_count}, "
        f"max_abs_diff={result.max_abs_difference:.6f})."
    )


if __name__ == "__main__":
    main()
