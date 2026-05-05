"""Train and serve a neural network ONNX model using local HTTP flow."""

from __future__ import annotations


def main() -> None:
    """Run neural-network training and serving demo."""
    from examples.common import run_train_and_serve_demo

    result = run_train_and_serve_demo("neural_network")
    print(
        "PASS: neural_network example completed "
        f"(rows={result.request_row_count}, "
        f"max_abs_diff={result.max_abs_difference:.6f})."
    )


if __name__ == "__main__":
    main()
