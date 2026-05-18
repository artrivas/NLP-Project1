"""Select the best DistilBERT ablation configuration per dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


DEFAULT_INPUT = "results/metrics/distilbert_ablation_metrics.csv"
DEFAULT_OUTPUT = "results/metrics/best_distilbert_ablation.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for best-ablation selection."""
    parser = argparse.ArgumentParser(description="Select best DistilBERT ablation per dataset.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def normalize_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Create unprefixed metric aliases from Trainer eval metric columns."""
    aliases = {
        "accuracy": "eval_accuracy",
        "precision_macro": "eval_precision_macro",
        "recall_macro": "eval_recall_macro",
        "f1_macro": "eval_f1_macro",
    }
    for alias, source in aliases.items():
        if alias not in frame.columns and source in frame.columns:
            frame[alias] = frame[source]
    return frame


def select_best(input_path: str = DEFAULT_INPUT, output_path: str = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Select best ablation config per dataset and save the result CSV."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Ablation metrics file not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Ablation metrics file is empty: {path}")

    frame = normalize_metric_columns(frame)
    required = {"dataset_run_name", "ablation_config", "f1_macro", "accuracy", "trainable_params"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required column(s) in ablation metrics: {', '.join(missing)}")

    if "latency_ms_per_sample" not in frame.columns:
        frame["latency_ms_per_sample"] = float("inf")
    if "total_params" not in frame.columns:
        frame["total_params"] = pd.NA
    if "gpu_memory_peak_mb" not in frame.columns:
        frame["gpu_memory_peak_mb"] = pd.NA

    sortable = frame.copy()
    sortable["latency_sort"] = sortable["latency_ms_per_sample"].fillna(float("inf"))
    sortable = sortable.sort_values(
        by=["dataset_run_name", "f1_macro", "accuracy", "latency_sort", "trainable_params"],
        ascending=[True, False, False, True, True],
    )
    best = sortable.groupby("dataset_run_name", as_index=False).head(1)

    output = pd.DataFrame(
        {
            "dataset": best["dataset_run_name"],
            "best_config": best["ablation_config"],
            "accuracy": best["accuracy"],
            "precision_macro": best.get("precision_macro"),
            "recall_macro": best.get("recall_macro"),
            "f1_macro": best["f1_macro"],
            "total_params": best["total_params"],
            "trainable_params": best["trainable_params"],
            "gpu_memory_peak_mb": best["gpu_memory_peak_mb"],
            "latency_ms_per_sample": best["latency_ms_per_sample"],
        }
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    output = select_best(args.input, args.output)
    print(f"Saved best ablation selections to {args.output}")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
