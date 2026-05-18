"""Final comparison between best DistilBERT ablation and BERT-base."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.benchmark import benchmark_checkpoint
from src.train import train_distilbert


logger = logging.getLogger(__name__)

DATASET_CONFIGS = {
    "ag_news": "config/ag_news.yaml",
    "sst2": "config/sst2.yaml",
    "yelp_review_full": "config/yelp_review_full.yaml",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-comparison CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare best DistilBERT config with BERT-base.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--debug-fast", action="store_true")
    mode.add_argument("--full-run", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--best-distilbert", default="results/metrics/best_distilbert_ablation.csv")
    parser.add_argument("--ablation-metrics", default="results/metrics/distilbert_ablation_metrics.csv")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    return parser.parse_args(argv)


def load_best_distilbert(args: argparse.Namespace) -> pd.DataFrame:
    """Load best DistilBERT configs, with debug fallback when missing."""
    path = Path(args.best_distilbert)
    if path.exists():
        return pd.read_csv(path)
    if args.full_run:
        raise FileNotFoundError(f"Full-run comparison requires {path}")

    logger.warning("%s not found; using debug fallback DistilBERT baseline configs.", path)
    return pd.DataFrame(
        {
            "dataset": list(DATASET_CONFIGS),
            "best_config": ["baseline"] * len(DATASET_CONFIGS),
        }
    )


def latest_distilbert_metric(ablation_metrics_path: str, dataset: str, config_name: str) -> dict[str, Any]:
    """Return the latest matching DistilBERT ablation metrics row when available."""
    path = Path(ablation_metrics_path)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    subset = frame[(frame["dataset_run_name"] == dataset) & (frame["ablation_config"] == config_name)]
    if subset.empty:
        return {}
    return subset.sort_values("timestamp_utc").iloc[-1].to_dict()


def latest_bert_metric(output_dir: str, dataset: str) -> dict[str, Any]:
    """Return the latest matching BERT metrics row when available."""
    path = Path(output_dir) / "metrics" / "bert_metrics.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "dataset_run_name" not in frame.columns:
        return {}
    subset = frame[frame["dataset_run_name"] == dataset]
    if subset.empty:
        return {}
    sort_col = "timestamp_utc" if "timestamp_utc" in subset.columns else "run_id"
    return subset.sort_values(sort_col).iloc[-1].to_dict()


def metric_value(row: dict[str, Any], name: str) -> Any:
    """Read a metric from either unprefixed or Trainer-prefixed columns."""
    return row.get(name, row.get(f"eval_{name}"))


def comparison_row(
    dataset: str,
    model_type: str,
    model_name: str,
    configuration: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build one performance comparison row."""
    return {
        "dataset": dataset,
        "model_type": model_type,
        "model_name": model_name,
        "configuration": configuration,
        "accuracy": metric_value(metrics, "accuracy"),
        "precision_macro": metric_value(metrics, "precision_macro"),
        "recall_macro": metric_value(metrics, "recall_macro"),
        "f1_macro": metric_value(metrics, "f1_macro"),
        "precision_weighted": metric_value(metrics, "precision_weighted"),
        "recall_weighted": metric_value(metrics, "recall_weighted"),
        "f1_weighted": metric_value(metrics, "f1_weighted"),
    }


def efficiency_row(
    dataset: str,
    model_type: str,
    model_name: str,
    configuration: str,
    metrics: dict[str, Any],
    benchmark: dict[str, Any] | None = None,
    batch_size: int | None = None,
    fp16: bool | None = None,
) -> dict[str, Any]:
    """Build one efficiency comparison row."""
    benchmark = benchmark or {}
    return {
        "dataset": dataset,
        "model_type": model_type,
        "model_name": model_name,
        "configuration": configuration,
        "total_params": benchmark.get("total_params", metrics.get("total_params")),
        "trainable_params": benchmark.get("trainable_params", metrics.get("trainable_params")),
        "latency_ms_per_sample": benchmark.get("latency_ms_per_sample", metrics.get("latency_ms_per_sample")),
        "latency_ms_per_batch": benchmark.get("latency_ms_per_batch", metrics.get("latency_ms_per_batch")),
        "gpu_memory_peak_mb": benchmark.get("gpu_memory_peak_mb", metrics.get("gpu_memory_peak_mb")),
        "batch_size": benchmark.get("batch_size", batch_size),
        "fp16": fp16,
        "device_name": benchmark.get("device_name", metrics.get("gpu_name")),
    }


def train_bert_for_dataset(args: argparse.Namespace, dataset_config: str) -> dict[str, Any]:
    """Train BERT-base using the shared training CLI implementation."""
    train_args = argparse.Namespace(
        config=dataset_config,
        model="bert-base-uncased",
        output_dir=args.output_dir,
        training_config="config/training/default_distilbert.yaml",
        debug_fast=args.debug_fast,
        full_run=args.full_run,
        max_train_samples=4 if args.debug_fast else None,
        max_eval_samples=4 if args.debug_fast else None,
        epochs=1.0 if args.debug_fast else None,
        batch_size=2 if args.debug_fast else None,
        learning_rate=None,
        seed=args.seed,
        fp16=args.fp16,
        gradient_accumulation_steps=None,
    )
    return train_distilbert(train_args)


def main(argv: Sequence[str] | None = None) -> None:
    """Run final comparison and save performance/efficiency tables."""
    args = parse_args(argv)
    best_distilbert = load_best_distilbert(args)

    comparison_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []

    for _, best_row in best_distilbert.iterrows():
        dataset = str(best_row["dataset"])
        if dataset not in DATASET_CONFIGS:
            logger.warning("Skipping unknown dataset in best table: %s", dataset)
            continue

        config_name = str(best_row["best_config"])
        distilbert_metrics = latest_distilbert_metric(args.ablation_metrics, dataset, config_name)
        if not distilbert_metrics:
            distilbert_metrics = best_row.to_dict()

        comparison_rows.append(
            comparison_row(
                dataset,
                "distilbert",
                "distilbert-base-uncased",
                config_name,
                distilbert_metrics,
            )
        )
        efficiency_rows.append(
            efficiency_row(
                dataset,
                "distilbert",
                "distilbert-base-uncased",
                config_name,
                distilbert_metrics,
                batch_size=distilbert_metrics.get("eval_batch_size"),
                fp16=distilbert_metrics.get("fp16"),
            )
        )

        bert_checkpoint = Path(args.output_dir) / "checkpoints" / "bert" / dataset / "best_model"
        bert_metrics = latest_bert_metric(args.output_dir, dataset)
        if not bert_metrics or not bert_checkpoint.exists():
            bert_metrics = train_bert_for_dataset(args, DATASET_CONFIGS[dataset])
            bert_checkpoint = Path(args.output_dir) / "checkpoints" / "bert" / dataset / "best_model"
        benchmark = benchmark_checkpoint(
            config_path=DATASET_CONFIGS[dataset],
            model_path=str(bert_checkpoint),
            model_name="bert-base-uncased",
            batch_size=2 if args.debug_fast else int(bert_metrics.get("eval_batch_size", 32)),
            max_eval_samples=4 if args.debug_fast else 2048,
            warmup_batches=0 if args.debug_fast else 5,
            measure_batches=1 if args.debug_fast else 100,
            output_path=str(Path(args.output_dir) / "benchmarks" / "benchmark_results.csv"),
            model_type="bert",
            configuration="bert_base",
        )

        comparison_rows.append(
            comparison_row(dataset, "bert", "bert-base-uncased", "bert_base", bert_metrics)
        )
        efficiency_rows.append(
            efficiency_row(
                dataset,
                "bert",
                "bert-base-uncased",
                "bert_base",
                bert_metrics,
                benchmark=benchmark,
                batch_size=bert_metrics.get("eval_batch_size"),
                fp16=bert_metrics.get("fp16"),
            )
        )

    metrics_dir = Path(args.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = metrics_dir / "final_comparison_metrics.csv"
    efficiency_path = metrics_dir / "final_efficiency_metrics.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    pd.DataFrame(efficiency_rows).to_csv(efficiency_path, index=False)

    print(f"Saved final comparison metrics to {comparison_path}")
    print(f"Saved final efficiency metrics to {efficiency_path}")
    print(pd.DataFrame(comparison_rows).to_string(index=False))


if __name__ == "__main__":
    main()
