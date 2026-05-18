"""Benchmark trained classifier checkpoints."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification

from src.data import prepare_dataset
from src.gpu_utils import empty_cuda_cache, get_gpu_name
from src.modeling import BertCustomClassifier, DistilBertCustomClassifier, count_total_parameters, count_trainable_parameters


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for checkpoint benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark a trained classifier checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output", default="results/benchmarks/benchmark_results.csv")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--debug-fast", action="store_true")
    mode.add_argument("--full-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return parser.parse_args(argv)


def load_checkpoint_model(model_path: str) -> torch.nn.Module:
    """Load a supported model checkpoint for inference benchmarking."""
    config = AutoConfig.from_pretrained(model_path)
    try:
        return AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception:
        if config.model_type == "distilbert":
            return DistilBertCustomClassifier.from_pretrained(model_path)
        if config.model_type == "bert":
            return BertCustomClassifier.from_pretrained(model_path)
        raise


def append_benchmark(row: dict[str, Any], output_path: str) -> None:
    """Append one benchmark row to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def benchmark_checkpoint(
    config_path: str,
    model_path: str,
    model_name: str | None = None,
    batch_size: int = 8,
    max_eval_samples: int = 16,
    warmup_batches: int = 1,
    measure_batches: int = 2,
    output_path: str | None = "results/benchmarks/benchmark_results.csv",
    model_type: str | None = None,
    configuration: str = "",
) -> dict[str, Any]:
    """Benchmark average inference latency and parameter counts for a checkpoint."""
    model_name_for_tokenizer = model_name or model_path
    _, eval_dataset, _, dataset_config = prepare_dataset(
        config_path=config_path,
        model_name=model_name_for_tokenizer,
        max_train_samples=1,
        max_eval_samples=max_eval_samples,
    )
    dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    model = load_checkpoint_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    empty_cuda_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    timings: list[float] = []
    sample_count = 0
    measured_batches = 0

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            batch = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(**batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if index >= warmup_batches:
                timings.append(elapsed)
                sample_count += next(iter(batch.values())).shape[0]
                measured_batches += 1
                if measured_batches >= measure_batches:
                    break

    total_time = sum(timings)
    latency_ms_per_batch = (total_time / measured_batches) * 1000.0 if measured_batches else None
    latency_ms_per_sample = (total_time / sample_count) * 1000.0 if sample_count else None
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None

    inferred_model_type = model_type
    if inferred_model_type is None:
        name = (model_name or model_path).lower()
        inferred_model_type = "distilbert" if "distilbert" in name else "bert" if "bert" in name else "unknown"

    dataset_name = dataset_config.get("dataset_config") or dataset_config["dataset_name"]
    row = {
        "dataset": dataset_name,
        "model_type": inferred_model_type,
        "model_name": model_name or model_path,
        "configuration": configuration,
        "model_path": model_path,
        "total_params": count_total_parameters(model),
        "trainable_params": count_trainable_parameters(model),
        "latency_ms_per_sample": latency_ms_per_sample,
        "latency_ms_per_batch": latency_ms_per_batch,
        "gpu_memory_peak_mb": peak_memory_mb,
        "batch_size": batch_size,
        "device_name": get_gpu_name(),
    }
    if output_path:
        append_benchmark(row, output_path)
    return row


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    row = benchmark_checkpoint(
        config_path=args.config,
        model_path=args.model_path,
        model_name=args.model_name,
        batch_size=args.batch_size or (4 if args.debug_fast else 32),
        max_eval_samples=args.max_eval_samples or (8 if args.debug_fast else 1024),
        warmup_batches=0 if args.debug_fast else 5,
        measure_batches=1 if args.debug_fast else 100,
        output_path=args.output,
    )
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
