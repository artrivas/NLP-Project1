"""CLI for BERT-family baseline fine-tuning."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from src.data import prepare_dataset
from src.gpu_utils import empty_cuda_cache, get_cuda_memory_summary, get_gpu_name, log_gpu_info
from src.metrics import compute_metrics_for_trainer
from src.utils import load_yaml_config, set_seed


logger = logging.getLogger(__name__)

DEFAULT_TRAINING_CONFIG = "config/training/default_distilbert.yaml"
DEFAULT_OUTPUT_DIR = "results"
DEBUG_MAX_TRAIN_SAMPLES = 8
DEBUG_MAX_EVAL_SAMPLES = 4
DEBUG_MAX_STEPS = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for DistilBERT training."""
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for text classification.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument("--model", default=None, help="Model name. Defaults to training YAML model_name.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Base output directory.")
    parser.add_argument("--training-config", default=DEFAULT_TRAINING_CONFIG)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--debug-fast", action="store_true", help="Run a small, quick training check.")
    mode.add_argument("--full-run", action="store_true", help="Run full baseline training.")

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    return parser.parse_args(argv)


def dataset_run_name(config: dict[str, Any]) -> str:
    """Return a stable run-safe dataset identifier."""
    if config.get("dataset_config"):
        return str(config["dataset_config"])
    return str(config["dataset_name"])


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge default training config, execution mode defaults, and CLI overrides."""
    training_config = load_yaml_config(args.training_config)
    model_name = args.model or training_config["model_name"]

    run_config = {
        **training_config,
        "model_name": model_name,
        "output_dir": args.output_dir,
        "execution_mode": "debug-fast" if args.debug_fast else "full-run",
    }

    if args.debug_fast:
        run_config.update(
            {
                "max_train_samples": DEBUG_MAX_TRAIN_SAMPLES,
                "max_eval_samples": DEBUG_MAX_EVAL_SAMPLES,
                "num_train_epochs": 1.0,
                "train_batch_size": min(int(training_config.get("train_batch_size", 32)), 8),
                "eval_batch_size": min(int(training_config.get("eval_batch_size", 64)), 16),
                "max_steps": DEBUG_MAX_STEPS,
                "logging_steps": 1,
                "eval_steps": 1,
                "save_steps": 1,
                "save_total_limit": 1,
                "eval_strategy": "no",
                "save_strategy": "no",
                "load_best_model_at_end": False,
            }
        )
    else:
        run_config["max_steps"] = -1
        run_config["eval_strategy"] = "steps"
        run_config["save_strategy"] = "steps"
        run_config["load_best_model_at_end"] = True

    override_map = {
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "num_train_epochs": args.epochs,
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "fp16": args.fp16,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    for key, value in override_map.items():
        if value is not None:
            run_config[key] = value

    if run_config.get("fp16", False) and not torch.cuda.is_available():
        logger.warning("fp16 requested but CUDA is unavailable; disabling fp16 for this run.")
        run_config["fp16"] = False

    return run_config


def model_family(model_name: str) -> str:
    """Return a stable model family identifier for output organization."""
    if "distilbert" in model_name:
        return "distilbert"
    if "bert" in model_name:
        return "bert"
    return model_name.replace("/", "_").replace("-", "_")


def ensure_output_dirs(base_output_dir: str, dataset_name: str, family: str = "distilbert") -> dict[str, Path]:
    """Create and return standard output directories for a training run."""
    base = Path(base_output_dir)
    dirs = {
        "metrics": base / "metrics",
        "logs": base / "logs",
        "checkpoint": base / "checkpoints" / family / dataset_name,
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def append_csv(row: dict[str, Any], path: Path) -> None:
    """Append one row to a CSV file, preserving existing rows."""
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def write_training_logs(
    log_history: list[dict[str, Any]],
    path: Path,
    run_metadata: dict[str, Any],
) -> None:
    """Append Trainer log history rows to the consolidated training log CSV."""
    if not log_history:
        return

    rows = []
    for entry in log_history:
        row = {**run_metadata, **entry}
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def train_distilbert(args: argparse.Namespace) -> dict[str, Any]:
    """Run one BERT-family fine-tuning job and persist metrics, logs, and checkpoints."""
    run_config = build_run_config(args)
    set_seed(int(run_config["seed"]))
    log_gpu_info()
    empty_cuda_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_dataset, eval_dataset, tokenizer, dataset_config = prepare_dataset(
        config_path=args.config,
        model_name=run_config["model_name"],
        max_train_samples=run_config.get("max_train_samples"),
        max_eval_samples=run_config.get("max_eval_samples"),
    )
    dataset_name = dataset_run_name(dataset_config)
    family = model_family(run_config["model_name"])
    output_dirs = ensure_output_dirs(run_config["output_dir"], dataset_name, family)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    model = AutoModelForSequenceClassification.from_pretrained(
        run_config["model_name"],
        num_labels=int(dataset_config["num_labels"]),
    )

    training_args = TrainingArguments(
        output_dir=str(output_dirs["checkpoint"]),
        run_name=f"{family}-{dataset_name}-{run_config['execution_mode']}-{run_id}",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=int(run_config["train_batch_size"]),
        per_device_eval_batch_size=int(run_config["eval_batch_size"]),
        gradient_accumulation_steps=int(run_config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(run_config["learning_rate"]),
        weight_decay=float(run_config["weight_decay"]),
        num_train_epochs=float(run_config["num_train_epochs"]),
        max_steps=int(run_config.get("max_steps", -1)),
        warmup_ratio=float(run_config.get("warmup_ratio", 0.0)),
        logging_strategy="steps",
        logging_steps=int(run_config["logging_steps"]),
        logging_first_step=True,
        eval_strategy=str(run_config.get("eval_strategy", "steps")),
        eval_steps=int(run_config["eval_steps"]),
        save_strategy=str(run_config.get("save_strategy", "steps")),
        save_steps=int(run_config["save_steps"]),
        save_total_limit=int(run_config.get("save_total_limit", 2)),
        load_best_model_at_end=bool(run_config.get("load_best_model_at_end", True)),
        metric_for_best_model=str(run_config.get("metric_for_best_model", "f1_macro")),
        greater_is_better=True,
        fp16=bool(run_config.get("fp16", False)),
        report_to="none",
        seed=int(run_config["seed"]),
        data_seed=int(run_config["seed"]),
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_for_trainer,
    )

    logger.info(
        "Starting %s %s run for %s: train=%s eval=%s",
        run_config["execution_mode"],
        family,
        dataset_name,
        len(train_dataset),
        len(eval_dataset),
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dirs["checkpoint"] / "best_model"))
    tokenizer.save_pretrained(str(output_dirs["checkpoint"] / "best_model"))

    peak_memory_mb = None
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        logger.info("Peak CUDA memory allocated: %.2f MB", peak_memory_mb)
        logger.info("Final CUDA memory summary:\n%s", get_cuda_memory_summary())

    common_metadata = {
        "run_id": run_id,
        "timestamp_utc": run_id,
        "execution_mode": run_config["execution_mode"],
        "dataset_name": dataset_config["dataset_name"],
        "dataset_config": dataset_config.get("dataset_config", ""),
        "dataset_run_name": dataset_name,
        "model_name": run_config["model_name"],
        "seed": int(run_config["seed"]),
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "num_labels": int(dataset_config["num_labels"]),
        "max_length": int(dataset_config["max_length"]),
        "learning_rate": float(run_config["learning_rate"]),
        "epochs": float(run_config["num_train_epochs"]),
        "train_batch_size": int(run_config["train_batch_size"]),
        "eval_batch_size": int(run_config["eval_batch_size"]),
        "gradient_accumulation_steps": int(run_config.get("gradient_accumulation_steps", 1)),
        "fp16": bool(run_config.get("fp16", False)),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": get_gpu_name(),
        "peak_cuda_memory_mb": peak_memory_mb,
        "model_type": family,
    }

    metrics_row = {
        **common_metadata,
        "train_loss": train_result.training_loss,
        **eval_metrics,
    }
    if family == "bert":
        metrics_csv = output_dirs["metrics"] / "bert_metrics.csv"
        logs_csv = output_dirs["logs"] / "bert_training_logs.csv"
    else:
        metrics_csv = output_dirs["metrics"] / "distilbert_baseline_metrics.csv"
        logs_csv = output_dirs["logs"] / "distilbert_baseline_training_logs.csv"
    append_csv(metrics_row, metrics_csv)
    write_training_logs(trainer.state.log_history, logs_csv, common_metadata)

    metrics_json = output_dirs["metrics"] / f"{family}_{dataset_name}_{run_config['execution_mode']}_{run_id}.json"
    with metrics_json.open("w", encoding="utf-8") as file:
        json.dump(metrics_row, file, indent=2)

    print(f"\n{family} baseline run complete")
    print(f"Dataset: {dataset_name}")
    print(f"Mode: {run_config['execution_mode']}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Eval size: {len(eval_dataset)}")
    print(f"Train loss: {train_result.training_loss:.6f}")
    print(f"Eval f1_macro: {eval_metrics.get('eval_f1_macro')}")
    print(f"Eval accuracy: {eval_metrics.get('eval_accuracy')}")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Training logs CSV: {logs_csv}")
    print(f"Checkpoint dir: {output_dirs['checkpoint']}")

    return metrics_row


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    train_distilbert(args)


if __name__ == "__main__":
    main()
