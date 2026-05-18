"""CLI for DistilBERT-only ablation studies."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from transformers import Trainer, TrainingArguments

from src.data import prepare_dataset
from src.gpu_utils import empty_cuda_cache, get_cuda_memory_summary, get_gpu_name, log_gpu_info
from src.metrics import compute_metrics_for_trainer
from src.modeling import (
    DistilBertCustomClassifier,
    count_total_parameters,
    count_trainable_parameters,
    freeze_embeddings,
    freeze_entire_transformer,
    freeze_lower_transformer_layers,
)
from src.train import append_csv, dataset_run_name, write_training_logs
from src.utils import load_yaml_config, set_seed


logger = logging.getLogger(__name__)

DEFAULT_TRAINING_CONFIG = "config/training/default_distilbert.yaml"
DEFAULT_OUTPUT_DIR = "results"
DEBUG_MAX_TRAIN_SAMPLES = 8
DEBUG_MAX_EVAL_SAMPLES = 4
DEBUG_MAX_STEPS = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for DistilBERT ablation training."""
    parser = argparse.ArgumentParser(description="Run DistilBERT ablation configurations.")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--ablation-config", required=True)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--config-name", default=None, help="Run only one ablation config.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training-config", default=DEFAULT_TRAINING_CONFIG)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--debug-fast", action="store_true")
    mode.add_argument("--full-run", action="store_true")

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


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge shared training defaults, mode defaults, and CLI overrides."""
    training_config = load_yaml_config(args.training_config)
    run_config = {
        **training_config,
        "model_name": args.model,
        "output_dir": args.output_dir,
        "execution_mode": "debug-fast" if args.debug_fast else "full-run",
    }

    if args.debug_fast:
        run_config.update(
            {
                "max_train_samples": DEBUG_MAX_TRAIN_SAMPLES,
                "max_eval_samples": DEBUG_MAX_EVAL_SAMPLES,
                "num_train_epochs": 1.0,
                "train_batch_size": min(int(training_config.get("train_batch_size", 32)), 4),
                "eval_batch_size": min(int(training_config.get("eval_batch_size", 64)), 4),
                "max_steps": DEBUG_MAX_STEPS,
                "logging_steps": 1,
                "eval_steps": 1,
                "save_steps": 1,
                "save_total_limit": 1,
                "load_best_model_at_end": False,
                "save_strategy": "no",
                "eval_strategy": "no",
            }
        )
    else:
        run_config.update(
            {
                "max_steps": -1,
                "load_best_model_at_end": True,
                "save_strategy": "steps",
                "eval_strategy": "steps",
            }
        )

    overrides = {
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
    for key, value in overrides.items():
        if value is not None:
            run_config[key] = value

    if run_config.get("fp16", False) and not torch.cuda.is_available():
        logger.warning("fp16 requested but CUDA is unavailable; disabling fp16 for this run.")
        run_config["fp16"] = False

    return run_config


def select_ablation_configs(
    ablation_config: dict[str, Any],
    config_name: str | None,
) -> dict[str, dict[str, Any]]:
    """Return the selected ablation configuration or all configurations."""
    if config_name is None:
        return ablation_config

    if config_name not in ablation_config:
        available = ", ".join(sorted(ablation_config))
        raise ValueError(f"Unknown ablation config '{config_name}'. Available: {available}")

    return {config_name: ablation_config[config_name]}


def ensure_output_dirs(base_output_dir: str, dataset_name: str, config_name: str) -> dict[str, Path]:
    """Create and return ablation output directories."""
    base = Path(base_output_dir)
    dirs = {
        "metrics": base / "metrics",
        "logs": base / "logs",
        "checkpoint": base / "checkpoints" / "distilbert_ablation" / dataset_name / config_name,
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def apply_freezing(model: DistilBertCustomClassifier, config: dict[str, Any]) -> None:
    """Apply an ablation freeze mode to the model."""
    freeze_mode = config.get("freeze_mode", "none")
    if config.get("freeze_embeddings", False):
        freeze_embeddings(model)

    if freeze_mode == "none":
        return
    if freeze_mode == "all":
        freeze_entire_transformer(model)
        return
    if freeze_mode == "lower_layers":
        freeze_lower_transformer_layers(model, int(config.get("num_frozen_layers", 0)))
        return

    raise ValueError(f"Unsupported freeze_mode: {freeze_mode}")


def create_model(
    model_name: str,
    num_labels: int,
    ablation_config: dict[str, Any],
) -> DistilBertCustomClassifier:
    """Create a DistilBERT ablation model and apply freeze settings."""
    model = DistilBertCustomClassifier.from_pretrained(
        model_name,
        num_labels=num_labels,
        classifier_hidden_sizes=ablation_config.get("classifier_hidden_sizes", []),
        dropout=float(ablation_config.get("dropout", 0.2)),
    )
    apply_freezing(model, ablation_config)
    return model


def latency_ms_per_sample(eval_metrics: dict[str, Any], eval_size: int) -> float | None:
    """Compute evaluation latency in milliseconds per sample when runtime is available."""
    runtime = eval_metrics.get("eval_runtime")
    if runtime is None or eval_size <= 0:
        return None
    return float(runtime) * 1000.0 / eval_size


def run_one_ablation(
    args: argparse.Namespace,
    run_config: dict[str, Any],
    ablation_name: str,
    ablation_config: dict[str, Any],
) -> dict[str, Any]:
    """Run one ablation configuration and persist its artifacts."""
    set_seed(int(run_config["seed"]))
    empty_cuda_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_dataset, eval_dataset, tokenizer, dataset_config = prepare_dataset(
        config_path=args.dataset_config,
        model_name=run_config["model_name"],
        max_train_samples=run_config.get("max_train_samples"),
        max_eval_samples=run_config.get("max_eval_samples"),
    )
    dataset_name = dataset_run_name(dataset_config)
    output_dirs = ensure_output_dirs(run_config["output_dir"], dataset_name, ablation_name)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    model = create_model(
        run_config["model_name"],
        int(dataset_config["num_labels"]),
        ablation_config,
    )
    total_params = count_total_parameters(model)
    trainable_params = count_trainable_parameters(model)

    training_args = TrainingArguments(
        output_dir=str(output_dirs["checkpoint"]),
        run_name=f"distilbert-ablation-{dataset_name}-{ablation_name}-{run_config['execution_mode']}-{run_id}",
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
        "Starting %s ablation=%s dataset=%s train=%s eval=%s trainable_params=%s/%s",
        run_config["execution_mode"],
        ablation_name,
        dataset_name,
        len(train_dataset),
        len(eval_dataset),
        trainable_params,
        total_params,
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
        "ablation_config": ablation_name,
        "freeze_mode": ablation_config.get("freeze_mode", "none"),
        "classifier_hidden_sizes": json.dumps(ablation_config.get("classifier_hidden_sizes", [])),
        "dropout": float(ablation_config.get("dropout", 0.2)),
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
        "gpu_memory_peak_mb": peak_memory_mb,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    metrics_row = {
        **common_metadata,
        "train_loss": train_result.training_loss,
        **eval_metrics,
        "latency_ms_per_sample": latency_ms_per_sample(eval_metrics, len(eval_dataset)),
    }
    metrics_csv = output_dirs["metrics"] / "distilbert_ablation_metrics.csv"
    logs_csv = output_dirs["logs"] / "distilbert_ablation_training_logs.csv"
    append_csv(metrics_row, metrics_csv)
    write_training_logs(trainer.state.log_history, logs_csv, common_metadata)

    metrics_json = (
        output_dirs["metrics"]
        / f"distilbert_ablation_{dataset_name}_{ablation_name}_{run_config['execution_mode']}_{run_id}.json"
    )
    with metrics_json.open("w", encoding="utf-8") as file:
        json.dump(metrics_row, file, indent=2)

    print("\nDistilBERT ablation run complete")
    print(f"Dataset: {dataset_name}")
    print(f"Ablation config: {ablation_name}")
    print(f"Mode: {run_config['execution_mode']}")
    print(f"Trainable params: {trainable_params} / {total_params}")
    print(f"Eval f1_macro: {eval_metrics.get('eval_f1_macro')}")
    print(f"Eval accuracy: {eval_metrics.get('eval_accuracy')}")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Checkpoint dir: {output_dirs['checkpoint']}")

    return metrics_row


def run_ablation(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run selected ablation configurations."""
    run_config = build_run_config(args)
    ablation_config = load_yaml_config(args.ablation_config)
    selected_configs = select_ablation_configs(ablation_config, args.config_name)
    log_gpu_info()

    results = []
    for ablation_name, config in selected_configs.items():
        results.append(run_one_ablation(args, run_config, ablation_name, config))
    return results


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    run_ablation(args)


if __name__ == "__main__":
    main()
