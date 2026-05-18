"""Run DistilBERT baseline training across all Day 1 datasets."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from src.train import train_distilbert


logger = logging.getLogger(__name__)

DATASET_CONFIGS = [
    "config/ag_news.yaml",
    "config/sst2.yaml",
    "config/yelp_review_full.yaml",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for running all DistilBERT baselines."""
    parser = argparse.ArgumentParser(description="Run DistilBERT baselines on all datasets.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--debug-fast", action="store_true")
    mode.add_argument("--full-run", action="store_true")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--training-config", default="config/training/default_distilbert.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested DistilBERT baseline mode on all configured datasets."""
    args = parse_args(argv)
    completed = []

    for config_path in DATASET_CONFIGS:
        logger.info("Running DistilBERT baseline for %s", config_path)
        train_args = argparse.Namespace(
            config=config_path,
            model=args.model,
            output_dir=args.output_dir,
            training_config=args.training_config,
            debug_fast=args.debug_fast,
            full_run=args.full_run,
            max_train_samples=None,
            max_eval_samples=None,
            epochs=None,
            batch_size=None,
            learning_rate=None,
            seed=args.seed,
            fp16=args.fp16,
            gradient_accumulation_steps=None,
        )
        result = train_distilbert(train_args)
        completed.append(result["dataset_run_name"])

    print("\nCompleted DistilBERT baseline runs:")
    for dataset_name in completed:
        print(f"- {dataset_name}")


if __name__ == "__main__":
    main()
