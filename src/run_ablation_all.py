"""Run DistilBERT ablations across all project datasets."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from src.ablation import run_ablation


logger = logging.getLogger(__name__)

DATASET_CONFIGS = [
    "config/ag_news.yaml",
    "config/sst2.yaml",
    "config/yelp_review_full.yaml",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for all-dataset ablation runs."""
    parser = argparse.ArgumentParser(description="Run DistilBERT ablations on all datasets.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--debug-fast", action="store_true")
    mode.add_argument("--full-run", action="store_true")
    parser.add_argument("--ablation-config", default="config/ablation_distilbert.yaml")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--training-config", default="config/training/default_distilbert.yaml")
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run selected ablations for every dataset."""
    args = parse_args(argv)
    completed: list[str] = []

    for dataset_config in DATASET_CONFIGS:
        logger.info("Running DistilBERT ablations for %s", dataset_config)
        ablation_args = argparse.Namespace(
            dataset_config=dataset_config,
            ablation_config=args.ablation_config,
            model=args.model,
            config_name=args.config_name,
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
        run_ablation(ablation_args)
        completed.append(dataset_config)

    print("\nCompleted DistilBERT ablation runs:")
    for dataset_config in completed:
        print(f"- {dataset_config}")


if __name__ == "__main__":
    main()
