"""CLI smoke test for the Day 1 data preprocessing pipeline."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from src.data import prepare_dataset


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the data smoke test."""
    parser = argparse.ArgumentParser(description="Smoke test dataset loading and tokenization.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument(
        "--model",
        default="distilbert-base-uncased",
        help="Tokenizer model name. Defaults to distilbert-base-uncased.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the dataset loading and tokenization smoke test."""
    args = parse_args(argv)

    try:
        train_dataset, eval_dataset, tokenizer, config = prepare_dataset(
            config_path=args.config,
            model_name=args.model,
        )
    except Exception as exc:
        logger.error("Smoke test failed: %s", exc)
        raise SystemExit(1) from exc

    sample = train_dataset[0]
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    print(f"Dataset name: {config['dataset_name']}")
    if config.get("dataset_config"):
        print(f"Dataset config: {config['dataset_config']}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Eval size: {len(eval_dataset)}")
    print(f"Text column: {config['text_column']}")
    print(f"Label column: {config['label_column']} -> labels")
    print(f"Num labels: {config['num_labels']}")
    print(f"Tokenizer name: {tokenizer.name_or_path}")
    print(f"Tokenized sample keys: {sorted(sample.keys())}")
    print(f"input_ids length: {len(input_ids)}")
    print(f"attention_mask length: {len(attention_mask)}")


if __name__ == "__main__":
    main()
