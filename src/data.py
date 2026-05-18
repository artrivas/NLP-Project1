"""Dataset loading and tokenization utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(".cache/huggingface/datasets").resolve()))

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.utils import load_yaml_config


logger = logging.getLogger(__name__)

REQUIRED_CONFIG_FIELDS = {
    "dataset_name",
    "text_column",
    "label_column",
    "num_labels",
    "max_length",
    "train_split",
    "eval_split",
}


def validate_config(config: dict[str, Any]) -> None:
    """Validate the dataset configuration and raise clear errors if it is incomplete."""
    missing_fields = sorted(REQUIRED_CONFIG_FIELDS.difference(config))
    if missing_fields:
        raise ValueError(
            "YAML config is missing required field(s): "
            f"{', '.join(missing_fields)}"
        )

    if not isinstance(config["max_length"], int) or config["max_length"] <= 0:
        raise ValueError("YAML config field 'max_length' must be a positive integer.")

    if not isinstance(config["num_labels"], int) or config["num_labels"] <= 1:
        raise ValueError("YAML config field 'num_labels' must be an integer greater than 1.")


def load_dataset_from_config(config: dict[str, Any]) -> DatasetDict:
    """Load a Hugging Face dataset according to the provided configuration."""
    validate_config(config)
    dataset_name = config["dataset_name"]
    dataset_config = config.get("dataset_config")

    try:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config)
        else:
            dataset = load_dataset(dataset_name)
    except Exception as exc:
        message = (
            f"Failed to load dataset '{dataset_name}'"
            f"{f' with config {dataset_config!r}' if dataset_config else ''}. "
            "Check the dataset name, dataset config, network access, and installed "
            f"'datasets' version. Original error: {exc}"
        )
        logger.error(message)
        raise RuntimeError(message) from exc

    train_split = config["train_split"]
    eval_split = config["eval_split"]
    missing_splits = [split for split in (train_split, eval_split) if split not in dataset]
    if missing_splits:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing configured split(s): "
            f"{', '.join(missing_splits)}. Available splits: {', '.join(dataset.keys())}"
        )

    return dataset


def load_dataset_split_from_config(
    config: dict[str, Any],
    split: str,
    max_samples: int | None = None,
) -> Dataset:
    """Load one Hugging Face dataset split, optionally using split slicing."""
    validate_config(config)
    dataset_name = config["dataset_name"]
    dataset_config = config.get("dataset_config")
    split_expression = split if max_samples is None else f"{split}[:{max_samples}]"

    try:
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, split=split_expression)
        return load_dataset(dataset_name, split=split_expression)
    except Exception as exc:
        message = (
            f"Failed to load dataset '{dataset_name}' split '{split_expression}'"
            f"{f' with config {dataset_config!r}' if dataset_config else ''}. "
            "Check the dataset name, dataset config, split name, network access, "
            f"and installed 'datasets' version. Original error: {exc}"
        )
        logger.error(message)
        raise RuntimeError(message) from exc


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer from Hugging Face Transformers."""
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as exc:
        message = (
            f"Failed to load tokenizer '{model_name}'. Check the model name, network "
            f"access, and installed 'transformers' version. Original error: {exc}"
        )
        logger.error(message)
        raise RuntimeError(message) from exc


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    config: dict[str, Any],
) -> Dataset:
    """Tokenize a dataset split using the configured text column and max length."""
    text_column = config["text_column"]
    max_length = config["max_length"]

    if text_column not in dataset.column_names:
        raise ValueError(
            f"Configured text_column '{text_column}' was not found. "
            f"Available columns: {', '.join(dataset.column_names)}"
        )

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        try:
            return tokenizer(
                batch[text_column],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Tokenization failed for text_column '{text_column}'. "
                "Confirm that the column contains string-like text values."
            ) from exc

    columns_to_remove = [column for column in dataset.column_names if column != config["label_column"]]
    return dataset.map(tokenize_batch, batched=True, remove_columns=columns_to_remove)


def prepare_dataset(
    config_path: str,
    model_name: str = "distilbert-base-uncased",
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[Dataset, Dataset, PreTrainedTokenizerBase, dict[str, Any]]:
    """Load, tokenize, label-normalize, and format train/eval datasets for PyTorch."""
    config = load_yaml_config(config_path)
    validate_config(config)

    tokenizer = get_tokenizer(model_name)

    if max_train_samples is not None or max_eval_samples is not None:
        raw_train_dataset = load_dataset_split_from_config(
            config,
            config["train_split"],
            max_train_samples,
        )
        raw_eval_dataset = load_dataset_split_from_config(
            config,
            config["eval_split"],
            max_eval_samples,
        )
    else:
        raw_dataset = load_dataset_from_config(config)
        raw_train_dataset = raw_dataset[config["train_split"]]
        raw_eval_dataset = raw_dataset[config["eval_split"]]

    train_dataset = tokenize_dataset(raw_train_dataset, tokenizer, config)
    eval_dataset = tokenize_dataset(raw_eval_dataset, tokenizer, config)

    label_column = config["label_column"]
    if label_column not in train_dataset.column_names:
        raise ValueError(
            f"Configured label_column '{label_column}' was not found after tokenization. "
            f"Available columns: {', '.join(train_dataset.column_names)}"
        )

    if label_column != "labels":
        train_dataset = train_dataset.rename_column(label_column, "labels")
        eval_dataset = eval_dataset.rename_column(label_column, "labels")

    torch_columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_dataset.column_names:
        torch_columns.append("token_type_ids")

    train_dataset.set_format(type="torch", columns=torch_columns)
    eval_dataset.set_format(type="torch", columns=torch_columns)

    return train_dataset, eval_dataset, tokenizer, config
