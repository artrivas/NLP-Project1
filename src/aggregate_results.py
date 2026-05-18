"""Aggregate saved experiment outputs into report-ready tables.

Day 5 is intentionally read-only with respect to model training. This module
only reads CSV files produced by previous days and writes compact summary
tables for documentation and reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"


def warn(message: str) -> None:
    """Print a standardized warning without interrupting summary generation."""
    print(f"Warning: {message}")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read a CSV file if it exists, otherwise return an empty DataFrame."""
    if not path.exists():
        warn(f"missing file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive reporting path
        warn(f"could not read {path}: {exc}")
        return pd.DataFrame()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase snake-case column names."""
    if df.empty:
        return df
    normalized = df.copy()
    normalized.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in normalized.columns
    ]
    return normalized


def dataset_column(df: pd.DataFrame) -> pd.Series:
    """Return the most useful dataset identifier available in a result frame."""
    if "dataset" in df.columns:
        return df["dataset"]
    if "dataset_run_name" in df.columns:
        return df["dataset_run_name"]
    if "dataset_config" in df.columns:
        return df["dataset_config"].fillna(df.get("dataset_name", ""))
    if "dataset_name" in df.columns:
        return df["dataset_name"]
    return pd.Series([""] * len(df), index=df.index)


def save_table(df: pd.DataFrame, name: str) -> list[Path]:
    """Save a table as CSV and Markdown, returning generated paths."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / f"{name}.csv"
    md_path = TABLES_DIR / f"{name}.md"

    df.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(df), encoding="utf-8")
    return [csv_path, md_path]


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a simple GitHub-flavored Markdown table."""
    if df.empty:
        return "_No data available._\n"

    rendered = df.copy()
    rendered = rendered.fillna("")
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6g}"
            )

    headers = [str(column) for column in rendered.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in rendered.iterrows():
        values = [str(row[column]) for column in rendered.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def latest_rows(df: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    """Keep the latest row per key set using timestamp/run columns when present."""
    if df.empty:
        return df
    sort_columns = [column for column in ["timestamp_utc", "run_id"] if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)
    key_list = [key for key in keys if key in df.columns]
    if not key_list:
        return df
    return df.drop_duplicates(key_list, keep="last")


def build_final_performance_table() -> pd.DataFrame:
    """Build the final DistilBERT vs BERT performance table."""
    final = normalize_columns(read_csv_if_exists(METRICS_DIR / "final_comparison_metrics.csv"))
    if final.empty:
        warn("final performance table requires final_comparison_metrics.csv")
        return pd.DataFrame()

    columns = [
        "dataset",
        "model_type",
        "model_name",
        "configuration",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    return final[[column for column in columns if column in final.columns]]


def build_final_efficiency_table() -> pd.DataFrame:
    """Build the final efficiency table for report use."""
    efficiency = normalize_columns(read_csv_if_exists(METRICS_DIR / "final_efficiency_metrics.csv"))
    if efficiency.empty:
        warn("final efficiency table requires final_efficiency_metrics.csv")
        return pd.DataFrame()

    columns = [
        "dataset",
        "model_type",
        "model_name",
        "configuration",
        "total_params",
        "trainable_params",
        "latency_ms_per_sample",
        "gpu_memory_peak_mb",
        "batch_size",
        "device_name",
    ]
    return efficiency[[column for column in columns if column in efficiency.columns]]


def build_ablation_summary_table() -> pd.DataFrame:
    """Build a compact summary of DistilBERT ablation results."""
    ablation = normalize_columns(read_csv_if_exists(METRICS_DIR / "distilbert_ablation_metrics.csv"))
    if ablation.empty:
        warn("ablation summary requires distilbert_ablation_metrics.csv")
        return pd.DataFrame()

    ablation["dataset"] = dataset_column(ablation)
    if "ablation_config" not in ablation.columns:
        warn("distilbert_ablation_metrics.csv has no ablation_config column")
        return pd.DataFrame()

    rename_map = {
        "eval_accuracy": "accuracy",
        "eval_precision_macro": "precision_macro",
        "eval_recall_macro": "recall_macro",
        "eval_f1_macro": "f1_macro",
        "eval_precision_weighted": "precision_weighted",
        "eval_recall_weighted": "recall_weighted",
        "eval_f1_weighted": "f1_weighted",
    }
    ablation = ablation.rename(columns=rename_map)
    ablation = latest_rows(ablation, ["dataset", "ablation_config"])

    columns = [
        "dataset",
        "ablation_config",
        "freeze_mode",
        "classifier_hidden_sizes",
        "dropout",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "total_params",
        "trainable_params",
        "gpu_memory_peak_mb",
        "latency_ms_per_sample",
    ]
    available = [column for column in columns if column in ablation.columns]
    return ablation[available].sort_values(["dataset", "ablation_config"])


def build_best_distilbert_table() -> pd.DataFrame:
    """Build the selected best DistilBERT configuration table."""
    best = normalize_columns(read_csv_if_exists(METRICS_DIR / "best_distilbert_ablation.csv"))
    if best.empty:
        warn("best DistilBERT table requires best_distilbert_ablation.csv")
        return pd.DataFrame()

    columns = [
        "dataset",
        "best_config",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "total_params",
        "trainable_params",
        "gpu_memory_peak_mb",
        "latency_ms_per_sample",
    ]
    return best[[column for column in columns if column in best.columns]]


def aggregate_results() -> list[Path]:
    """Generate all Day 5 report tables and return their paths."""
    generated: list[Path] = []
    table_builders = {
        "final_performance_table": build_final_performance_table,
        "final_efficiency_table": build_final_efficiency_table,
        "ablation_summary_table": build_ablation_summary_table,
        "best_distilbert_table": build_best_distilbert_table,
    }

    for name, builder in table_builders.items():
        table = builder()
        generated.extend(save_table(table, name))

    return generated


def main() -> None:
    """CLI entry point for table aggregation."""
    generated = aggregate_results()
    print("Generated tables:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
