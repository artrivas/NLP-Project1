"""Generate report figures from saved metrics and logs.

This module does not train or load model checkpoints. It only reads CSV files
created by earlier experiment stages and writes PNG figures.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"

DATASETS = ["ag_news", "sst2", "yelp_review_full"]


def warn(message: str) -> None:
    """Print a standardized warning without stopping figure generation."""
    print(f"Warning: {message}")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read a CSV file if it exists, otherwise return an empty DataFrame."""
    if not path.exists():
        warn(f"missing file: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive reporting path
        warn(f"could not read {path}: {exc}")
        return pd.DataFrame()
    df.columns = [str(column).strip().lower().replace(" ", "_") for column in df.columns]
    return df


def save_current_figure(path: Path) -> Path:
    """Save the active matplotlib figure using report-friendly defaults."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def label_for(row: pd.Series) -> str:
    """Create a compact model/config label for plots."""
    model_type = str(row.get("model_type", "")).strip()
    configuration = str(row.get("configuration", "")).strip()
    if configuration and configuration.lower() != "nan":
        return f"{model_type}: {configuration}"
    return model_type or str(row.get("model_name", "model"))


def load_final_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load final performance, efficiency, and merged result frames."""
    performance = read_csv_if_exists(METRICS_DIR / "final_comparison_metrics.csv")
    efficiency = read_csv_if_exists(METRICS_DIR / "final_efficiency_metrics.csv")
    if performance.empty or efficiency.empty:
        return performance, efficiency, pd.DataFrame()

    keys = ["dataset", "model_type", "model_name", "configuration"]
    merged = performance.merge(efficiency, on=keys, how="outer")
    merged["plot_label"] = merged.apply(label_for, axis=1)
    return performance, efficiency, merged


def plot_bubble(data: pd.DataFrame, dataset: str | None = None) -> Path | None:
    """Plot parameter count vs accuracy, optionally filtered to one dataset."""
    if data.empty:
        warn("bubble plot skipped because final comparison/efficiency data is missing")
        return None

    plot_data = data.copy()
    if dataset is not None:
        plot_data = plot_data[plot_data["dataset"] == dataset]
    plot_data = plot_data.dropna(subset=["total_params", "accuracy"])
    if plot_data.empty:
        warn(f"bubble plot skipped for {dataset or 'combined'} because required values are missing")
        return None

    size_metric = "latency_ms_per_sample"
    if plot_data[size_metric].isna().all() and "gpu_memory_peak_mb" in plot_data.columns:
        size_metric = "gpu_memory_peak_mb"

    sizes = plot_data[size_metric].fillna(plot_data[size_metric].median())
    if sizes.isna().all() or (sizes <= 0).all():
        sizes = pd.Series([100.0] * len(plot_data), index=plot_data.index)
    else:
        sizes = 80 + 520 * (sizes / sizes.max())

    plt.figure(figsize=(8, 5))
    for label, group in plot_data.groupby("plot_label"):
        group_sizes = sizes.loc[group.index]
        plt.scatter(
            group["total_params"] / 1_000_000,
            group["accuracy"],
            s=group_sizes,
            alpha=0.65,
            label=label,
            edgecolors="black",
            linewidths=0.5,
        )
        for _, row in group.iterrows():
            plt.annotate(
                str(row["configuration"]),
                (row["total_params"] / 1_000_000, row["accuracy"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    title_dataset = "Combined" if dataset is None else dataset
    plt.title(f"Parameters vs Accuracy ({title_dataset})")
    plt.xlabel("Parameters (millions)")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)

    suffix = "combined" if dataset is None else dataset
    return save_current_figure(FIGURES_DIR / f"bubble_params_accuracy_{suffix}.png")


def plot_metric_bar(
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> Path | None:
    """Plot a grouped bar chart for a final comparison metric."""
    if data.empty or metric not in data.columns:
        warn(f"{filename} has no source data; writing placeholder figure")
        return plot_no_data(title, f"No {ylabel.lower()} data available", filename)

    plot_data = data.dropna(subset=[metric]).copy()
    if plot_data.empty:
        warn(f"{filename} has no {metric} values; writing placeholder figure")
        return plot_no_data(title, f"No {ylabel.lower()} data available", filename)

    plot_data["plot_label"] = plot_data.apply(label_for, axis=1)
    pivot = plot_data.pivot_table(
        index="dataset",
        columns="plot_label",
        values=metric,
        aggfunc="last",
    )
    pivot = pivot.reindex([dataset for dataset in DATASETS if dataset in pivot.index])

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model/configuration", fontsize=8)
    plt.xticks(rotation=0)
    return save_current_figure(FIGURES_DIR / filename)


def plot_no_data(title: str, message: str, filename: str) -> Path:
    """Create a report-stable placeholder figure when optional data is absent."""
    plt.figure(figsize=(8, 4.5))
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    plt.axis("off")
    return save_current_figure(FIGURES_DIR / filename)


def load_best_distilbert_configs() -> dict[str, str]:
    """Load selected DistilBERT ablation config per dataset."""
    best = read_csv_if_exists(METRICS_DIR / "best_distilbert_ablation.csv")
    if best.empty or not {"dataset", "best_config"}.issubset(best.columns):
        warn("best DistilBERT configs are unavailable; using any DistilBERT logs present")
        return {}
    return dict(zip(best["dataset"], best["best_config"]))


def normalize_log_dataset(df: pd.DataFrame) -> pd.Series:
    """Return dataset run names from a log frame."""
    if "dataset_run_name" in df.columns:
        return df["dataset_run_name"]
    if "dataset" in df.columns:
        return df["dataset"]
    if "dataset_config" in df.columns:
        return df["dataset_config"].fillna(df.get("dataset_name", ""))
    return pd.Series([""] * len(df), index=df.index)


def latest_run(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Keep rows from the latest run for each group."""
    if df.empty or "run_id" not in df.columns:
        return df
    selected = []
    for _, group in df.groupby(group_columns, dropna=False):
        latest = group["run_id"].astype(str).max()
        selected.append(group[group["run_id"].astype(str) == latest])
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True)


def collect_loss_logs(dataset: str) -> list[tuple[str, pd.DataFrame]]:
    """Collect BERT and best-DistilBERT loss logs for a dataset."""
    logs: list[tuple[str, pd.DataFrame]] = []
    best_configs = load_best_distilbert_configs()

    bert = read_csv_if_exists(LOGS_DIR / "bert_training_logs.csv")
    if not bert.empty:
        bert["dataset"] = normalize_log_dataset(bert)
        bert = bert[bert["dataset"] == dataset]
        bert = latest_run(bert, ["dataset", "model_name"])
        if not bert.empty:
            logs.append(("BERT-base", bert))

    ablation = read_csv_if_exists(LOGS_DIR / "distilbert_ablation_training_logs.csv")
    if not ablation.empty:
        ablation["dataset"] = normalize_log_dataset(ablation)
        ablation = ablation[ablation["dataset"] == dataset]
        best_config = best_configs.get(dataset)
        if best_config and "ablation_config" in ablation.columns:
            ablation = ablation[ablation["ablation_config"] == best_config]
        group_columns = ["dataset"]
        if "ablation_config" in ablation.columns:
            group_columns.append("ablation_config")
        ablation = latest_run(ablation, group_columns)
        if not ablation.empty:
            label = f"DistilBERT ({best_config})" if best_config else "DistilBERT"
            logs.append((label, ablation))

    return logs


def plot_loss_curves(dataset: str) -> Path | None:
    """Plot training and validation loss curves for one dataset."""
    logs = collect_loss_logs(dataset)
    if not logs:
        warn(f"loss curve skipped for {dataset}; no usable logs found")
        return None

    plt.figure(figsize=(9, 5))
    has_curve = False
    for label, df in logs:
        x = pd.to_numeric(df.get("step", pd.Series(range(len(df)))), errors="coerce")
        if x.isna().all():
            x = pd.Series(range(1, len(df) + 1), index=df.index)

        if "loss" in df.columns:
            train = pd.to_numeric(df["loss"], errors="coerce")
            mask = train.notna()
            if mask.any():
                plt.plot(x[mask], train[mask], marker="o", linewidth=1.5, label=f"{label} train")
                has_curve = True

        if "eval_loss" in df.columns:
            eval_loss = pd.to_numeric(df["eval_loss"], errors="coerce")
            mask = eval_loss.notna()
            if mask.any():
                plt.plot(
                    x[mask],
                    eval_loss[mask],
                    marker="s",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"{label} validation",
                )
                has_curve = True

    if not has_curve:
        plt.close()
        warn(f"loss curve skipped for {dataset}; no loss values found")
        return None

    plt.title(f"Training and Validation Loss ({dataset})")
    plt.xlabel("Iteration / Trainer step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    return save_current_figure(FIGURES_DIR / f"loss_curves_{dataset}.png")


def generate_plots() -> list[Path]:
    """Generate all Day 5 report figures and return generated paths."""
    generated: list[Path] = []
    performance, efficiency, merged = load_final_tables()

    for path in [
        plot_bubble(merged),
        *(plot_bubble(merged, dataset) for dataset in DATASETS),
        plot_metric_bar(
            performance,
            "f1_macro",
            "Macro F1",
            "Macro F1 Comparison by Dataset",
            "f1_comparison_by_dataset.png",
        ),
        plot_metric_bar(
            efficiency,
            "latency_ms_per_sample",
            "Latency (ms/sample)",
            "Latency Comparison by Dataset",
            "latency_comparison_by_dataset.png",
        ),
        plot_metric_bar(
            efficiency,
            "gpu_memory_peak_mb",
            "GPU peak memory (MB)",
            "GPU Memory Comparison by Dataset",
            "gpu_memory_comparison_by_dataset.png",
        ),
        *(plot_loss_curves(dataset) for dataset in DATASETS),
    ]:
        if path is not None:
            generated.append(path)

    return generated


def main() -> None:
    """CLI entry point for plot generation."""
    generated = generate_plots()
    print("Generated figures:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
