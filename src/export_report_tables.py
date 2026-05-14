"""Export compact LaTeX table snippets for the Day 6 report.

This script only reads CSV files from ``results/tables`` and writes LaTeX
snippets under ``report/tables``. It does not train models or load checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


RESULT_TABLES_DIR = Path("results/tables")
REPORT_TABLES_DIR = Path("report/tables")


METRIC_COLUMNS = {
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
}
PARAM_COLUMNS = {"total_params", "trainable_params"}
LATENCY_COLUMNS = {"latency_ms_per_sample", "latency_ms_per_batch"}
MEMORY_COLUMNS = {"gpu_memory_peak_mb", "peak_cuda_memory_mb"}


def warn(message: str) -> None:
    """Print a clear warning without interrupting the export process."""
    print(f"Warning: {message}")


def latex_escape(value: object) -> str:
    """Escape text for LaTeX table cells."""
    if pd.isna(value):
        return "--"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_value(column: str, value: object) -> str:
    """Format table values according to report requirements."""
    if pd.isna(value):
        return "--"
    if column in METRIC_COLUMNS:
        return f"{float(value):.4f}"
    if column in LATENCY_COLUMNS:
        return f"{float(value):.2f}"
    if column in MEMORY_COLUMNS:
        return f"{float(value):.1f}"
    if column in PARAM_COLUMNS:
        return f"{float(value) / 1_000_000:.2f}"
    if column == "dropout":
        return f"{float(value):.2f}"
    return latex_escape(value)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV table if available."""
    if not path.exists():
        warn(f"missing source table: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive report path
        warn(f"could not read {path}: {exc}")
        return pd.DataFrame()


def compact_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select available columns in a stable order."""
    return df[[column for column in columns if column in df.columns]].copy()


def rename_for_report(df: pd.DataFrame) -> pd.DataFrame:
    """Rename technical columns to compact report headers."""
    rename_map = {
        "dataset": "Dataset",
        "model_type": "Model",
        "model_name": "Model name",
        "configuration": "Config",
        "ablation_config": "Config",
        "best_config": "Best config",
        "accuracy": "Acc.",
        "precision_macro": "Prec.",
        "recall_macro": "Rec.",
        "f1_macro": "F1",
        "precision_weighted": "Prec. w.",
        "recall_weighted": "Rec. w.",
        "f1_weighted": "F1 w.",
        "freeze_mode": "Freeze",
        "classifier_hidden_sizes": "Head",
        "dropout": "Drop.",
        "total_params": "Params (M)",
        "trainable_params": "Trainable (M)",
        "latency_ms_per_sample": "Latency",
        "latency_ms_per_batch": "Batch lat.",
        "gpu_memory_peak_mb": "GPU MB",
        "batch_size": "Batch",
        "fp16": "FP16",
        "device_name": "Device",
    }
    return df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns})


def to_latex_tabular(df: pd.DataFrame, source_columns: list[str]) -> str:
    """Convert a DataFrame to a compact LaTeX tabular snippet."""
    if df.empty:
        return "% TODO: Source result file missing. Regenerate full-run results and rerun src.export_report_tables.\n"

    formatted = df.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(lambda value: format_value(column, value))

    formatted = rename_for_report(formatted)
    alignment = "l" * min(3, len(formatted.columns)) + "r" * max(0, len(formatted.columns) - 3)
    lines = [
        r"\begin{tabular}{" + alignment + "}",
        r"\toprule",
        " & ".join(latex_escape(column) for column in formatted.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in formatted.iterrows():
        lines.append(" & ".join(str(row[column]) for column in formatted.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def export_table(
    source_name: str,
    output_name: str,
    columns: list[str],
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> Path:
    """Export one report table snippet."""
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    source = RESULT_TABLES_DIR / source_name
    output = REPORT_TABLES_DIR / output_name
    df = read_csv(source)
    if transform is not None and not df.empty:
        df = transform(df)
    if not df.empty:
        df = compact_columns(df, columns)
    output.write_text(to_latex_tabular(df, columns), encoding="utf-8")
    return output


def top_ablation_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a compact ablation table by selecting top rows per dataset."""
    if "f1_macro" not in df.columns:
        return df
    sorted_df = df.sort_values(["dataset", "f1_macro", "accuracy"], ascending=[True, False, False])
    return sorted_df.groupby("dataset", as_index=False, group_keys=False).head(3)


def export_report_tables() -> list[Path]:
    """Export all LaTeX tables required by the report."""
    generated = [
        export_table(
            "final_performance_table.csv",
            "final_performance_table.tex",
            ["dataset", "model_type", "configuration", "accuracy", "f1_macro", "f1_weighted"],
        ),
        export_table(
            "final_efficiency_table.csv",
            "final_efficiency_table.tex",
            [
                "dataset",
                "model_type",
                "configuration",
                "total_params",
                "trainable_params",
                "latency_ms_per_sample",
                "gpu_memory_peak_mb",
            ],
        ),
        export_table(
            "ablation_summary_table.csv",
            "ablation_summary_table.tex",
            [
                "dataset",
                "ablation_config",
                "freeze_mode",
                "classifier_hidden_sizes",
                "accuracy",
                "f1_macro",
                "total_params",
                "trainable_params",
                "latency_ms_per_sample",
            ],
            transform=top_ablation_rows,
        ),
        export_table(
            "best_distilbert_table.csv",
            "best_distilbert_table.tex",
            [
                "dataset",
                "best_config",
                "accuracy",
                "f1_macro",
                "total_params",
                "trainable_params",
                "latency_ms_per_sample",
                "gpu_memory_peak_mb",
            ],
        ),
    ]
    return generated


def main() -> None:
    """CLI entry point."""
    generated = export_report_tables()
    print("Generated report tables:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
