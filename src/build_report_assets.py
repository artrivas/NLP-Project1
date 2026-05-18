"""Build report assets from saved Day 5 outputs.

This script copies figures and exports LaTeX table snippets. It does not train
models, load checkpoints, or run any heavy GPU computation.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.export_report_tables import export_report_tables


RESULT_FIGURES_DIR = Path("results/figures")
REPORT_FIGURES_DIR = Path("report/figures")
REPORT_TABLES_DIR = Path("report/tables")

REQUIRED_FIGURES = [
    "bubble_params_accuracy_combined.png",
    "f1_comparison_by_dataset.png",
    "latency_comparison_by_dataset.png",
    "gpu_memory_comparison_by_dataset.png",
    "loss_curves_ag_news.png",
    "loss_curves_sst2.png",
    "loss_curves_yelp_review_full.png",
]


def copy_report_figures() -> tuple[list[Path], list[Path]]:
    """Copy available result figures into the report asset directory."""
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    missing: list[Path] = []

    for filename in REQUIRED_FIGURES:
        source = RESULT_FIGURES_DIR / filename
        destination = REPORT_FIGURES_DIR / filename
        if source.exists():
            shutil.copy2(source, destination)
            copied.append(destination)
        else:
            missing.append(source)

    return copied, missing


def build_report_assets() -> None:
    """Create report asset folders, copy figures, and export tables."""
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    copied, missing = copy_report_figures()
    tables = export_report_tables()

    print("Report asset build complete.")
    print("Copied figures:")
    for path in copied:
        print(f"- {path}")

    if missing:
        print("Missing figures:")
        for path in missing:
            print(f"- {path}")
        print("Run `python -m src.make_results_summary` after generating full-run results.")

    print("Generated LaTeX tables:")
    for path in tables:
        print(f"- {path}")

    print("No training or checkpoint loading was performed.")


def main() -> None:
    """CLI entry point."""
    build_report_assets()


if __name__ == "__main__":
    main()
