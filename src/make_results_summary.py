"""Generate Day 5 result tables and figures.

This script does not train models. It only aggregates existing metrics/log CSVs
and renders report-ready artifacts.
"""

from __future__ import annotations

from pathlib import Path

from src.aggregate_results import aggregate_results
from src.plot_results import generate_plots


def main() -> None:
    """Run table aggregation and plotting, then print generated artifacts."""
    table_paths = aggregate_results()
    figure_paths = generate_plots()

    print("\nDay 5 results summary complete.")
    print(f"Tables generated: {len(table_paths)}")
    for path in table_paths:
        print(f"- {path}")

    print(f"Figures generated: {len(figure_paths)}")
    for path in figure_paths:
        print(f"- {path}")

    print("\nNo training or checkpoint loading was performed.")
    print(f"Tables directory: {Path('results/tables')}")
    print(f"Figures directory: {Path('results/figures')}")


if __name__ == "__main__":
    main()
