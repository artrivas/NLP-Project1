# NeurIPS-Style Mini Report

This directory contains the Day 6 mini technical report:

- `main.tex`: report source
- `references.bib`: BibTeX references
- `neurips_2025.sty`: official NeurIPS 2025 style file
- `figures/`: copied PNG figures from `results/figures/`
- `tables/`: generated LaTeX table snippets from `results/tables/`
- `Makefile`: build and clean commands

## NeurIPS 2025 Style

The official style file should be downloaded from:

```text
https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip
```

Place `neurips_2025.sty` directly in this `report/` directory. In this workspace it has already been copied into place. Do not hand-write or modify the style file content.

## Generate Report Assets

From the repository root:

```bash
python -m src.build_report_assets
```

This command copies available figures from `results/figures/` into `report/figures/` and generates LaTeX table snippets in `report/tables/`.

If assets are missing, first regenerate the Day 5 summaries:

```bash
python -m src.make_results_summary
python -m src.build_report_assets
```

## Compile

From this directory:

```bash
make pdf
```

The output is:

```text
report/main.pdf
```

The Makefile uses `latexmk` when available. Otherwise it falls back to `pdflatex` and `bibtex`.

## Clean

```bash
make clean
```

## Important

Debug-fast results are only for checking that the pipeline and report compile. They must not be used for final conclusions. Run the full experiments first, regenerate Day 5 summaries, rebuild report assets, and then compile the final PDF.
