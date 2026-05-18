# Results Directory

This directory stores experiment outputs and Day 5 reporting artifacts.

## Safe to Commit

- `results/README.md`
- `results/tables/*.csv`
- `results/tables/*.md`
- `results/figures/*.png`
- `.gitkeep` placeholders

These files are compact summaries and plots that can be regenerated from saved metrics.

## Usually Do Not Commit

- `results/checkpoints/`
- raw Trainer checkpoints such as `checkpoint-*`
- model weights such as `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.ckpt`
- `results/logs/` if the logs become large
- `results/metrics/*.json` run snapshots if they become large
- `results/benchmarks/benchmark_results.csv` if it contains machine-specific repeated runs

## Folder Guide

- `metrics/`: consolidated CSV and JSON metrics from training, ablation, benchmarking, and final comparison.
- `logs/`: Trainer log-history CSV files, including training and validation loss entries.
- `benchmarks/`: latency, parameter count, and memory benchmark summaries.
- `checkpoints/`: saved model checkpoints and tokenizer files. These are ignored by git.
- `tables/`: Day 5 report-ready CSV and Markdown tables.
- `figures/`: Day 5 report-ready PNG figures.

## Regenerate Tables and Figures

Run:

```bash
python -m src.make_results_summary
```

This command does not train models and does not run heavy GPU computation. It only reads existing CSV logs/metrics and writes `results/tables/` and `results/figures/`.

Debug-fast results are useful for validating the reporting pipeline, but they are not valid for final experimental conclusions.
