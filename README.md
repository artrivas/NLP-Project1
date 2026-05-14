# DistilBERT vs BERT Text Classification

This project compares DistilBERT and BERT-base for text classification on AG News, SST-2, and Yelp Review Full.

Day 1 focused on reproducible project setup, dataset loading, and tokenization.
Day 2 adds DistilBERT baseline fine-tuning only. BERT comparison, ablations, and the final report are intentionally not implemented yet.

## Project Structure

```text
.
├── config/
│   ├── ag_news.yaml
│   ├── sst2.yaml
│   └── yelp_review_full.yaml
├── src/
│   ├── data.py
│   ├── metrics.py
│   ├── run_distilbert_baselines.py
│   ├── smoke_test_data.py
│   ├── train.py
│   └── utils.py
├── results/
├── experiments/
└── notebooks/
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Day 1 Smoke Tests

Run the data pipeline checks without training any model:

```bash
python -m src.smoke_test_data --config config/ag_news.yaml --model distilbert-base-uncased
python -m src.smoke_test_data --config config/sst2.yaml --model distilbert-base-uncased
python -m src.smoke_test_data --config config/yelp_review_full.yaml --model distilbert-base-uncased
```

Each command should print the dataset name, train and evaluation split sizes, configured text and label columns, number of labels, tokenizer name, tokenized sample keys, and the lengths of `input_ids` and `attention_mask`.

## Day 2 DistilBERT Baseline Training

Debug-fast mode is only for checking that the training code runs end to end. By default it uses 32 train samples, 16 eval samples, and 2 optimizer steps:

```bash
python -m src.train --config config/sst2.yaml --model distilbert-base-uncased --debug-fast
python -m src.run_distilbert_baselines --debug-fast
```

Full-run mode is intended for real DistilBERT baseline training, ideally on the RTX 4090:

```bash
python -m src.train --config config/sst2.yaml --model distilbert-base-uncased --full-run
python -m src.train --config config/ag_news.yaml --model distilbert-base-uncased --full-run
python -m src.train --config config/yelp_review_full.yaml --model distilbert-base-uncased --full-run
python -m src.run_distilbert_baselines --full-run
```

Useful overrides:

```bash
python -m src.train --config config/ag_news.yaml --model distilbert-base-uncased --full-run --batch-size 32 --epochs 3 --learning-rate 2e-5
python -m src.train --config config/sst2.yaml --model distilbert-base-uncased --debug-fast --max-train-samples 128 --max-eval-samples 64 --no-fp16
```

Generated files:

```text
results/
├── metrics/
│   ├── distilbert_baseline_metrics.csv
│   └── distilbert_<dataset>_<mode>_<run_id>.json
├── logs/
│   └── distilbert_baseline_training_logs.csv
└── checkpoints/
    └── distilbert/
        ├── ag_news/
        ├── sst2/
        └── yelp_review_full/
```

The consolidated metrics CSV is appended to on each run. The training log CSV stores Trainer log history, including training loss and evaluation loss entries.

Day 3 next step: compare the best DistilBERT baseline configuration against BERT-base. Do not run that comparison as part of Day 2.

## Day 3 DistilBERT Ablation Study

Day 3 trains DistilBERT-only classifier variants. It does not implement BERT comparison.

Ablation configurations:

- `baseline`: fully trainable DistilBERT with the standard simple classifier head.
- `frozen_transformer`: frozen DistilBERT transformer, train classifier only.
- `small_classifier`: fully trainable DistilBERT with `768 -> 128 -> num_labels`.
- `large_classifier`: fully trainable DistilBERT with `768 -> 512 -> 256 -> num_labels`.
- `freeze_lower_layers`: freeze embeddings and the first 3 DistilBERT layers, train upper layers with `768 -> 256 -> num_labels`.

Run one dataset in debug-fast mode:

```bash
python -m src.ablation --dataset-config config/sst2.yaml --ablation-config config/ablation_distilbert.yaml --debug-fast
```

Run one ablation configuration only:

```bash
python -m src.ablation --dataset-config config/sst2.yaml --ablation-config config/ablation_distilbert.yaml --config-name frozen_transformer --debug-fast
```

Run all ablations across all datasets:

```bash
python -m src.run_ablation_all --debug-fast
python -m src.run_ablation_all --full-run
```

Select the best DistilBERT ablation per dataset:

```bash
python -m src.select_best_ablation
```

Generated Day 3 files:

```text
results/
├── metrics/
│   ├── distilbert_ablation_metrics.csv
│   └── best_distilbert_ablation.csv
├── logs/
│   └── distilbert_ablation_training_logs.csv
└── checkpoints/
    └── distilbert_ablation/
        ├── ag_news/
        ├── sst2/
        └── yelp_review_full/
```

## Day 4 Final Comparison

Day 4 compares the best DistilBERT ablation configuration against `bert-base-uncased`. The best DistilBERT configuration is read from:

```text
results/metrics/best_distilbert_ablation.csv
```

Run BERT debug-fast training:

```bash
python -m src.train --config config/sst2.yaml --model bert-base-uncased --debug-fast
```

Run BERT full training:

```bash
python -m src.train --config config/ag_news.yaml --model bert-base-uncased --full-run
python -m src.train --config config/sst2.yaml --model bert-base-uncased --full-run
python -m src.train --config config/yelp_review_full.yaml --model bert-base-uncased --full-run
```

Run the final comparison:

```bash
python -m src.final_comparison --debug-fast
python -m src.final_comparison --full-run
```

Benchmark a trained checkpoint:

```bash
python -m src.benchmark --config config/sst2.yaml --model-path results/checkpoints/bert/sst2/best_model --model-name bert-base-uncased --debug-fast
```

Generated Day 4 files:

```text
results/
├── metrics/
│   ├── bert_metrics.csv
│   ├── final_comparison_metrics.csv
│   └── final_efficiency_metrics.csv
├── logs/
│   └── bert_training_logs.csv
├── benchmarks/
│   └── benchmark_results.csv
└── checkpoints/
    └── bert/
        ├── ag_news/
        ├── sst2/
        └── yelp_review_full/
```

Debug-fast results are only for verifying code correctness. Trust the final comparison only after running full-run experiments on the target RTX 4090.
