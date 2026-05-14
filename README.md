# DistilBERT vs BERT Text Classification

This project compares DistilBERT and BERT-base for text classification on AG News, SST-2, and Yelp Review Full.

Day 1 focuses only on reproducible project setup, dataset loading, and tokenization. No model fine-tuning is implemented yet.

## Project Structure

```text
.
├── config/
│   ├── ag_news.yaml
│   ├── sst2.yaml
│   └── yelp_review_full.yaml
├── src/
│   ├── data.py
│   ├── smoke_test_data.py
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

## Day 2 Next Step

Implement the DistilBERT baseline fine-tuning loop, including training arguments, evaluation metrics, checkpointing, and reproducible experiment logging.
