"""Comprehensive visualization of all experiment results.

Generates ~25 publication-quality figures covering performance, efficiency,
ablation analysis, training dynamics, and model comparison.

Usage:
    python -m src.visualize --results-dir results --figures-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DATASETS = ["ag_news", "sst2", "yelp_review_full"]
DATASET_LABELS = {
    "ag_news": "AG News",
    "sst2": "SST-2",
    "yelp_review_full": "Yelp Review Full",
}
ABLATION_CONFIGS = [
    "baseline",
    "frozen_transformer",
    "small_classifier",
    "large_classifier",
    "freeze_lower_layers",
]
ABLATION_LABELS = {
    "baseline": "Baseline",
    "frozen_transformer": "Frozen Transformer",
    "small_classifier": "Small Head (128)",
    "large_classifier": "Large Head (512→256)",
    "freeze_lower_layers": "Freeze Lower 3 + Emb",
}
MODEL_COLORS = {
    "distilbert": "#3B82F6",
    "bert": "#EF4444",
}
ABLATION_CMAP = {
    "baseline": "#6366F1",
    "frozen_transformer": "#F59E0B",
    "small_classifier": "#10B981",
    "large_classifier": "#EC4899",
    "freeze_lower_layers": "#8B5CF6",
}


def warn(msg: str) -> None:
    print(f"[VIS] Warning: {msg}")


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        warn(f"missing: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
        return df
    except Exception as exc:
        warn(f"error reading {path}: {exc}")
        return pd.DataFrame()


def save(name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    dest = out_dir / f"{name}.png"
    plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {dest}")
    return dest


def ds_label(ds: str) -> str:
    return DATASET_LABELS.get(ds, ds)


def fix_eval_prefix(df: pd.DataFrame) -> pd.DataFrame:
    renames = {}
    for c in df.columns:
        if c.startswith("eval_"):
            renames[c] = c[5:]
    return df.rename(columns=renames)


def filter_ds(df: pd.DataFrame, ds: str, col: str = "dataset") -> pd.DataFrame:
    if col not in df.columns:
        alt = "dataset_run_name" if "dataset_run_name" in df.columns else None
        if alt:
            return df[df[alt] == ds].copy()
        return pd.DataFrame()
    return df[df[col] == ds].copy()


def get_ds_col(df: pd.DataFrame) -> pd.Series:
    if "dataset" in df.columns:
        return df["dataset"]
    if "dataset_run_name" in df.columns:
        return df["dataset_run_name"]
    return pd.Series([""] * len(df), index=df.index)


# ─── 1. F1 & Accuracy Grouped Bars ───────────────────────────────────────────

def plot_f1_comparison(perf: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    for metric, label in [("f1_macro", "Macro F1"), ("accuracy", "Accuracy")]:
        fig, ax = plt.subplots(figsize=(9, 5))
        if perf.empty or metric not in perf.columns:
            ax.set_title(f"{label} — no data")
            ax.axis("off")
            paths.append(save(f"{metric}_comparison_by_dataset", out))
            continue

        ds_col = get_ds_col(perf)
        _df = perf.copy()
        _df["dataset"] = ds_col

        pivot = _df.pivot_table(
            index="dataset", columns="configuration", values=metric, aggfunc="last"
        )
        pivot = pivot.reindex([d for d in DATASETS if d in pivot.index])
        pivot.plot(kind="bar", ax=ax, width=0.7)
        ax.set_title(f"{label} Comparison by Dataset")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(label)
        ax.set_xticklabels([ds_label(d) for d in pivot.index], rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Config", fontsize=8)
        paths.append(save(f"{metric}_comparison_by_dataset", out))
    return paths


# ─── 2. Ablation Detailed Bars (per dataset) ─────────────────────────────────

def plot_ablation_detail(abl: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    abl = fix_eval_prefix(abl)
    abl["dataset"] = get_ds_col(abl)

    for ds in DATASETS:
        sub = abl[abl["dataset"] == ds] if "dataset" in abl.columns else pd.DataFrame()
        if sub.empty:
            warn(f"ablation detail: no data for {ds}")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        configs = sub["ablation_config"].tolist() if "ablation_config" in sub.columns else []
        colors = [ABLATION_CMAP.get(c, "#999") for c in configs]

        for i, (metric, ylabel) in enumerate([
            ("accuracy", "Accuracy"),
            ("f1_macro", "Macro F1"),
            ("trainable_params", "Trainable Params"),
        ]):
            ax = axes[i]
            if metric not in sub.columns:
                ax.set_title(f"{ylabel} — missing"); continue
            vals = sub[metric].astype(float)
            ax.bar(range(len(configs)), vals, color=colors)
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ds_label(ds)} — {ylabel}")
            ax.grid(axis="y", alpha=0.25)

        paths.append(save(f"ablation_detail_{ds}", out))
    return paths


# ─── 3. Ablation Heatmap ──────────────────────────────────────────────────────

def plot_ablation_heatmap(abl: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    abl = fix_eval_prefix(abl)
    if abl.empty or "ablation_config" not in abl.columns:
        warn("ablation heatmap: no data")
        return paths

    abl["dataset"] = get_ds_col(abl)
    for metric, label in [("f1_macro", "Macro F1"), ("accuracy", "Accuracy")]:
        if metric not in abl.columns:
            continue
        pivot = abl.pivot_table(
            index="dataset", columns="ablation_config", values=metric, aggfunc="last"
        )
        pivot = pivot.reindex(
            [d for d in DATASETS if d in pivot.index],
            columns=[c for c in ABLATION_CONFIGS if c in pivot.columns],
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticklabels([ds_label(d) for d in pivot.index])
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)
        ax.set_title(f"Ablation {label} Heatmap")
        fig.colorbar(im, ax=ax, label=label)
        paths.append(save(f"ablation_heatmap_{metric}", out))
    return paths


# ─── 4. Parameter Efficiency ─────────────────────────────────────────────────

def plot_param_efficiency(abl: pd.DataFrame, merged: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    source = fix_eval_prefix(abl) if not abl.empty else merged
    if source.empty:
        warn("param efficiency: no data"); return paths
    source = source.copy()
    source["dataset"] = get_ds_col(source)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, metric in enumerate(["f1_macro", "accuracy", "latency_ms_per_sample"]):
        ax = axes[i]
        needs_x = "trainable_params"
        if needs_x not in source.columns or metric not in source.columns:
            ax.set_title(f"{metric} — missing"); continue
        sub = source.dropna(subset=[needs_x, metric])
        if sub.empty:
            continue

        for ds, grp in sub.groupby("dataset"):
            ax.scatter(
                grp[needs_x] / 1e6,
                grp[metric].astype(float),
                label=ds_label(ds),
                s=60, alpha=0.7,
            )
        ax.set_xlabel("Trainable Params (M)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Trainable Params")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    paths.append(save("param_efficiency_triple", out))

    for ds in DATASETS:
        sub = source[source["dataset"] == ds] if "dataset" in source.columns else pd.DataFrame()
        if sub.empty or "trainable_params" not in sub.columns or "f1_macro" not in sub.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sub["trainable_params"] / 1e6, sub["f1_macro"].astype(float), s=80, alpha=0.7, c="#6366F1")
        if "ablation_config" in sub.columns:
            for _, row in sub.iterrows():
                ax.annotate(str(row["ablation_config"]), (row["trainable_params"]/1e6, float(row["f1_macro"])),
                            textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("Trainable Params (M)")
        ax.set_ylabel("Macro F1")
        ax.set_title(f"Parameter Efficiency — {ds_label(ds)}")
        ax.grid(alpha=0.25)
        paths.append(save(f"param_efficiency_{ds}", out))
    return paths


# ─── 5. Speed / Latency Bars ──────────────────────────────────────────────────

def plot_latency(eff: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if eff.empty:
        warn("latency: no efficiency data"); return paths

    eff = eff.copy()
    eff["dataset"] = get_ds_col(eff)

    for metric, label in [
        ("latency_ms_per_sample", "Latency (ms/sample)"),
        ("latency_ms_per_batch", "Latency (ms/batch)"),
    ]:
        if metric not in eff.columns:
            continue
        pivot = eff.pivot_table(
            index="dataset", columns="configuration", values=metric, aggfunc="last"
        )
        pivot = pivot.reindex([d for d in DATASETS if d in pivot.index])
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax, width=0.7)
        ax.set_title(f"{label} by Dataset")
        ax.set_ylabel(label)
        ax.set_xticklabels([ds_label(d) for d in pivot.index], rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Config", fontsize=8)
        paths.append(save(f"{metric}_by_dataset", out))
    return paths


# ─── 6. GPU Memory Bars ───────────────────────────────────────────────────────

def plot_gpu_memory(eff: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    eff = eff.copy()
    eff["dataset"] = get_ds_col(eff)
    for col in ["gpu_memory_peak_mb", "peak_cuda_memory_mb"]:
        if col not in eff.columns:
            continue
        pivot = eff.pivot_table(
            index="dataset", columns="configuration", values=col, aggfunc="last"
        )
        pivot = pivot.reindex([d for d in DATASETS if d in pivot.index])
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax, width=0.7)
        ax.set_title("GPU Peak Memory (MB) by Dataset")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_xticklabels([ds_label(d) for d in pivot.index], rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Config", fontsize=8)
        paths.append(save(f"{col}_by_dataset", out))
    return paths


# ─── 7. Model Size Comparison ─────────────────────────────────────────────────

def plot_model_size(eff: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if eff.empty or "total_params" not in eff.columns:
        warn("model size: no data"); return paths
    eff = eff.copy()
    eff["dataset"] = get_ds_col(eff)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, label in [
        (axes[0], "total_params", "Total Parameters"),
        (axes[1], "trainable_params", "Trainable Parameters"),
    ]:
        if col not in eff.columns:
            ax.set_title(f"{label} — missing"); continue
        pivot = eff.pivot_table(index="dataset", columns="configuration", values=col, aggfunc="last")
        pivot = pivot.reindex([d for d in DATASETS if d in pivot.index])
        if pivot.empty:
            continue
        (pivot / 1e6).plot(kind="bar", ax=ax, width=0.7)
        ax.set_title(f"{label} by Dataset")
        ax.set_ylabel("Parameters (M)")
        ax.set_xticklabels([ds_label(d) for d in pivot.index], rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Config", fontsize=8)
    paths.append(save("model_size_comparison", out))
    return paths


# ─── 8. DistilBERT vs BERT Head-to-Head ───────────────────────────────────────

def plot_distilbert_vs_bert(perf: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if perf.empty:
        return paths
    perf = perf.copy()
    perf["dataset"] = get_ds_col(perf)
    perf = fix_eval_prefix(perf)

    for metric, label in [("f1_macro", "Macro F1"), ("accuracy", "Accuracy")]:
        if metric not in perf.columns or "model_type" not in perf.columns:
            continue
        pivot = perf.pivot_table(
            index="dataset", columns="model_type", values=metric, aggfunc="last"
        )
        pivot = pivot.reindex([d for d in DATASETS if d in pivot.index])
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(pivot.index))
        w = 0.35
        models = [c for c in ["distilbert", "bert"] if c in pivot.columns]
        for i, m in enumerate(models):
            ax.bar(x + i * w, pivot[m].astype(float), w, label=m.upper(), color=MODEL_COLORS.get(m))
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels([ds_label(d) for d in pivot.index])
        ax.set_ylabel(label)
        ax.set_title(f"DistilBERT vs BERT — {label}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        paths.append(save(f"distilbert_vs_bert_{metric}", out))
    return paths


# ─── 9. Speedup Ratio ─────────────────────────────────────────────────────────

def plot_speedup(eff: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if eff.empty or "latency_ms_per_sample" not in eff.columns:
        warn("speedup: no latency data"); return paths
    eff = eff.copy()
    eff["dataset"] = get_ds_col(eff)

    pivot = eff.pivot_table(
        index="dataset", columns="model_type", values="latency_ms_per_sample", aggfunc="last"
    )
    if "distilbert" not in pivot.columns or "bert" not in pivot.columns:
        warn("speedup: need both distilbert and bert latency")
        return paths

    speedup = pivot["bert"] / pivot["distilbert"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(speedup)), speedup.values, color="#10B981", edgecolor="black", linewidth=0.5)
    for i, v in enumerate(speedup.values):
        ax.text(i, v + 0.05, f"{v:.2f}×", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(range(len(speedup)))
    ax.set_xticklabels([ds_label(d) for d in speedup.index])
    ax.set_ylabel("BERT Latency / DistilBERT Latency")
    ax.set_title("Inference Speedup: DistilBERT vs BERT")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0)
    paths.append(save("speedup_ratio", out))
    return paths


# ─── 10. Bubble Plot (Params vs Accuracy) ──────────────────────────────────────

def plot_bubble(merged: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if merged.empty:
        warn("bubble: no merged data"); return paths

    merged = merged.copy()
    merged["dataset"] = get_ds_col(merged)

    targets = [None] + DATASETS
    for ds in targets:
        sub = merged if ds is None else merged[merged["dataset"] == ds]
        if sub.empty or "total_params" not in sub.columns or "accuracy" not in sub.columns:
            continue
        sub = sub.dropna(subset=["total_params", "accuracy"])

        size_col = "latency_ms_per_sample"
        if sub[size_col].isna().all() and "gpu_memory_peak_mb" in sub.columns:
            size_col = "gpu_memory_peak_mb"
        sizes = sub[size_col].fillna(sub[size_col].median())
        if sizes.isna().all() or (sizes <= 0).all():
            sizes = pd.Series([100.0] * len(sub), index=sub.index)
        else:
            sizes = 80 + 520 * (sizes / sizes.max())

        fig, ax = plt.subplots(figsize=(8, 5))
        for label, grp in sub.groupby("plot_label" if "plot_label" in sub.columns else "configuration"):
            ax.scatter(
                grp["total_params"] / 1e6, grp["accuracy"].astype(float),
                s=sizes.loc[grp.index], alpha=0.65, label=label, edgecolors="black", linewidths=0.5,
            )
            for _, row in grp.iterrows():
                ax.annotate(
                    str(row.get("configuration", label)),
                    (row["total_params"] / 1e6, float(row["accuracy"])),
                    textcoords="offset points", xytext=(5, 5), fontsize=8,
                )
        suffix = "combined" if ds is None else ds
        ax.set_title(f"Parameters vs Accuracy ({ds_label(ds) if ds else 'Combined'})")
        ax.set_xlabel("Parameters (M)")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        paths.append(save(f"bubble_params_accuracy_{suffix}", out))
    return paths


# ─── 11. Loss Curves ──────────────────────────────────────────────────────────

def plot_loss_curves(log_dir: Path, out: Path) -> list[Path]:
    paths = []
    best = read_csv_safe(Path("results/metrics/best_distilbert_ablation.csv"))

    log_files = {
        "distilbert_baseline": log_dir / "distilbert_baseline_training_logs.csv",
        "distilbert_ablation": log_dir / "distilbert_ablation_training_logs.csv",
        "bert": log_dir / "bert_training_logs.csv",
    }

    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(9, 5))
        has_curve = False

        for tag, path in log_files.items():
            df = read_csv_safe(path)
            if df.empty:
                continue
            ds_col = get_ds_col(df)
            mask = ds_col == ds
            if not mask.any():
                continue
            sub = df[mask].copy()

            if tag == "distilbert_ablation" and not best.empty and "best_config" in best.columns:
                cfg = best.loc[best["dataset"] == ds, "best_config"]
                if not cfg.empty and "ablation_config" in sub.columns:
                    sub = sub[sub["ablation_config"] == cfg.values[0]]

            x = pd.to_numeric(sub.get("step", pd.Series(range(len(sub)))), errors="coerce")
            if x.isna().all():
                x = pd.Series(range(len(sub)), index=sub.index)
            label = tag.replace("_", " ").title()

            if "loss" in sub.columns:
                y = pd.to_numeric(sub["loss"], errors="coerce")
                m = y.notna()
                if m.any():
                    ax.plot(x[m].values, y[m].values, marker="o", lw=1.5, label=f"{label} train", markersize=3)
                    has_curve = True
            if "eval_loss" in sub.columns:
                y = pd.to_numeric(sub["eval_loss"], errors="coerce")
                m = y.notna()
                if m.any():
                    ax.plot(x[m].values, y[m].values, marker="s", ls="--", lw=1.5, label=f"{label} val", markersize=3)
                    has_curve = True

        if not has_curve:
            plt.close()
            warn(f"loss curves: no data for {ds}")
            continue

        ax.set_title(f"Training & Validation Loss — {ds_label(ds)}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        paths.append(save(f"loss_curves_{ds}", out))
    return paths


# ─── 12. Ablation Loss Comparison (all configs on one plot per dataset) ────────

def plot_ablation_loss_curves(log_dir: Path, out: Path) -> list[Path]:
    paths = []
    abl_log = read_csv_safe(log_dir / "distilbert_ablation_training_logs.csv")
    if abl_log.empty:
        warn("ablation loss curves: no log data")
        return paths
    abl_log["dataset"] = get_ds_col(abl_log)

    for ds in DATASETS:
        sub = abl_log[abl_log["dataset"] == ds] if "dataset" in abl_log.columns else pd.DataFrame()
        if sub.empty or "ablation_config" not in sub.columns:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        has_curve = False
        for cfg in ABLATION_CONFIGS:
            cfg_data = sub[sub["ablation_config"] == cfg]
            if cfg_data.empty:
                continue
            if "eval_loss" not in cfg_data.columns:
                continue
            x = pd.to_numeric(cfg_data.get("step", pd.Series(range(len(cfg_data)))), errors="coerce")
            if x.isna().all():
                x = pd.Series(range(len(cfg_data)), index=cfg_data.index)
            y = pd.to_numeric(cfg_data["eval_loss"], errors="coerce")
            m = y.notna()
            if m.any():
                ax.plot(x[m].values, y[m].values, marker="o", lw=1.3,
                        label=ABLATION_LABELS.get(cfg, cfg), markersize=3,
                        color=ABLATION_CMAP.get(cfg))
                has_curve = True

        if not has_curve:
            plt.close(); continue

        ax.set_title(f"Ablation Validation Loss — {ds_label(ds)}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Validation Loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        paths.append(save(f"ablation_loss_{ds}", out))
    return paths


# ─── 13. Radar / Spider Chart ─────────────────────────────────────────────────

def plot_radar(perf: pd.DataFrame, eff: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if perf.empty:
        return paths
    perf = fix_eval_prefix(perf.copy())
    perf["dataset"] = get_ds_col(perf)

    categories = ["accuracy", "f1_macro", "f1_weighted"]
    cat_labels = ["Accuracy", "Macro F1", "Weighted F1"]

    for ds in DATASETS:
        sub = perf[perf["dataset"] == ds] if "dataset" in perf.columns else pd.DataFrame()
        if sub.empty or "model_type" not in sub.columns:
            continue
        if "latency_ms_per_sample" not in eff.columns:
            # skip radar if no efficiency data for normalization
            continue

        eff_ds = eff.copy()
        eff_ds["dataset"] = get_ds_col(eff_ds)
        eff_sub = eff_ds[eff_ds["dataset"] == ds] if "dataset" in eff_ds.columns else pd.DataFrame()

        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for _, row in sub.iterrows():
            values = []
            for cat in categories:
                v = row.get(cat)
                values.append(float(v) if v is not None and not pd.isna(v) else 0.0)
            values += values[:1]
            ax.plot(angles, values, "o-", lw=2,
                    label=f"{row.get('model_type', '?')} / {row.get('configuration', '?')}",
                    markersize=5)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cat_labels)
        ax.set_title(f"Model Comparison — {ds_label(ds)}", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        paths.append(save(f"radar_{ds}", out))
    return paths


# ─── 14. Freeze Strategy Comparison ───────────────────────────────────────────

def plot_freeze_comparison(abl: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    abl = fix_eval_prefix(abl)
    if abl.empty or "freeze_mode" not in abl.columns:
        warn("freeze comparison: no ablation data with freeze_mode"); return paths
    abl = abl.copy()
    abl["dataset"] = get_ds_col(abl)

    for metric, label in [("f1_macro", "Macro F1"), ("accuracy", "Accuracy")]:
        if metric not in abl.columns:
            continue
        pivot = abl.pivot_table(
            index="dataset", columns="freeze_mode", values=metric, aggfunc="last"
        )
        pivot = pivot.reindex([d for d in DATASETS if d in pivot.index])
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax, width=0.7)
        ax.set_title(f"{label} by Freeze Strategy")
        ax.set_xticklabels([ds_label(d) for d in pivot.index], rotation=0)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Freeze Mode", fontsize=8)
        paths.append(save(f"freeze_{metric}_comparison", out))
    return paths


# ─── 15. Confusion Matrices (best per dataset) ─────────────────────────────────

def plot_confusion_matrices(results_dir: Path, out: Path) -> list[Path]:
    paths = []
    from sklearn.metrics import confusion_matrix as sk_cm, ConfusionMatrixDisplay
    import torch

    for ds in DATASETS:
        config_path = Path(f"config/{ds}.yaml")
        for model_type, model_name in [("distilbert", "distilbert-base-uncased"), ("bert", "bert-base-uncased")]:
            ckpt_dir = results_dir / "checkpoints" / model_type / ds / "best_model"
            if not ckpt_dir.exists():
                continue
            try:
                from src.data import prepare_dataset
                from src.modeling import build_model
                from src.utils import load_yaml_config
                from src.metrics import compute_classification_metrics
                import numpy as np

                ds_config = load_yaml_config(str(config_path))
                _, eval_ds, tokenizer, cfg = prepare_dataset(str(config_path), model_name=model_name)
                from torch.utils.data import DataLoader
                dl = DataLoader(eval_ds, batch_size=64, shuffle=False)

                model = build_model(model_name, int(cfg["num_labels"]))
                sf_path = ckpt_dir / "model.safetensors"
                if not sf_path.exists():
                    continue
                from safetensors.torch import load_file as load_sf
                model.load_state_dict(load_sf(str(sf_path)))
                model.eval()

                all_preds, all_labels = [], []
                with torch.no_grad():
                    for batch in dl:
                        labels = batch.pop("labels").numpy()
                        input_batch = {k: v for k, v in batch.items() if k != "labels"}
                        out = model(**input_batch)
                        all_preds.append(np.argmax(out.logits.numpy(), axis=-1))
                        all_labels.append(labels)

                y_true = np.concatenate(all_labels)
                y_pred = np.concatenate(all_preds)
                cm = sk_cm(y_true, y_pred)

                fig, ax = plt.subplots(figsize=(6, 5))
                disp = ConfusionMatrixDisplay(cm)
                disp.plot(ax=ax, cmap="Blues", values_format="d")
                ax.set_title(f"Confusion Matrix — {model_type.upper()} on {ds_label(ds)}")
                paths.append(save(f"confusion_{model_type}_{ds}", out))
            except Exception as e:
                warn(f"confusion matrix {model_type}/{ds}: {e}")
    return paths


# ─── 16. Metric Summary Table Plot ─────────────────────────────────────────────

def plot_summary_table(perf: pd.DataFrame, eff: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    if perf.empty:
        return paths
    perf = fix_eval_prefix(perf.copy())
    perf["dataset"] = get_ds_col(perf)

    cols = ["dataset", "model_type", "configuration", "accuracy", "f1_macro"]
    avail = [c for c in cols if c in perf.columns]
    if len(avail) < 4:
        return paths
    sub = perf[avail].copy()
    for c in ["accuracy", "f1_macro"]:
        if c in sub.columns:
            sub[c] = sub[c].astype(float).round(4)

    fig, ax = plt.subplots(figsize=(10, max(2, len(sub) * 0.5 + 1)))
    ax.axis("off")
    ax.set_title("Final Performance Summary", fontsize=14, fontweight="bold", pad=20)
    table = ax.table(
        cellText=sub.values,
        colLabels=sub.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    paths.append(save("performance_summary_table", out))

    if not eff.empty:
        eff = eff.copy()
        eff["dataset"] = get_ds_col(eff)
        ecols = ["dataset", "model_type", "configuration", "total_params", "trainable_params", "latency_ms_per_sample"]
        eavail = [c for c in ecols if c in eff.columns]
        if len(eavail) >= 4:
            esub = eff[eavail].copy()
            for c in ["total_params", "trainable_params"]:
                if c in esub.columns:
                    esub[c] = (esub[c].astype(float) / 1e6).round(2).astype(str) + "M"
            for c in ["latency_ms_per_sample"]:
                if c in esub.columns:
                    esub[c] = esub[c].astype(float).round(2)

            fig2, ax2 = plt.subplots(figsize=(10, max(2, len(esub) * 0.5 + 1)))
            ax2.axis("off")
            ax2.set_title("Efficiency Summary", fontsize=14, fontweight="bold", pad=20)
            t2 = ax2.table(cellText=esub.values, colLabels=esub.columns, loc="center", cellLoc="center")
            t2.auto_set_font_size(False)
            t2.set_fontsize(9)
            t2.scale(1.2, 1.5)
            paths.append(save("efficiency_summary_table", out))
    return paths


# ─── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comprehensive experiment visualizations.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()

    results = Path(args.results_dir)
    fig_dir = Path(args.figures_dir)
    metrics_dir = results / "metrics"
    log_dir = results / "logs"

    print(f"Loading data from {results} ...")
    abl = read_csv_safe(metrics_dir / "distilbert_ablation_metrics.csv")
    perf = read_csv_safe(metrics_dir / "final_comparison_metrics.csv")
    eff = read_csv_safe(metrics_dir / "final_efficiency_metrics.csv")
    baseline_metrics = read_csv_safe(metrics_dir / "distilbert_baseline_metrics.csv")
    bert_metrics = read_csv_safe(metrics_dir / "bert_metrics.csv")
    best = read_csv_safe(metrics_dir / "best_distilbert_ablation.csv")
    bench = read_csv_safe(results / "benchmarks" / "benchmark_results.csv")

    merged = pd.DataFrame()
    if not perf.empty and not eff.empty:
        keys = ["dataset", "model_type", "model_name", "configuration"]
        keys = [k for k in keys if k in perf.columns and k in eff.columns]
        if keys:
            merged = perf.merge(eff, on=keys, how="outer")
            merged["plot_label"] = merged.apply(
                lambda r: f"{r.get('model_type', '')}: {r.get('configuration', '')}" if pd.notna(r.get('configuration')) else str(r.get('model_type', 'model')), axis=1
            )

    all_paths: list[Path] = []
    generators = [
        ("F1 & Accuracy bars", lambda: plot_f1_comparison(perf, fig_dir)),
        ("Ablation detail", lambda: plot_ablation_detail(abl, fig_dir)),
        ("Ablation heatmap", lambda: plot_ablation_heatmap(abl, fig_dir)),
        ("Param efficiency", lambda: plot_param_efficiency(abl, merged, fig_dir)),
        ("Latency bars", lambda: plot_latency(eff, fig_dir)),
        ("GPU memory bars", lambda: plot_gpu_memory(eff, fig_dir)),
        ("Model size", lambda: plot_model_size(eff, fig_dir)),
        ("DistilBERT vs BERT", lambda: plot_distilbert_vs_bert(perf, fig_dir)),
        ("Speedup ratio", lambda: plot_speedup(eff, fig_dir)),
        ("Bubble plots", lambda: plot_bubble(merged, fig_dir)),
        ("Loss curves", lambda: plot_loss_curves(log_dir, fig_dir)),
        ("Ablation loss curves", lambda: plot_ablation_loss_curves(log_dir, fig_dir)),
        ("Radar charts", lambda: plot_radar(perf, eff, fig_dir)),
        ("Freeze comparison", lambda: plot_freeze_comparison(abl, fig_dir)),
        ("Confusion matrices", lambda: plot_confusion_matrices(results, fig_dir)),
        ("Summary tables", lambda: plot_summary_table(perf, eff, fig_dir)),
    ]

    for name, gen in generators:
        print(f"\n>>> {name}")
        try:
            paths = gen()
            all_paths.extend(paths)
            print(f"    {len(paths)} figures generated")
        except Exception as exc:
            print(f"    ERROR: {exc}")

    print(f"\nDone! {len(all_paths)} total figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()