from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from tqdm import tqdm

from bacdiff.data.dataset import create_dataloader, prepare_split
from bacdiff.models.model import BacDiffModel
from bacdiff.utils.config import load_config, resolve_run_dir
from bacdiff.utils.runtime import set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate BacDiff on held-out test set.")
    parser.add_argument("--config", required=True, type=str, help="Path to config.yaml")
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields to the target device."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation safely for possibly constant vectors."""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pearsonr(x, y)[0])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation safely for possibly constant vectors."""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(spearmanr(x, y)[0])


def population_distances(true_x: np.ndarray, pred_x: np.ndarray, seed: int) -> tuple[float, float]:
    """Estimate RBF-MMD and energy distance on a bounded random cell sample."""
    rng = np.random.default_rng(seed)
    n = min(512, len(true_x), len(pred_x))
    true_x = true_x[rng.choice(len(true_x), n, replace=False)]
    pred_x = pred_x[rng.choice(len(pred_x), n, replace=False)]
    pooled = np.concatenate([true_x, pred_x], axis=0)
    scale = pooled.std(axis=0).clip(1e-3)
    true_z, pred_z = true_x / scale, pred_x / scale
    all_sq = pdist(np.concatenate([true_z, pred_z]), metric="sqeuclidean")
    positive = all_sq[all_sq > 0]
    gamma = 1.0 / max(float(np.median(positive)) if len(positive) else 1.0, 1e-8)
    tt = np.exp(-gamma * cdist(true_z, true_z, metric="sqeuclidean"))
    pp = np.exp(-gamma * cdist(pred_z, pred_z, metric="sqeuclidean"))
    tp = np.exp(-gamma * cdist(true_z, pred_z, metric="sqeuclidean"))
    mmd = float(tt.mean() + pp.mean() - 2.0 * tp.mean())
    energy = float(2 * cdist(true_z, pred_z).mean() - cdist(true_z, true_z).mean() - cdist(pred_z, pred_z).mean())
    return mmd, energy


@torch.no_grad()
def evaluate_fold(
    model: BacDiffModel,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one fold using cross-cell condition prediction.

    For each test cell i:
      1. Use mean_z_base from train "none" cells (already in model.mean_z_base)
      2. Take test cell's bacteria condition
      3. diffusion(mean_z_base, bacteria_i) → delta_hat → z_hat → decode → pred_expr
      4. Compare pred_expr with test cell's true expression
    """
    model.eval()

    # Collect predictions
    obs_list = []
    pred_list = []
    delta_pred_list = []
    bac_load_list = []
    label_list = []

    for batch in tqdm(test_loader, desc="evaluate", leave=False):
        batch = move_batch_to_device(batch, device)

        x_input = batch["x_input"]
        bac_raw = batch["bac_raw"]
        library_size = batch["library_size"]

        # Cross-cell condition prediction
        result = model.predict(
            x_input=x_input,
            bac_raw=bac_raw,
            library_size=library_size,
            is_none=batch["is_none"],
            num_diffusion_samples=int(config.get("counterfactual", {}).get("num_diffusion_samples", 1)),
        )

        obs = batch["x_counts"].cpu().numpy().astype(np.float32)
        pred = result["recon_mu"].cpu().numpy().astype(np.float32)
        delta_pred = result["delta_pred"].cpu().numpy().astype(np.float32)
        bac_load = batch["bac_total_load"].cpu().numpy().astype(np.float32)
        labels = batch["bacteria_label"]

        obs_list.append(obs)
        pred_list.append(pred)
        delta_pred_list.append(delta_pred)
        bac_load_list.append(bac_load)
        label_list.extend(labels if isinstance(labels, list) else labels.tolist())

    true_expr = np.concatenate(obs_list, axis=0)
    pred_expr = np.concatenate(pred_list, axis=0)
    delta_pred = np.concatenate(delta_pred_list, axis=0)
    bac_total_load = np.concatenate(bac_load_list, axis=0).ravel()
    label_array = np.array(label_list)

    train_data = train_loader.dataset
    train_counts = train_data.x_counts.numpy()
    train_none = train_data.is_none.numpy()
    control_mean_log = np.log1p(train_counts[train_none]).mean(axis=0)

    # 指标在 log1p 空间计算
    # NB loss target = raw counts，所以 mu 也在 raw count 空间
    # 对 pred 和 true 都做 log1p，在同一空间比较
    true_log = np.log1p(true_expr)
    pred_log = np.log1p(pred_expr)

    # Per-group metrics (mean-then-metric, like Squidiff)
    unique_labels = sorted(set(label_list))
    group_metrics = {}

    for label in unique_labels:
        mask = label_array == label
        n = mask.sum()
        if n < 2:
            continue

        true_group = true_log[mask]
        pred_group = pred_log[mask]

        # 群体内求均值，再算指标
        true_mean = true_group.mean(axis=0)
        pred_mean = pred_group.mean(axis=0)

        r2 = r2_score(true_mean, pred_mean)
        pearson_val, _ = pearsonr(true_mean, pred_mean)
        spearman_val, _ = spearmanr(true_mean, pred_mean)
        true_delta = true_mean - control_mean_log
        pred_delta = pred_mean - control_mean_log
        top_idx = np.argsort(np.abs(true_delta))[-min(50, len(true_delta)):]
        top_sign_accuracy = float(np.mean(np.sign(true_delta[top_idx]) == np.sign(pred_delta[top_idx])))
        mmd, energy = population_distances(true_log[mask], pred_log[mask], seed=42)

        group_metrics[label] = {
            "r2": float(r2),
            "pearson": float(pearson_val),
            "spearman": float(spearman_val),
            "rmse": float(np.sqrt(np.mean((true_mean - pred_mean) ** 2))),
            "mae": float(np.mean(np.abs(true_mean - pred_mean))),
            "delta_pearson": safe_pearson(true_delta, pred_delta),
            "top50_de_sign_accuracy": top_sign_accuracy,
            "mmd_rbf": mmd,
            "energy_distance": energy,
            "n_cells": int(n),
        }

    # ---- 按细菌浓度分层的指标（每个细菌组内按 load 分桶） ----
    load_stratified_metrics = {}
    BIN_CONFIGS = [
        (3, ["low", "mid", "high"]),
        (2, ["low", "high"]),
    ]

    for label in unique_labels:
        mask = label_array == label
        n = mask.sum()
        if n < 4:  # 至少 4 个细胞才尝试分桶
            continue

        loads = bac_total_load[mask]
        # 所有细胞 load 相同（如 none 或只检出 1 条的组），无法分桶
        if np.ptp(loads) < 1e-10:
            continue

        # 尝试 3 桶，不够则降级为 2 桶
        for n_bins, bin_labels in BIN_CONFIGS:
            edges = np.percentile(loads, np.linspace(0, 100, n_bins + 1))
            edges[0] -= 1e-6
            edges[-1] += 1e-6

            # 检查每桶是否有足够细胞
            valid_bins = 0
            for i in range(n_bins):
                if i == n_bins - 1:
                    bin_count = ((loads >= edges[i]) & (loads <= edges[i + 1])).sum()
                else:
                    bin_count = ((loads >= edges[i]) & (loads < edges[i + 1])).sum()
                if bin_count >= 2:
                    valid_bins += 1

            if valid_bins >= 2:
                break  # 当前 n_bins 可用

        for i in range(n_bins):
            if i == n_bins - 1:
                bin_mask_local = (loads >= edges[i]) & (loads <= edges[i + 1])
            else:
                bin_mask_local = (loads >= edges[i]) & (loads < edges[i + 1])
            n_bin = bin_mask_local.sum()
            if n_bin < 2:
                continue

            true_bin = true_log[mask][bin_mask_local]
            pred_bin = pred_log[mask][bin_mask_local]

            true_mean = true_bin.mean(axis=0)
            pred_mean = pred_bin.mean(axis=0)

            key = f"{label}__{bin_labels[i]}"
            load_stratified_metrics[key] = {
                "label": label,
                "load_bin": bin_labels[i],
                "load_range": [float(edges[i]), float(edges[i + 1])],
                "r2": float(r2_score(true_mean, pred_mean)),
                "pearson": float(pearsonr(true_mean, pred_mean)[0]),
                "spearman": float(spearmanr(true_mean, pred_mean)[0]),
                "n_cells": int(n_bin),
                "mean_load": float(loads[bin_mask_local].mean()),
            }

    # ---- Per-cell 指标 vs 细菌浓度的连续相关性 ----
    # 对每个细胞，计算其基因表达的 per-cell Pearson（true vs pred）
    per_cell_corr = np.full(len(true_log), np.nan, dtype=np.float32)
    for i in range(len(true_log)):
        t, p = true_log[i], pred_log[i]
        if np.std(t) < 1e-12 or np.std(p) < 1e-12:
            continue
        per_cell_corr[i] = float(np.corrcoef(t, p)[0, 1])

    valid = ~np.isnan(per_cell_corr)
    if valid.sum() >= 3:
        corr_quality_vs_load = float(np.corrcoef(per_cell_corr[valid], bac_total_load[valid])[0, 1])
        spearman_quality_vs_load = safe_spearman(per_cell_corr[valid], bac_total_load[valid])
    else:
        corr_quality_vs_load = float("nan")
        spearman_quality_vs_load = float("nan")

    # 全局指标（所有细胞混合）
    true_mean_all = true_log.mean(axis=0)
    pred_mean_all = pred_log.mean(axis=0)
    global_r2 = r2_score(true_mean_all, pred_mean_all)
    global_pearson, _ = pearsonr(true_mean_all, pred_mean_all)
    global_spearman, _ = spearmanr(true_mean_all, pred_mean_all)

    # Delta norm vs bacteria load
    delta_norm = np.linalg.norm(delta_pred, axis=1)
    delta_vs_load_spearman = safe_spearman(delta_norm, bac_total_load)

    # 汇总：各组指标的均值
    if group_metrics:
        avg_r2 = np.mean([m["r2"] for m in group_metrics.values()])
        avg_pearson = np.mean([m["pearson"] for m in group_metrics.values()])
        avg_spearman = np.mean([m["spearman"] for m in group_metrics.values()])
    else:
        avg_r2 = avg_pearson = avg_spearman = float("nan")

    # 汇总：浓度分层指标
    if load_stratified_metrics:
        avg_r2_strat = np.mean([m["r2"] for m in load_stratified_metrics.values()])
        avg_pearson_strat = np.mean([m["pearson"] for m in load_stratified_metrics.values()])
        avg_spearman_strat = np.mean([m["spearman"] for m in load_stratified_metrics.values()])
    else:
        avg_r2_strat = avg_pearson_strat = avg_spearman_strat = float("nan")

    metrics = {
        "r2_global": float(global_r2),
        "pearson_global": float(global_pearson),
        "spearman_global": float(global_spearman),
        "r2_per_group_mean": float(avg_r2),
        "pearson_per_group_mean": float(avg_pearson),
        "spearman_per_group_mean": float(avg_spearman),
        "r2_per_load_stratum_mean": float(avg_r2_strat),
        "pearson_per_load_stratum_mean": float(avg_pearson_strat),
        "spearman_per_load_stratum_mean": float(avg_spearman_strat),
        "per_cell_corr_vs_load_pearson": corr_quality_vs_load,
        "per_cell_corr_vs_load_spearman": spearman_quality_vs_load,
        "delta_norm_vs_load_spearman": delta_vs_load_spearman,
        "mean_per_cell_pearson": float(np.nanmean(per_cell_corr)),
        "rmse_per_group_mean": float(np.mean([m["rmse"] for m in group_metrics.values()])),
        "mae_per_group_mean": float(np.mean([m["mae"] for m in group_metrics.values()])),
        "delta_pearson_per_group_mean": float(np.nanmean([m["delta_pearson"] for m in group_metrics.values()])),
        "top50_de_sign_accuracy_mean": float(np.mean([m["top50_de_sign_accuracy"] for m in group_metrics.values()])),
        "mmd_rbf_per_group_mean": float(np.mean([m["mmd_rbf"] for m in group_metrics.values()])),
        "energy_distance_per_group_mean": float(np.mean([m["energy_distance"] for m in group_metrics.values()])),
        "n_test_cells": int(true_expr.shape[0]),
        "n_groups": len(group_metrics),
        "n_load_strata": len(load_stratified_metrics),
    }

    return {
        "metrics": metrics,
        "group_metrics": group_metrics,
        "load_stratified_metrics": load_stratified_metrics,
        "true_expr": true_expr,
        "pred_expr": pred_expr,
        "delta_pred": delta_pred,
        "bac_total_load": bac_total_load,
        "per_cell_corr": per_cell_corr,
    }


def save_true_pred_scatter(true_expr: np.ndarray, pred_expr: np.ndarray, path: Path, seed: int) -> None:
    """Save a scatter plot comparing flattened true and predicted expression (log1p space)."""
    true_log = np.log1p(true_expr)
    pred_log = np.log1p(pred_expr)

    rng = np.random.default_rng(seed)
    n = true_log.ravel().shape[0]
    max_points = 20000
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        true_flat = true_log.ravel()[idx]
        pred_flat = pred_log.ravel()[idx]
    else:
        true_flat = true_log.ravel()
        pred_flat = pred_log.ravel()

    plt.figure(figsize=(6, 6))
    plt.scatter(true_flat, pred_flat, s=6, alpha=0.25)
    max_val = float(max(true_flat.max(), pred_flat.max(), 1.0))
    plt.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1.0, color="black")
    plt.xlabel("True expression (log1p)")
    plt.ylabel("Predicted expression (log1p)")
    plt.title("True vs Predicted Expression (log1p)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_quality_vs_load_scatter(
    per_cell_corr: np.ndarray,
    bac_total_load: np.ndarray,
    path: Path,
    seed: int,
) -> None:
    """Scatter plot of per-cell prediction quality vs bacterial load."""
    valid = ~np.isnan(per_cell_corr)
    if valid.sum() < 3:
        return

    x = bac_total_load[valid]
    y = per_cell_corr[valid]

    rng = np.random.default_rng(seed)
    n = x.shape[0]
    max_points = 10000
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        x, y = x[idx], y[idx]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=6, alpha=0.2)
    # 加一条 LOWESS 趋势线
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y, x, frac=0.3, return_sorted=True)
        plt.plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=2, label="LOWESS")
        plt.legend()
    except ImportError:
        pass
    plt.xlabel("Bacterial total load (log1p)")
    plt.ylabel("Per-cell Pearson (true vs pred)")
    plt.title("Prediction Quality vs Bacterial Load")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    """Run evaluation on the held-out test set."""
    args = parse_args()
    config = load_config(args.config)
    run_dir = resolve_run_dir(config)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["training"]["seed"])
    set_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare single 7:1:2 split
    train_dataset, val_dataset, test_dataset, split_info = prepare_split(config)
    n_genes = train_dataset.x_input.shape[1]
    n_bacteria = train_dataset.bac_raw.shape[1]

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])

    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}. Run train.py first.")

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load model
    model = BacDiffModel(
        input_dim=n_genes,
        bacteria_dim=n_bacteria,
        config=config,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore mean_z_base from checkpoint
    if "mean_z_base" in checkpoint:
        model.mean_z_base.copy_(checkpoint["mean_z_base"].to(device))
    else:
        # Recompute from train none cells
        _compute_and_store_mean_z_base(model, train_loader, device)

    print(f"\n{'='*60}")
    print(f"Evaluating on test set ({split_info['test_size']} cells, {split_info['test_none']} none)")
    print(f"{'='*60}")

    result = evaluate_fold(model, train_loader, test_loader, device, config)
    metrics = result["metrics"]
    group_metrics = result["group_metrics"]
    load_stratified_metrics = result["load_stratified_metrics"]

    # 打印全局指标
    print(f"  Global: R2={metrics['r2_global']:.4f}, "
          f"Pearson={metrics['pearson_global']:.4f}, "
          f"Spearman={metrics['spearman_global']:.4f}")

    # 打印各组指标
    print(f"  Per-group mean: R2={metrics['r2_per_group_mean']:.4f}, "
          f"Pearson={metrics['pearson_per_group_mean']:.4f}, "
          f"Spearman={metrics['spearman_per_group_mean']:.4f}")
    print(f"  {'Group':50s} {'R2':>8s} {'Pearson':>8s} {'Spearman':>8s} {'N':>6s}")
    print(f"  {'-'*80}")
    for label, gm in sorted(group_metrics.items(), key=lambda x: -x[1]["n_cells"]):
        print(f"  {label:50s} {gm['r2']:8.4f} {gm['pearson']:8.4f} {gm['spearman']:8.4f} {gm['n_cells']:6d}")

    # 打印浓度分层指标
    if load_stratified_metrics:
        print(f"\n  Load-stratified mean: R2={metrics['r2_per_load_stratum_mean']:.4f}, "
              f"Pearson={metrics['pearson_per_load_stratum_mean']:.4f}, "
              f"Spearman={metrics['spearman_per_load_stratum_mean']:.4f}")
        print(f"  {'Label__Load':50s} {'R2':>8s} {'Pearson':>8s} {'Spearman':>8s} {'N':>6s} {'MeanLoad':>10s}")
        print(f"  {'-'*92}")
        for key, sm in sorted(load_stratified_metrics.items(), key=lambda x: -x[1]["n_cells"]):
            print(f"  {key:50s} {sm['r2']:8.4f} {sm['pearson']:8.4f} {sm['spearman']:8.4f} "
                  f"{sm['n_cells']:6d} {sm['mean_load']:10.2f}")

    # 打印 per-cell 质量 vs 浓度相关性
    print(f"\n  Per-cell quality vs load: Pearson={metrics['per_cell_corr_vs_load_pearson']:.4f}, "
          f"Spearman={metrics['per_cell_corr_vs_load_spearman']:.4f}")
    print(f"  Mean per-cell Pearson: {metrics['mean_per_cell_pearson']:.4f}")

    # 保存分组指标
    group_df = pd.DataFrame([
        {"group": label, **gm} for label, gm in group_metrics.items()
    ])
    group_df.to_csv(eval_dir / "group_metrics.csv", index=False)

    # 保存浓度分层指标
    if load_stratified_metrics:
        strat_df = pd.DataFrame([
            {"stratum": key, **sm} for key, sm in load_stratified_metrics.items()
        ])
        strat_df.to_csv(eval_dir / "load_stratified_metrics.csv", index=False)

    # 保存指标汇总
    with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    # Save plots
    save_true_pred_scatter(
        result["true_expr"], result["pred_expr"],
        eval_dir / "true_vs_predicted.png", seed=seed,
    )
    save_quality_vs_load_scatter(
        result["per_cell_corr"], result["bac_total_load"],
        eval_dir / "quality_vs_load.png", seed=seed,
    )

    print(f"\nResults saved to: {eval_dir}")


@torch.no_grad()
def _compute_and_store_mean_z_base(
    model: BacDiffModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """Compute mean z_base from none cells in the loader and store in model."""
    model.eval()
    z_means = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        is_none = batch["is_none"]
        if is_none.any():
            x_input = batch["x_input"][is_none]
            encoded = model.rna_encoder(x_input=x_input, sample=False)
            z_means.append(encoded["z_mean"].cpu())
    if z_means:
        all_z = torch.cat(z_means, dim=0)
        model.mean_z_base.copy_(all_z.mean(dim=0).to(device))


if __name__ == "__main__":
    main()
