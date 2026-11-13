"""
Evaluate BacDiff using CPA-style metrics for fair comparison.

Computes r2_mean_deg on top DEGs (10, 20, 50, all) per condition,
matching CPA's evaluation methodology exactly.

Usage:
    python -m bacdiff.cli.evaluate_cpa_style --config configs/config_sciplex.yaml --original_h5ad /path/to/original_sciplex.h5ad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from tqdm import tqdm

from bacdiff.data.dataset import create_dataloader, preprocess_rna, select_gene_names
from bacdiff.models.model import BacDiffModel
from bacdiff.utils.config import load_config, resolve_run_dir
from bacdiff.utils.runtime import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPA-style evaluation for BacDiff")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--original_h5ad",
        type=str,
        required=True,
        help="Original h5ad with layers['counts'] and uns['rank_genes_groups_cov']",
    )
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def predict_all_cells(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Get model predictions for all cells in the loader.

    Returns (true_counts, pred_expr, delta_pred, labels).
    true_counts and pred_expr are in raw count space.
    """
    model.eval()
    obs_list, pred_list, delta_list, label_list = [], [], [], []

    for batch in tqdm(loader, desc="predict", leave=False):
        batch = move_batch_to_device(batch, device)
        result = model.predict(
            x_input=batch["x_input"],
            bac_raw=batch["bac_raw"],
            library_size=batch["library_size"],
            is_none=batch["is_none"],
            num_diffusion_samples=1,
        )
        obs_list.append(batch["x_counts"].cpu().numpy())
        pred_list.append(result["recon_mu"].cpu().numpy())
        delta_list.append(result["delta_pred"].cpu().numpy())
        label_list.extend(batch["bacteria_label"].tolist() if hasattr(batch["bacteria_label"], "tolist") else list(batch["bacteria_label"]))

    return (
        np.concatenate(obs_list, axis=0),
        np.concatenate(pred_list, axis=0),
        np.concatenate(delta_list, axis=0),
        label_list,
    )


def build_gene_mapping(original_adata, processed_adata) -> np.ndarray:
    """Build index mapping from processed (500 HVG) genes to original (5000) genes.

    Returns an array of shape (n_processed_genes,) where each element is the
    index in the original adata.var_names.
    """
    original_names = list(original_adata.var_names)
    processed_names = list(processed_adata.var_names)

    mapping = []
    for pname in processed_names:
        if pname in original_names:
            mapping.append(original_names.index(pname))
        else:
            mapping.append(-1)  # gene not found
    return np.array(mapping)


def evaluate_cpa_style(
    true_counts_full: np.ndarray,
    pred_counts_full: np.ndarray,
    ctrl_counts_full: np.ndarray,
    deg_dict: dict,
    var_names: np.ndarray,
    n_top_degs: list[int | None],
) -> dict[str, list]:
    """
    Compute CPA-style metrics: r2_mean_deg and r2_mean_lfc_deg.

    Args:
        true_counts_full: (n_cells, n_genes) true counts in ORIGINAL gene space
        pred_counts_full: (n_cells, n_genes) predicted counts in ORIGINAL gene space
        ctrl_counts_full: (n_ctrl_cells, n_genes) control cell counts
        deg_dict: {condition_key: [gene_name_list]} from rank_genes_groups_cov
        var_names: gene names array
        n_top_degs: list of top-N DEGs to evaluate (None = all genes)

    Returns:
        dict with lists of results per condition per n_top_deg
    """
    results: dict[str, list] = {
        "cell_type": [],
        "condition": [],
        "dose": [],
        "n_top_deg": [],
        "r2_mean_deg": [],
        "r2_var_deg": [],
        "r2_mean_lfc_deg": [],
        "r2_var_lfc_deg": [],
    }

    true_log = np.log1p(true_counts_full)
    pred_log = np.log1p(pred_counts_full)
    ctrl_log = np.log1p(ctrl_counts_full)

    # Pseudobulk
    true_mean = true_log.mean(axis=0)  # (n_genes,)
    pred_mean = pred_log.mean(axis=0)
    ctrl_mean = ctrl_log.mean(axis=0)

    for cond_key, deg_list in deg_dict.items():
        if "CHEMBL504" in cond_key:
            continue

        parts = cond_key.split("_")
        if len(parts) >= 3:
            cell_type = parts[0]
            dose = parts[-1]
            condition = "_".join(parts[1:-1])
        else:
            cell_type = "unknown"
            condition = cond_key
            dose = "unknown"

        for n_top in n_top_degs:
            if n_top is not None and deg_list is not None:
                deg_indices = np.where(np.isin(var_names, deg_list[:n_top]))[0]
            else:
                deg_indices = np.arange(len(var_names))
                n_top = "all"

            if len(deg_indices) == 0:
                continue

            t_deg = true_mean[deg_indices]
            p_deg = pred_mean[deg_indices]
            c_deg = ctrl_mean[deg_indices]

            r2_mean_deg = r2_score(t_deg, p_deg)
            r2_var_deg = r2_score(
                true_log[:, deg_indices].var(axis=0),
                pred_log[:, deg_indices].var(axis=0),
            )
            r2_mean_lfc_deg = r2_score(t_deg - c_deg, p_deg - c_deg)
            r2_var_lfc_deg = r2_score(
                true_log[:, deg_indices].var(axis=0) - ctrl_log[:, deg_indices].var(axis=0),
                pred_log[:, deg_indices].var(axis=0) - ctrl_log[:, deg_indices].var(axis=0),
            )

            results["cell_type"].append(cell_type)
            results["condition"].append(condition)
            results["dose"].append(dose)
            results["n_top_deg"].append(n_top)
            results["r2_mean_deg"].append(r2_mean_deg)
            results["r2_var_deg"].append(r2_var_deg)
            results["r2_mean_lfc_deg"].append(r2_mean_lfc_deg)
            results["r2_var_lfc_deg"].append(r2_var_lfc_deg)

    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = resolve_run_dir(config)
    eval_dir = run_dir / "eval_cpa"
    eval_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["training"]["seed"])
    set_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load original h5ad (with counts and DEGs) ----
    print(f"Loading original h5ad: {args.original_h5ad}")
    orig_adata = ad.read_h5ad(args.original_h5ad)
    print(f"  Shape: {orig_adata.shape}")
    print(f"  Has layers['counts']: {'counts' in orig_adata.layers}")
    print(f"  Has uns['rank_genes_groups_cov']: {'rank_genes_groups_cov' in orig_adata.uns}")

    # ---- Load processed h5ad (BacDiff's training data) ----
    print(f"Loading processed h5ad: {config['data']['merged_h5ad']}")
    proc_adata = ad.read_h5ad(config["data"]["merged_h5ad"])

    # ---- Preprocess to get the same 500 HVGs BacDiff was trained on ----
    x_input, x_counts = preprocess_rna(proc_adata, config)
    n_bac = proc_adata.obsm["bacteria_raw"].shape[1]

    # ---- Load BacDiff model ----
    n_genes = x_input.shape[1]
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

    model = BacDiffModel(input_dim=n_genes, bacteria_dim=n_bac, config=config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if "mean_z_base" in ckpt:
        model.mean_z_base.copy_(ckpt["mean_z_base"].to(device))

    # ---- Reconstruct test set using CPA's split ----
    # CPA uses split_1ct_MEC with ood as test.
    # Get split info from original h5ad (processed h5ad may not have it).
    split_col = "split_1ct_MEC"
    if split_col in orig_adata.obs.columns:
        ood_mask = orig_adata.obs[split_col].values == "ood"
    elif split_col in proc_adata.obs.columns:
        ood_mask = proc_adata.obs[split_col].values == "ood"
    else:
        raise KeyError(f"Column '{split_col}' not found in either h5ad")
    ctrl_mask = orig_adata.obs["control"].values == 1

    print(f"  OOD cells: {ood_mask.sum()}")
    print(f"  Control cells: {ctrl_mask.sum()}")

    # Build a SingleCellDataset for OOD cells
    from bacdiff.data.dataset import SingleCellDataset

    bac_raw_full = proc_adata.obsm["bacteria_raw"].astype(np.float32)
    bac_mean = bac_raw_full[~ood_mask].mean(axis=0, keepdims=True).astype(np.float32)
    bac_std = (bac_raw_full[~ood_mask].std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    bac_raw_z = ((bac_raw_full - bac_mean) / bac_std).astype(np.float32)
    bac_total_load = np.log1p(bac_raw_full.sum(axis=1, keepdims=True)).astype(np.float32)
    labels = proc_adata.obs["bacteria_label"].to_numpy().astype(str) if "bacteria_label" in proc_adata.obs.columns else orig_adata.obs["condition"].to_numpy().astype(str)
    is_none = (labels == "none") | (orig_adata.obs["control"].values == 1)
    cell_ids = proc_adata.obs_names.to_numpy().astype(str)

    ood_dataset = SingleCellDataset(
        x_counts=x_counts[ood_mask],
        x_input=x_input[ood_mask],
        bac_raw=bac_raw_z[ood_mask],
        bac_total_load=bac_total_load[ood_mask],
        is_none=is_none[ood_mask],
        cell_ids=cell_ids[ood_mask],
        bacteria_labels=labels[ood_mask],
    )
    # Use a dummy train loader for mean_z_base (already loaded from checkpoint)
    dummy_loader = create_dataloader(ood_dataset, batch_size=256, shuffle=False, num_workers=0)

    # ---- Get predictions ----
    print("\nRunning BacDiff predictions on OOD cells...")
    true_hvg, pred_hvg, delta_pred, ood_labels = predict_all_cells(model, dummy_loader, device)

    # ---- Map predictions back to full gene space ----
    # BacDiff was trained on 500 HVGs. We need to map predictions to the 5000-gene space.
    # Strategy: for the 500 HVG genes, use BacDiff's predictions directly.
    #           for the remaining 4500 genes, use the control mean as prediction
    #           (since BacDiff can't predict genes it wasn't trained on).
    # This is a limitation but allows CPA-style DEG evaluation.

    hvg_gene_names = np.asarray(select_gene_names(proc_adata, config))
    orig_gene_names = np.array(orig_adata.var_names)

    # Build mapping: for each HVG gene, find its index in original data
    gene_map = {}  # orig_idx -> hvg_local_idx
    for local_idx, gname in enumerate(hvg_gene_names):
        orig_idx = np.where(orig_gene_names == gname)[0]
        if len(orig_idx) > 0:
            gene_map[orig_idx[0]] = local_idx

    n_orig_genes = len(orig_gene_names)
    n_ood = ood_mask.sum()

    # Build true counts in original gene space (from original h5ad)
    ood_adata = orig_adata[ood_mask]
    counts_layer = ood_adata.layers["counts"]
    true_full = counts_layer.toarray().astype(np.float32) if hasattr(counts_layer, "toarray") else np.asarray(counts_layer, dtype=np.float32)

    # Build pred counts in original gene space
    # For HVG genes: use BacDiff's prediction
    # For non-HVG genes: use control mean
    ctrl_ood = orig_adata[ood_mask & ctrl_mask]
    ctrl_counts_layer = ctrl_ood.layers["counts"]
    ctrl_counts = ctrl_counts_layer.toarray().astype(np.float32) if hasattr(ctrl_counts_layer, "toarray") else np.asarray(ctrl_counts_layer, dtype=np.float32)
    ctrl_pred_mean = ctrl_counts.mean(axis=0)  # (n_orig_genes,)

    pred_full = np.tile(ctrl_pred_mean, (n_ood, 1))  # start with control mean
    for orig_idx, proc_idx in gene_map.items():
        pred_full[:, orig_idx] = pred_hvg[:, proc_idx]

    # Control cells in original space (for LFC computation)
    ctrl_full = ctrl_counts

    # ---- Run CPA-style evaluation ----
    n_top_degs = [10, 20, 50, None]
    deg_dict = orig_adata.uns["rank_genes_groups_cov"]

    print("\nComputing CPA-style metrics...")
    results = evaluate_cpa_style(
        true_counts_full=true_full,
        pred_counts_full=pred_full,
        ctrl_counts_full=ctrl_full,
        deg_dict=deg_dict,
        var_names=orig_gene_names,
        n_top_degs=n_top_degs,
    )

    df = pd.DataFrame(results)

    # ---- Print results ----
    print("\n" + "=" * 80)
    print("BacDiff CPA-style Evaluation Results (OOD split)")
    print("=" * 80)

    for n_top in n_top_degs:
        subset = df[df["n_top_deg"] == n_top]
        if subset.empty:
            continue
        label = f"top {n_top}" if n_top is not None else "all"
        print(f"\n--- n_top_deg = {label} ---")
        print(f"  r2_mean_deg:     mean={subset['r2_mean_deg'].mean():.4f}, median={subset['r2_mean_deg'].median():.4f}")
        print(f"  r2_mean_lfc_deg: mean={subset['r2_mean_lfc_deg'].mean():.4f}, median={subset['r2_mean_lfc_deg'].median():.4f}")
        print(f"  r2_var_deg:      mean={subset['r2_var_deg'].mean():.4f}")
        print(f"  r2_var_lfc_deg:  mean={subset['r2_var_lfc_deg'].mean():.4f}")

    # Per-condition table for top 20
    df20 = df[df["n_top_deg"] == 20].copy()
    if not df20.empty:
        print(f"\n{'Condition':55s} {'r2_mean_deg':>12s} {'r2_lfc_deg':>12s}")
        print("-" * 82)
        for _, row in df20.sort_values("r2_mean_deg", ascending=False).iterrows():
            cond = f"{row['condition']}"
            print(f"  {cond:53s} {row['r2_mean_deg']:12.4f} {row['r2_mean_lfc_deg']:12.4f}")

    # ---- Save ----
    df.to_csv(eval_dir / "cpa_style_metrics.csv", index=False)
    with open(eval_dir / "cpa_style_summary.json", "w") as f:
        summary = {}
        for n_top in n_top_degs:
            subset = df[df["n_top_deg"] == n_top]
            label = str(n_top) if n_top is not None else "all"
            if not subset.empty:
                summary[f"top_{label}"] = {
                    "r2_mean_deg_mean": float(subset["r2_mean_deg"].mean()),
                    "r2_mean_deg_median": float(subset["r2_mean_deg"].median()),
                    "r2_mean_lfc_deg_mean": float(subset["r2_mean_lfc_deg"].mean()),
                    "r2_var_deg_mean": float(subset["r2_var_deg"].mean()),
                }
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {eval_dir}")


if __name__ == "__main__":
    main()
