from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from bacdiff.data.dataset import create_dataloader, prepare_datasets
from bacdiff.models.model import BacDiffModel
from bacdiff.utils.config import load_config, resolve_run_dir
from bacdiff.utils.runtime import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for counterfactual analysis."""
    parser = argparse.ArgumentParser(description="Run counterfactual inference and dose scanning.")
    parser.add_argument("--config", required=True, type=str, help="Path to config.yaml")
    parser.add_argument("--cell-indices", nargs="*", default=None, help="Optional dataset-relative cell indices.")
    parser.add_argument("--sample-n", type=int, default=None, help="Randomly sample N cells from the selected split.")
    parser.add_argument("--gene-list", nargs="*", default=None, help="Optional gene names for dose-curve plotting.")
    parser.add_argument("--bacteria-list", nargs="*", default=None, help="Optional bacterial feature names to keep.")
    parser.add_argument("--dose-scales", nargs="*", type=float, default=None, help="Dose scales for total-load scanning.")
    return parser.parse_args()


def normalize_str_list(values: list[str] | None) -> list[str]:
    """Normalize a list of CLI strings supporting comma-separated input."""
    if not values:
        return []
    normalized: list[str] = []
    for value in values:
        normalized.extend([part for part in str(value).split(",") if part])
    return normalized


def normalize_int_list(values: list[str] | None) -> list[int]:
    """Normalize optional CLI integers supporting comma-separated input."""
    if not values:
        return []
    normalized: list[int] = []
    for value in values:
        normalized.extend(int(part) for part in str(value).split(",") if part)
    return normalized


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move batch tensors to device."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def select_dataset(prepared: Any, split_name: str) -> Any:
    """Select a dataset split by name."""
    return {
        "train": prepared.train_dataset,
        "val": prepared.val_dataset,
        "test": prepared.test_dataset,
        "all": prepared.full_dataset,
    }[split_name]


def select_indices(dataset_size: int, args: argparse.Namespace, config: dict[str, Any]) -> list[int]:
    """Resolve which dataset indices to analyze."""
    explicit = normalize_int_list(args.cell_indices)
    if explicit:
        valid = [idx for idx in explicit if 0 <= idx < dataset_size]
        if not valid:
            raise ValueError("No valid --cell-indices remain after bounds checking.")
        return valid

    if args.sample_n is not None:
        sample_n = min(int(args.sample_n), dataset_size)
        rng = np.random.default_rng(int(config["training"]["seed"]))
        return sorted(rng.choice(dataset_size, size=sample_n, replace=False).tolist())

    max_cells = min(int(config["counterfactual"]["max_cells"]), dataset_size)
    return list(range(max_cells))


def compute_composition(raw: np.ndarray) -> np.ndarray:
    """Convert raw bacterial abundance to per-cell composition."""
    row_sum = raw.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
    return (raw / row_sum).astype(np.float32)


def build_bacteria_batch_from_raw(raw: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    """Build the bacteria-only batch fields expected by the model."""
    raw = np.asarray(raw, dtype=np.float32)
    return {
        "bac_raw": torch.from_numpy(raw).to(device),
        "bac_presence": torch.from_numpy((raw > 0).astype(np.float32)).to(device),
        "bac_composition": torch.from_numpy(compute_composition(raw)).to(device),
        "bac_total_load": torch.from_numpy(np.log1p(raw.sum(axis=1, keepdims=True)).astype(np.float32)).to(device),
    }


def apply_bacteria_filter(raw: np.ndarray, keep_indices: list[int]) -> np.ndarray:
    """Optionally keep only selected bacterial features."""
    if not keep_indices:
        return raw.astype(np.float32)
    filtered = np.zeros_like(raw, dtype=np.float32)
    filtered[:, keep_indices] = raw[:, keep_indices]
    return filtered


def build_scaled_condition(raw_selected: np.ndarray, dose: float) -> np.ndarray:
    """Scale bacterial total load while preserving composition."""
    return (raw_selected * dose).astype(np.float32)


def compute_gene_level_effects(
    gene_names: list[str],
    pred_zero: np.ndarray,
    pred_observed: np.ndarray,
    dose_scales: list[float],
    dose_predictions: dict[float, np.ndarray],
) -> pd.DataFrame:
    """Compute per-gene counterfactual effect summaries."""
    effect = pred_observed - pred_zero
    rows: list[dict[str, float | str]] = []
    sorted_doses = sorted(dose_scales)
    dose_means = np.stack([dose_predictions[dose].mean(axis=0) for dose in sorted_doses], axis=0)

    for gene_idx, gene in enumerate(gene_names):
        y = dose_means[:, gene_idx]
        if len(sorted_doses) >= 2:
            slope = float(np.polyfit(sorted_doses, y, deg=1)[0])
        else:
            slope = 0.0
        gene_effect = effect[:, gene_idx]
        rows.append(
            {
                "gene": gene,
                "mean_effect": float(gene_effect.mean()),
                "abs_mean_effect": float(np.abs(gene_effect).mean()),
                "dose_slope": slope,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_mean_effect", ascending=False).reset_index(drop=True)


def save_gene_dose_plot(
    dose_scales: list[float],
    dose_predictions: dict[float, np.ndarray],
    gene_names: list[str],
    selected_genes: list[str],
    path: Path,
) -> None:
    """Plot mean predicted expression versus dose for selected genes."""
    if not selected_genes:
        return
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    plt.figure(figsize=(8, 5))
    for gene in selected_genes:
        if gene not in gene_to_idx:
            continue
        idx = gene_to_idx[gene]
        y = [dose_predictions[dose][:, idx].mean() for dose in dose_scales]
        plt.plot(dose_scales, y, marker="o", label=gene)
    plt.xlabel("Dose scale")
    plt.ylabel("Predicted expression")
    plt.title("Gene Dose Curves")
    if len(selected_genes) <= 10:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_cell_delta_plot(
    dose_long_df: pd.DataFrame,
    path: Path,
    max_cells_to_draw: int = 25,
) -> None:
    """Plot cell delta norms across dose scales."""
    plt.figure(figsize=(8, 5))
    dose_summary = dose_long_df.groupby("dose_scale")["delta_norm"].agg(["mean", "std"]).reset_index()
    plt.plot(dose_summary["dose_scale"], dose_summary["mean"], color="black", linewidth=2.0, label="Mean")
    plt.fill_between(
        dose_summary["dose_scale"],
        dose_summary["mean"] - dose_summary["std"],
        dose_summary["mean"] + dose_summary["std"],
        alpha=0.2,
        color="black",
    )
    for cell_id, frame in list(dose_long_df.groupby("cell_id"))[:max_cells_to_draw]:
        plt.plot(frame["dose_scale"], frame["delta_norm"], alpha=0.2, linewidth=1.0)
    plt.xlabel("Dose scale")
    plt.ylabel("Delta norm")
    plt.title("Cell Delta Norm Across Dose")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    """Run counterfactual inference and dose scanning."""
    args = parse_args()
    config = load_config(args.config)
    run_dir = resolve_run_dir(config)
    cf_dir = run_dir / "counterfactual"
    cf_dir.mkdir(parents=True, exist_ok=True)

    preprocess_path = run_dir / config["output"]["preprocess_name"]
    checkpoint_path = run_dir / "best_model.pt"

    seed = int(config["training"]["seed"])
    set_seed(seed, deterministic=bool(config["training"]["deterministic"]))
    device = resolve_device(str(config["training"]["device"]))

    prepared = prepare_datasets(config, fit=False, preprocess_path=preprocess_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BacDiffModel(
        input_dim=len(prepared.gene_names),
        bacteria_dim=len(prepared.bacteria_names),
        config=config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    split_name = str(config["counterfactual"]["split"])
    dataset = select_dataset(prepared, split_name)
    selected_indices = select_indices(len(dataset), args, config)
    subset = torch.utils.data.Subset(dataset, selected_indices)
    loader = create_dataloader(
        subset,
        batch_size=int(config["counterfactual"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )

    bacteria_names = prepared.bacteria_names
    gene_names = prepared.gene_names
    selected_bacteria = normalize_str_list(args.bacteria_list)
    keep_bacteria_indices = [bacteria_names.index(name) for name in selected_bacteria if name in bacteria_names]
    dose_scales = list(args.dose_scales) if args.dose_scales else [float(x) for x in config["counterfactual"]["dose_grid"]]
    num_samples = int(config["counterfactual"].get("num_diffusion_samples", 1))

    selected_gene_names = normalize_str_list(args.gene_list)
    observed_expr_list: list[np.ndarray] = []
    zero_expr_list: list[np.ndarray] = []
    observed_delta_list: list[np.ndarray] = []
    zero_delta_list: list[np.ndarray] = []
    dose_delta_store: dict[float, list[np.ndarray]] = {dose: [] for dose in dose_scales}
    dose_expr_store: dict[float, list[np.ndarray]] = {dose: [] for dose in dose_scales}
    cell_rows: list[dict[str, float | str | int]] = []
    dose_long_rows: list[dict[str, float | str | int]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="counterfactual"):
            raw_cell_ids = [str(x) for x in batch["cell_id"]]
            raw_cell_index = batch["cell_index"].numpy().tolist()
            observed_raw = batch["bac_raw"].numpy().astype(np.float32)
            observed_selected_raw = apply_bacteria_filter(observed_raw, keep_bacteria_indices)

            batch_device = move_batch_to_device(batch, device)

            zero_bacteria_batch = build_bacteria_batch_from_raw(np.zeros_like(observed_raw, dtype=np.float32), device=device)
            observed_bacteria_batch = build_bacteria_batch_from_raw(observed_selected_raw, device=device)

            zero_cf = model.counterfactual_forward(
                batch=batch_device,
                new_bacteria_batch=zero_bacteria_batch,
                num_diffusion_samples=num_samples,
            )
            observed_cf = model.counterfactual_forward(
                batch=batch_device,
                new_bacteria_batch=observed_bacteria_batch,
                num_diffusion_samples=num_samples,
            )

            pred_zero = zero_cf["recon_mu"].detach().cpu().numpy().astype(np.float32)
            pred_observed = observed_cf["recon_mu"].detach().cpu().numpy().astype(np.float32)
            delta_zero = zero_cf["delta_cf"].detach().cpu().numpy().astype(np.float32)
            delta_observed = observed_cf["delta_cf"].detach().cpu().numpy().astype(np.float32)
            effect = pred_observed - pred_zero

            observed_expr_list.append(pred_observed)
            zero_expr_list.append(pred_zero)
            observed_delta_list.append(delta_observed)
            zero_delta_list.append(delta_zero)

            for local_idx, cell_id in enumerate(raw_cell_ids):
                cell_rows.append(
                    {
                        "cell_id": cell_id,
                        "cell_index": int(raw_cell_index[local_idx]),
                        "split": split_name,
                        "delta_norm_zero": float(np.linalg.norm(delta_zero[local_idx])),
                        "delta_norm_observed": float(np.linalg.norm(delta_observed[local_idx])),
                        "effect_norm": float(np.linalg.norm(effect[local_idx])),
                        "observed_bacteria_total_load": float(np.log1p(observed_raw[local_idx].sum())),
                    }
                )

            for dose in dose_scales:
                dose_raw = build_scaled_condition(observed_selected_raw, dose)
                dose_bacteria_batch = build_bacteria_batch_from_raw(dose_raw, device=device)
                dose_cf = model.counterfactual_forward(
                    batch=batch_device,
                    new_bacteria_batch=dose_bacteria_batch,
                    num_diffusion_samples=num_samples,
                )
                dose_expr = dose_cf["recon_mu"].detach().cpu().numpy().astype(np.float32)
                dose_delta = dose_cf["delta_cf"].detach().cpu().numpy().astype(np.float32)
                dose_expr_store[dose].append(dose_expr)
                dose_delta_store[dose].append(dose_delta)

                for local_idx, cell_id in enumerate(raw_cell_ids):
                    dose_long_rows.append(
                        {
                            "cell_id": cell_id,
                            "cell_index": int(raw_cell_index[local_idx]),
                            "dose_scale": float(dose),
                            "delta_norm": float(np.linalg.norm(dose_delta[local_idx])),
                            "effect_norm": float(np.linalg.norm(dose_expr[local_idx] - pred_zero[local_idx])),
                            "mean_pred_expression": float(dose_expr[local_idx].mean()),
                            "scaled_bacteria_total_load": float(np.log1p(dose_raw[local_idx].sum())),
                        }
                    )

    predicted_expression_observed = np.concatenate(observed_expr_list, axis=0).astype(np.float32)
    predicted_expression_zero = np.concatenate(zero_expr_list, axis=0).astype(np.float32)
    delta_observed = np.concatenate(observed_delta_list, axis=0).astype(np.float32)
    delta_zero = np.concatenate(zero_delta_list, axis=0).astype(np.float32)
    dose_predictions = {
        dose: np.concatenate(arrays, axis=0).astype(np.float32)
        for dose, arrays in dose_expr_store.items()
    }
    dose_deltas = {
        dose: np.concatenate(arrays, axis=0).astype(np.float32)
        for dose, arrays in dose_delta_store.items()
    }

    cell_level_df = pd.DataFrame(cell_rows)
    dose_scan_long_df = pd.DataFrame(dose_long_rows).sort_values(["cell_index", "dose_scale"]).reset_index(drop=True)
    gene_level_df = compute_gene_level_effects(
        gene_names=gene_names,
        pred_zero=predicted_expression_zero,
        pred_observed=predicted_expression_observed,
        dose_scales=sorted(dose_scales),
        dose_predictions=dose_predictions,
    )

    if not selected_gene_names:
        selected_gene_names = gene_level_df["gene"].head(5).tolist()

    delta_embeddings = np.stack(
        [delta_zero, delta_observed] + [dose_deltas[dose] for dose in sorted(dose_scales)],
        axis=0,
    ).astype(np.float32)

    cell_level_df.to_csv(cf_dir / "cell_level_counterfactual.csv", index=False)
    gene_level_df.to_csv(cf_dir / "gene_level_effects.csv", index=False)
    dose_scan_long_df.to_csv(cf_dir / "dose_scan_long.csv", index=False)
    np.save(cf_dir / "predicted_expression_observed.npy", predicted_expression_observed)
    np.save(cf_dir / "predicted_expression_zero.npy", predicted_expression_zero)
    np.save(cf_dir / "delta_embeddings.npy", delta_embeddings)

    save_gene_dose_plot(
        dose_scales=sorted(dose_scales),
        dose_predictions=dose_predictions,
        gene_names=gene_names,
        selected_genes=selected_gene_names,
        path=cf_dir / "gene_dose_curves.png",
    )
    save_cell_delta_plot(dose_scan_long_df, cf_dir / "cell_delta_norm_vs_dose.png")

    print(f"Saved counterfactual outputs to: {cf_dir}")


if __name__ == "__main__":
    main()
