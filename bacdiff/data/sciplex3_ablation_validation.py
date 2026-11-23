"""Fixed validation samples and a common distribution scorer for BacDiff ablations."""

from __future__ import annotations

import anndata as ad
import numpy as np
import torch
from scipy.spatial.distance import pdist

from checking_data.score_sciplex3_pair_ood import correlation, distances, fit_reference, within


def _split_rows(obs, rows, split):
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    if rows.ndim != 1 or not len(rows) or rows[0] < 0 or rows[-1] >= len(obs):
        raise ValueError(f"Invalid {split} row indices")
    if np.any(rows[1:] == rows[:-1]) or not obs["split"].iloc[rows].eq(split).all():
        raise ValueError(f"Expected distinct rows belonging to {split}")
    return rows


def build_validation_reference(handle, obs, train_rows, valid_rows, seed, max_cells) -> dict:
    """Build a weights-only-safe reference from train expression and fixed valid draws."""
    if max_cells < 2:
        raise ValueError("Validation needs at least two draws per condition")
    train_rows = _split_rows(obs, train_rows, "train")
    valid_rows = _split_rows(obs, valid_rows, "valid")
    expression = ad.io.sparse_dataset(handle["X"])
    train = obs.iloc[train_rows]
    train_x = expression[train_rows]
    pca, median, _ = fit_reference(train, train_x, 50, seed)
    gammas = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0]) / median

    contexts = ["cell_type", "time_point"]
    control_mask = train["is_control"].to_numpy(dtype=bool)
    control_positions = np.flatnonzero(control_mask)
    control_means = {
        tuple(map(str, key)): np.asarray(train_x[control_positions[idx]].mean(axis=0)).ravel()
        for key, idx in train.iloc[control_positions].groupby(contexts, observed=True).indices.items()
    }
    del train_x

    valid = obs.iloc[valid_rows]
    if valid["is_control"].any():
        raise ValueError("Validation groups must contain treated cells only")
    grouped = valid.groupby("bacteria_label", observed=True).indices
    rng = np.random.default_rng(seed + 1)
    labels, n_draws, sampled, controls = [], [], [], []
    for label, idx in sorted(grouped.items(), key=lambda item: str(item[0])):
        n = min(int(max_cells), len(idx) // 2)
        if n < 2:
            raise ValueError(f"Validation condition {label} has fewer than four real cells")
        context = valid.iloc[idx][contexts].drop_duplicates()
        if len(context) != 1:
            raise ValueError(f"Validation condition {label} spans multiple control contexts")
        key = tuple(map(str, context.iloc[0]))
        if key not in control_means:
            raise ValueError(f"Validation condition {label} has no matching train controls")
        labels.append(str(label))
        n_draws.append(n)
        sampled.append(valid_rows[rng.choice(idx, n, replace=False)])
        controls.append(control_means[key])
    if not labels:
        raise ValueError("No observed validation labels")

    chosen = np.concatenate(sampled)
    ordered = np.argsort(chosen)
    target = expression[chosen[ordered]].toarray()[np.argsort(ordered)]
    return {
        "labels": labels,
        "n_draws": n_draws,
        "target": torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)),
        "pca_mean": torch.from_numpy(pca.mean_),
        "pca_components": torch.from_numpy(pca.components_.copy()),
        "gammas": torch.from_numpy(gammas),
        "control_means": torch.from_numpy(np.stack(controls).astype(np.float32, copy=False)),
    }


def make_validation_scorer(reference):
    """Cache observed statistics and score rows stacked in reference label order."""
    target = reference["target"].cpu().numpy()
    components = reference["pca_components"].cpu().numpy().T
    projected_mean = reference["pca_mean"].cpu().numpy() @ components
    gammas = reference["gammas"].cpu().numpy()
    controls = reference["control_means"].cpu().numpy()
    ends = np.cumsum(reference["n_draws"])
    groups = [slice(start, end) for start, end in zip([0, *ends[:-1]], ends)]
    target_pc = target @ components - projected_mean
    observed = [
        (within(pdist(target_pc[g], "sqeuclidean"), len(target_pc[g]), gammas),
         target[g].mean(axis=0) - ctrl)
        for g, ctrl in zip(groups, controls)
    ]

    def score(expr: np.ndarray) -> dict[str, float]:
        pred = np.asarray(expr, dtype=np.float32)
        if pred.shape != target.shape or not np.isfinite(pred).all():
            raise ValueError("Expected finite predictions matching the validation sample shape")
        pred_pc = pred @ components - projected_mean
        scores = []
        for g, ctrl, (ref_within, target_delta) in zip(groups, controls, observed):
            energy, mmd2 = distances(target_pc[g], pred_pc[g], gammas, ref_within)
            delta = correlation(target_delta, pred[g].mean(axis=0) - ctrl)
            scores.append((energy, mmd2, delta))
        energy, mmd2, delta = np.asarray(scores).T
        finite_delta = delta[np.isfinite(delta)]
        return {"energy": float(energy.mean()), "mmd2": float(mmd2.mean()),
                "delta_pearson": float(finite_delta.mean()) if len(finite_delta) else float("nan")}

    return score
