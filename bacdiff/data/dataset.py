from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

def _to_numpy_dense(matrix: Any) -> np.ndarray:
    """Convert a dense or sparse matrix to a float32 numpy array."""
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SingleCellDataset(Dataset):
    """Dataset for merged single-cell RNA + bacteria data."""

    def __init__(
        self,
        *,
        x_counts: np.ndarray,
        x_input: np.ndarray,
        bac_raw: np.ndarray,
        bac_total_load: np.ndarray,
        is_none: np.ndarray,
        cell_ids: np.ndarray,
        bacteria_labels: np.ndarray,
        gene_names: list[str] | None = None,
        bacteria_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
    ) -> None:
        self.x_counts = torch.from_numpy(x_counts.astype(np.float32))
        self.x_input = torch.from_numpy(x_input.astype(np.float32))
        self.bac_raw = torch.from_numpy(bac_raw.astype(np.float32))
        self.bac_total_load = torch.from_numpy(bac_total_load.astype(np.float32))
        self.is_none = torch.from_numpy(is_none.astype(np.bool_))
        self.cell_ids = np.asarray(cell_ids)
        self.bacteria_labels = np.asarray(bacteria_labels)
        self.gene_names = gene_names or [f"gene_{i}" for i in range(x_input.shape[1])]
        self.bacteria_names = bacteria_names or [f"bac_{i}" for i in range(bac_raw.shape[1])]
        self.covariate_names = covariate_names or []
        library = x_counts.sum(axis=1, keepdims=True).astype(np.float32)
        self.library_size = torch.from_numpy(np.clip(library, a_min=1.0, a_max=None))

    def __len__(self) -> int:
        return self.x_counts.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "x_counts": self.x_counts[idx],
            "x_input": self.x_input[idx],
            "bac_raw": self.bac_raw[idx],
            "bac_total_load": self.bac_total_load[idx],
            "is_none": self.is_none[idx],
            "cell_id": self.cell_ids[idx],
            "cell_index": idx,
            "bacteria_label": self.bacteria_labels[idx],
            "library_size": self.library_size[idx],
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_merged_h5ad(path: str | Path) -> ad.AnnData:
    """Load the merged h5ad produced by split.ipynb."""
    return ad.read_h5ad(str(path))


def preprocess_rna(adata: ad.AnnData, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Normalize + log1p + HVG selection. Returns (x_input, x_counts) as numpy.

    If ``log1p_input`` is False the data is assumed to be already normalized and
    log-transformed; ``normalize_total`` and ``log1p`` are both skipped.
    """
    target_sum = float(config["preprocess"].get("target_sum", 10000.0))
    log1p = bool(config["preprocess"].get("log1p_input", True))
    n_top_genes = int(config["preprocess"].get("n_top_genes", 2000))
    hvg_flavor = str(config["preprocess"].get("hvg_flavor", "seurat"))

    # Prefer the original count layer when X stores normalized/log1p values.
    count_matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    x_counts = _to_numpy_dense(count_matrix).astype(np.float32)

    # Normalize + log1p (skip when data is already preprocessed)
    norm = adata.copy()
    if log1p:
        sc.pp.normalize_total(norm, target_sum=target_sum)
        sc.pp.log1p(norm)
    else:
        print("[data] Skipping normalize_total + log1p (data already preprocessed)")

    # HVG selection
    if n_top_genes > 0 and norm.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(norm, n_top_genes=n_top_genes, flavor=hvg_flavor)
        hvg_mask = norm.var["highly_variable"].values
        if hvg_mask.sum() == 0:
            # Fallback: top variance genes
            variances = np.asarray(norm.X.var(axis=0)).ravel()
            top_idx = np.argsort(variances)[::-1][:n_top_genes]
            hvg_mask = np.zeros(norm.n_vars, dtype=bool)
            hvg_mask[top_idx] = True
        norm = norm[:, hvg_mask].copy()
        x_counts = x_counts[:, hvg_mask]
        print(f"[data] HVG selected: {norm.n_vars} genes (flavor={hvg_flavor})")
    else:
        print(f"[data] No HVG selection (n_top_genes={n_top_genes}, total={norm.n_vars})")

    x_input = _to_numpy_dense(norm.X).astype(np.float32)
    return x_input, x_counts


def select_gene_names(adata: ad.AnnData, config: dict[str, Any]) -> list[str]:
    """Return feature names using the same transform and HVG selection as preprocessing."""
    preprocess_cfg = config["preprocess"]
    norm = adata.copy()
    if bool(preprocess_cfg.get("log1p_input", True)):
        sc.pp.normalize_total(norm, target_sum=float(preprocess_cfg.get("target_sum", 10000.0)))
        sc.pp.log1p(norm)

    n_top_genes = int(preprocess_cfg.get("n_top_genes", 2000))
    if n_top_genes <= 0 or norm.n_vars <= n_top_genes:
        return list(norm.var_names)

    sc.pp.highly_variable_genes(
        norm,
        n_top_genes=n_top_genes,
        flavor=str(preprocess_cfg.get("hvg_flavor", "seurat")),
    )
    hvg_mask = norm.var["highly_variable"].to_numpy()
    if not hvg_mask.any():
        variances = np.asarray(norm.X.var(axis=0)).ravel()
        top_idx = np.argsort(variances)[::-1][:n_top_genes]
        hvg_mask = np.zeros(norm.n_vars, dtype=bool)
        hvg_mask[top_idx] = True
    return list(norm.var_names[hvg_mask])


def prepare_split(config: dict[str, Any]) -> tuple[SingleCellDataset, SingleCellDataset, SingleCellDataset, dict]:
    """Load merged data, preprocess, and return a single 7:1:2 train/val/test split.

    Stratified by bacteria_label. Z-score fitted on train only.

    Returns (train_dataset, val_dataset, test_dataset, split_info).
    """
    # Check if we should use CPA's pre-defined split
    split_key = config.get("data", {}).get("split_key", None)
    if split_key is not None:
        return prepare_split_by_key(config, split_key)

    seed = int(config["training"]["seed"])

    # Load data
    adata = load_merged_h5ad(config["data"]["merged_h5ad"])
    print(f"[data] Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"[data] Bacteria: {len(adata.uns['bacteria_names'])} species")

    # RNA preprocessing: normalize + log1p + HVG
    x_input, x_counts = preprocess_rna(adata, config)
    print(f"[data] RNA preprocessed: {x_input.shape[1]} genes after HVG")

    gene_names = select_gene_names(adata, config)

    # Bacteria data
    bac_raw_full = adata.obsm["bacteria_raw"].astype(np.float32)
    bac_names = list(adata.uns["bacteria_names"])
    labels = adata.obs["bacteria_label"].to_numpy().astype(str)
    cell_ids = adata.obs_names.to_numpy().astype(str)
    is_none = (labels == "none")

    print(f"[data] Labels: {len(set(labels))} unique")
    print(f"[data] None cells: {is_none.sum()} / {len(is_none)}")

    # Two-stage stratified split: 70% train / 10% val / 20% test
    idx_all = np.arange(len(labels))

    # Stage 1: 80% train+val, 20% test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(idx_all, labels))

    # Stage 2: from train+val, split into 87.5% train / 12.5% val → 70% / 10% of total
    labels_trainval = labels[trainval_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    train_sub_idx, val_sub_idx = next(sss2.split(np.zeros(len(labels_trainval)), labels_trainval))
    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    # Binary perturbation indicators should keep an exact all-zero control.
    bac_train = bac_raw_full[train_idx]
    condition_normalization = str(config.get("data", {}).get("condition_normalization", "zscore"))
    if condition_normalization == "zscore":
        bac_mean = bac_train.mean(axis=0, keepdims=True).astype(np.float32)
        bac_std = (bac_train.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    elif condition_normalization == "none":
        bac_mean = np.zeros((1, bac_raw_full.shape[1]), dtype=np.float32)
        bac_std = np.ones((1, bac_raw_full.shape[1]), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported condition_normalization: {condition_normalization}")

    bac_train_z = ((bac_raw_full[train_idx] - bac_mean) / bac_std).astype(np.float32)
    bac_val_z = ((bac_raw_full[val_idx] - bac_mean) / bac_std).astype(np.float32)
    bac_test_z = ((bac_raw_full[test_idx] - bac_mean) / bac_std).astype(np.float32)

    bac_total_load_train = np.log1p(bac_raw_full[train_idx].sum(axis=1, keepdims=True)).astype(np.float32)
    bac_total_load_val = np.log1p(bac_raw_full[val_idx].sum(axis=1, keepdims=True)).astype(np.float32)
    bac_total_load_test = np.log1p(bac_raw_full[test_idx].sum(axis=1, keepdims=True)).astype(np.float32)

    bac_names = list(adata.uns["bacteria_names"])

    train_dataset = SingleCellDataset(
        x_counts=x_counts[train_idx],
        x_input=x_input[train_idx],
        bac_raw=bac_train_z,
        bac_total_load=bac_total_load_train,
        is_none=is_none[train_idx],
        cell_ids=cell_ids[train_idx],
        bacteria_labels=labels[train_idx],
        gene_names=gene_names,
        bacteria_names=bac_names,
    )
    val_dataset = SingleCellDataset(
        x_counts=x_counts[val_idx],
        x_input=x_input[val_idx],
        bac_raw=bac_val_z,
        bac_total_load=bac_total_load_val,
        is_none=is_none[val_idx],
        cell_ids=cell_ids[val_idx],
        bacteria_labels=labels[val_idx],
        gene_names=gene_names,
        bacteria_names=bac_names,
    )
    test_dataset = SingleCellDataset(
        x_counts=x_counts[test_idx],
        x_input=x_input[test_idx],
        bac_raw=bac_test_z,
        bac_total_load=bac_total_load_test,
        is_none=is_none[test_idx],
        cell_ids=cell_ids[test_idx],
        bacteria_labels=labels[test_idx],
        gene_names=gene_names,
        bacteria_names=bac_names,
    )

    split_info = {
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "train_none": int(is_none[train_idx].sum()),
        "val_none": int(is_none[val_idx].sum()),
        "test_none": int(is_none[test_idx].sum()),
        "bac_mean": bac_mean,
        "bac_std": bac_std,
    }

    print(f"[data] Split: train={len(train_idx)} (none={split_info['train_none']}), "
          f"val={len(val_idx)} (none={split_info['val_none']}), "
          f"test={len(test_idx)} (none={split_info['test_none']})")

    return train_dataset, val_dataset, test_dataset, split_info


def prepare_split_by_key(
    config: dict[str, Any],
    split_key: str,
) -> tuple[SingleCellDataset, SingleCellDataset, SingleCellDataset, dict]:
    """Load merged data, preprocess, and split by a pre-defined column in obs.

    Uses ``adata.obs[split_key]`` with expected values ``train`` / ``valid`` / ``ood``.
    Z-score fitted on train only.

    Returns (train_dataset, val_dataset, test_dataset, split_info).
    """
    # Load data
    adata = load_merged_h5ad(config["data"]["merged_h5ad"])
    print(f"[data] Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"[data] Bacteria: {len(adata.uns['bacteria_names'])} species")

    # RNA preprocessing
    x_input, x_counts = preprocess_rna(adata, config)
    print(f"[data] RNA preprocessed: {x_input.shape[1]} genes after HVG")
    gene_names = select_gene_names(adata, config)

    # Bacteria data
    bac_raw_full = adata.obsm["bacteria_raw"].astype(np.float32)
    labels = adata.obs["bacteria_label"].to_numpy().astype(str)
    cell_ids = adata.obs_names.to_numpy().astype(str)
    is_none = (labels == "none")

    # Split by pre-defined column
    if split_key not in adata.obs.columns:
        raise KeyError(f"Split column '{split_key}' not found in adata.obs")

    split_vals = adata.obs[split_key].to_numpy().astype(str)
    train_idx = np.where(split_vals == "train")[0]
    val_idx = np.where(split_vals == "valid")[0]
    test_idx = np.where(split_vals == "ood")[0]

    print(f"[data] Split key: {split_key}")
    print(f"[data] Train: {len(train_idx)}, Val: {len(val_idx)}, OOD: {len(test_idx)}")

    # Binary perturbation indicators should keep an exact all-zero control.
    bac_train = bac_raw_full[train_idx]
    condition_normalization = str(config.get("data", {}).get("condition_normalization", "zscore"))
    if condition_normalization == "zscore":
        bac_mean = bac_train.mean(axis=0, keepdims=True).astype(np.float32)
        bac_std = (bac_train.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    elif condition_normalization == "none":
        bac_mean = np.zeros((1, bac_raw_full.shape[1]), dtype=np.float32)
        bac_std = np.ones((1, bac_raw_full.shape[1]), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported condition_normalization: {condition_normalization}")

    bac_train_z = ((bac_raw_full[train_idx] - bac_mean) / bac_std).astype(np.float32)
    bac_val_z = ((bac_raw_full[val_idx] - bac_mean) / bac_std).astype(np.float32)
    bac_test_z = ((bac_raw_full[test_idx] - bac_mean) / bac_std).astype(np.float32)

    bac_total_load_train = np.log1p(bac_raw_full[train_idx].sum(axis=1, keepdims=True)).astype(np.float32)
    bac_total_load_val = np.log1p(bac_raw_full[val_idx].sum(axis=1, keepdims=True)).astype(np.float32)
    bac_total_load_test = np.log1p(bac_raw_full[test_idx].sum(axis=1, keepdims=True)).astype(np.float32)

    bac_names = list(adata.uns["bacteria_names"])

    train_dataset = SingleCellDataset(
        x_counts=x_counts[train_idx],
        x_input=x_input[train_idx],
        bac_raw=bac_train_z,
        bac_total_load=bac_total_load_train,
        is_none=is_none[train_idx],
        cell_ids=cell_ids[train_idx],
        bacteria_labels=labels[train_idx],
        gene_names=gene_names,
        bacteria_names=bac_names,
    )
    val_dataset = SingleCellDataset(
        x_counts=x_counts[val_idx],
        x_input=x_input[val_idx],
        bac_raw=bac_val_z,
        bac_total_load=bac_total_load_val,
        is_none=is_none[val_idx],
        cell_ids=cell_ids[val_idx],
        bacteria_labels=labels[val_idx],
        gene_names=gene_names,
        bacteria_names=bac_names,
    )
    test_dataset = SingleCellDataset(
        x_counts=x_counts[test_idx],
        x_input=x_input[test_idx],
        bac_raw=bac_test_z,
        bac_total_load=bac_total_load_test,
        is_none=is_none[test_idx],
        cell_ids=cell_ids[test_idx],
        bacteria_labels=labels[test_idx],
        gene_names=gene_names,
        bacteria_names=bac_names,
    )

    split_info = {
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "train_none": int(is_none[train_idx].sum()),
        "val_none": int(is_none[val_idx].sum()),
        "test_none": int(is_none[test_idx].sum()),
        "bac_mean": bac_mean,
        "bac_std": bac_std,
    }

    print(f"[data] Train none: {split_info['train_none']}, Val none: {split_info['val_none']}, "
          f"OOD none: {split_info['test_none']}")

    return train_dataset, val_dataset, test_dataset, split_info


def create_dataloader(
    dataset: SingleCellDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Build a PyTorch DataLoader for a dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


class PreparedData:
    """Container for prepared datasets and metadata."""

    def __init__(
        self,
        train_dataset: SingleCellDataset,
        val_dataset: SingleCellDataset,
        test_dataset: SingleCellDataset,
        split_info: dict,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.split_info = split_info
        self.gene_names = train_dataset.gene_names
        self.bacteria_names = train_dataset.bacteria_names
        self.covariate_names = train_dataset.covariate_names

    @property
    def full_dataset(self) -> SingleCellDataset:
        """Return a combined dataset of all splits."""
        import torch.utils.data as data
        return data.ConcatDataset([
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ])


def prepare_datasets(config: dict[str, Any], fit: bool = True, preprocess_path: str | Path | None = None) -> PreparedData:
    """Prepare datasets for training or inference.

    Args:
        config: Configuration dictionary.
        fit: If True, fit preprocessing on training data. If False, load from preprocess_path.
        preprocess_path: Path to saved preprocessing parameters (required when fit=False).

    Returns:
        PreparedData object with train/val/test datasets and metadata.
    """
    train_dataset, val_dataset, test_dataset, split_info = prepare_split(config)
    return PreparedData(train_dataset, val_dataset, test_dataset, split_info)
