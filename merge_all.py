#!/usr/bin/env python3
"""Merge paired GEX and 16S AnnData datasets."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np

DATASETS = (
    {
        "name": "OSCC15",
        "gex": "OSCC_15.GEX.h5ad",
        "bac": "OSCC_15.16S.h5ad",
        "out": "OSCC_15_merged.h5ad",
    },
    {
        "name": "M5",
        "gex": "GSM9120208_M5_2.GEX.h5ad",
        "bac": "GSM9120208_M5_2.16S.h5ad",
        "out": "M5_merged.h5ad",
    },
    {
        "name": "M20",
        "gex": "GSM9120214_M20_2.GEX.h5ad",
        "bac": "GSM9120214_M20_2.16S.h5ad",
        "out": "M20_merged.h5ad",
    },
)


def merge_dataset(ds: dict[str, str], data_dir: Path, min_cells_per_bacteria: int,
                  min_cells_per_label: int) -> None:
    print(f"\n{'=' * 60}\nMerging {ds['name']}\n{'=' * 60}")

    rna = ad.read_h5ad(data_dir / ds["gex"])
    bac = ad.read_h5ad(data_dir / ds["bac"])

    rna_names = rna.obs_names
    bac_names = bac.obs_names
    if not rna_names.is_unique or not bac_names.is_unique:
        raise ValueError(f"{ds['name']}: cell IDs must be unique in both files")
    if len(rna_names) != len(bac_names) or set(rna_names) != set(bac_names):
        missing_bac = rna_names.difference(bac_names)
        missing_rna = bac_names.difference(rna_names)
        raise ValueError(
            f"{ds['name']}: cell ID sets differ; "
            f"{len(missing_bac)} RNA-only and {len(missing_rna)} 16S-only cells"
        )

    print(f"Aligned cells: {rna.n_obs} (RNA: {rna.n_obs}, 16S: {bac.n_obs})")
    bac = bac[rna_names].copy()

    bac_raw = bac.X.toarray() if hasattr(bac.X, "toarray") else np.array(bac.X)
    bac_names = list(bac.var_names)
    print(f"RNA: {rna.n_obs} cells x {rna.n_vars} genes")
    print(f"BAC: {bac.n_obs} cells x {bac.n_vars} bacteria")

    # Filter bacteria by detection
    cells_per_bacteria = (bac_raw > 0).sum(axis=0)
    keep = cells_per_bacteria >= min_cells_per_bacteria
    print(f"Bacteria filter: {bac_raw.shape[1]} -> {keep.sum()} (>= {min_cells_per_bacteria} cells)")
    for name, n_cells, k in zip(bac_names, cells_per_bacteria, keep):
        status = "KEEP" if k else "DROP"
        print(f"  [{status}] {name:40s} ({n_cells} cells)")
    bac_raw = bac_raw[:, keep]
    bac_names = [n for n, k in zip(bac_names, keep) if k]

    # Create labels
    labels = []
    for i in range(bac_raw.shape[0]):
        present = [bac_names[j] for j in range(len(bac_names)) if bac_raw[i, j] > 0]
        labels.append("_".join(present) if present else "none")
    rna.obs["bacteria_label"] = labels

    # Filter labels
    label_counts = rna.obs["bacteria_label"].value_counts()
    keep_mask = rna.obs["bacteria_label"].isin(label_counts[label_counts >= min_cells_per_label].index)
    print(f"Label filter: {rna.n_obs} cells -> {keep_mask.sum()} cells (>= {min_cells_per_label})")
    for label, count in label_counts.items():
        status = "KEEP" if count >= min_cells_per_label else "DROP"
        print(f"  [{status}] {label:60s} {count:5d}")
    rna = rna[keep_mask].copy()
    bac_raw = bac_raw[keep_mask.values]

    # Save
    rna.obsm["bacteria_raw"] = bac_raw.astype(np.float32)
    rna.obs["bacteria_total_raw"] = bac_raw.sum(axis=1).astype(np.float32)
    rna.uns["bacteria_names"] = bac_names
    output_path = data_dir / ds["out"]
    rna.write_h5ad(output_path)
    print(f"\nSaved: {output_path}")
    print(f"Shape: {rna.n_obs} cells x {rna.n_vars} genes")
    print(f"Bacteria: {len(bac_names)} types")
    print(f"Labels: {rna.obs['bacteria_label'].nunique()} types")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory containing paired GEX and 16S .h5ad files",
    )
    parser.add_argument("--min-cells-per-bacteria", type=int, default=100)
    parser.add_argument("--min-cells-per-label", type=int, default=100)
    args = parser.parse_args()

    for ds in DATASETS:
        merge_dataset(ds, args.data_dir, args.min_cells_per_bacteria, args.min_cells_per_label)
    print("\nAll datasets merged!")


if __name__ == "__main__":
    main()
