"""
Prepare Combo Sci-Plex data for BacDiff pipeline.

Converts the Sci-Plex drug combination dataset into BacDiff's expected format:
  - obsm["bacteria_raw"]: multi-hot encoding of 17 drugs per cell (float32)
  - obs["bacteria_label"]: "none" for Vehicle, condition name for perturbed cells
  - uns["bacteria_names"]: list of 17 drug names

Usage:
    python data/prepare_sciplex.py \
        --input /path/to/combo_sciplex_prep_hvg_filtered.h5ad \
        --output data/sciplex_merged.h5ad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


# The 17 unique drugs in the Sci-Plex combo dataset (sorted for reproducibility)
DRUG_NAMES = sorted([
    "Alvespimycin", "Carmofur", "Cediranib", "Crizotinib", "Curcumin",
    "Dacinostat", "Danusertib", "Dasatinib", "Givinostat", "PCI-34051",
    "Panobinostat", "Pirarubicin", "SRT1720", "SRT2104", "SRT3025",
    "Sorafenib", "Tanespimycin",
])


def build_drug_multi_hot(adata: ad.AnnData) -> np.ndarray:
    """Build a (n_cells, 17) multi-hot matrix encoding drug presence."""
    n_cells = adata.n_obs
    n_drugs = len(DRUG_NAMES)
    drug_to_idx = {d: i for i, d in enumerate(DRUG_NAMES)}

    multi_hot = np.zeros((n_cells, n_drugs), dtype=np.float32)

    drug1 = adata.obs["Drug1"].values
    drug2 = adata.obs["Drug2"].values

    for i in range(n_cells):
        d1, d2 = str(drug1[i]), str(drug2[i])
        if d1 in drug_to_idx:
            multi_hot[i, drug_to_idx[d1]] = 1.0
        if d2 in drug_to_idx:
            multi_hot[i, drug_to_idx[d2]] = 1.0

    return multi_hot


def convert(args: argparse.Namespace) -> None:
    print(f"Loading: {args.input}")
    adata = ad.read_h5ad(args.input)
    print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # --- Build bacteria_raw (multi-hot drug encoding) ---
    multi_hot = build_drug_multi_hot(adata)
    print(f"  Multi-hot shape: {multi_hot.shape}")
    print(f"  Cells with ≥1 drug: {(multi_hot.sum(axis=1) > 0).sum()}")
    print(f"  Cells with 2 drugs: {(multi_hot.sum(axis=1) == 2).sum()}")

    # --- Build bacteria_label ---
    # Control → "none", perturbed → condition name (shortened)
    labels = []
    for i in range(adata.n_obs):
        if adata.obs["control"].iloc[i] == 1:
            labels.append("none")
        else:
            # Use condition name (drug combo)
            labels.append(str(adata.obs["condition"].iloc[i]))
    labels = np.array(labels)
    n_none = (labels == "none").sum()
    print(f"  None cells: {n_none} / {len(labels)}")
    print(f"  Unique labels: {len(set(labels))}")

    # --- Build the output AnnData ---
    # X: keep as-is (already log1p normalized)
    out = ad.AnnData(
        X=adata.X.copy(),
        obs=pd.DataFrame(index=adata.obs_names.copy()),
    )

    # Copy useful metadata
    for col in ["control", "condition", "condition_ID", "Drug1", "Drug2",
                 "dose_value", "pathway1", "pathway2", "cell_line", "split"]:
        if col in adata.obs.columns:
            out.obs[col] = adata.obs[col].values

    # BacDiff required fields
    out.obs["bacteria_label"] = pd.Categorical(labels)
    out.obsm["bacteria_raw"] = multi_hot
    out.uns["bacteria_names"] = np.array(DRUG_NAMES)

    # Copy var info
    out.var = adata.var.copy()

    # Copy embeddings if present
    for key in adata.obsm.keys():
        if key != "bacteria_raw":
            out.obsm[key] = adata.obsm[key].copy()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(str(output_path))
    print(f"\nSaved to: {output_path}")
    print(f"  Shape: {out.shape}")
    print(f"  obsm['bacteria_raw']: {out.obsm['bacteria_raw'].shape}")
    print(f"  uns['bacteria_names']: {len(out.uns['bacteria_names'])} drugs")
    print(f"  obs['bacteria_label']: {out.obs['bacteria_label'].nunique()} categories")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Sci-Plex data for BacDiff")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the prepared Sci-Plex h5ad",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "sciplex_merged.h5ad"),
        help="Output path for BacDiff-format h5ad",
    )
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
