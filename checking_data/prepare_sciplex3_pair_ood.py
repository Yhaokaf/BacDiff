"""Build the complete labeled Sci-Plex 3 pair holdout from GEO GSM4150378."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
RAW_PREFIX = "GSM4150378_sciPlex3_A549_MCF7_K562_screen"
TARGET_SUM = 10_000
EXPECTED_ENTRIES = 1_007_419_688
TARGET_COLUMNS = ["cell_type", "treatment", "dose_numeric", "time_point"]
SPLIT_NAMES = {1: "train", 2: "valid", 3: "ood"}


def read_annotations(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The R table includes row names without a corresponding header column.
    with gzip.open(raw / "GSM4150378_sciPlex3_pData.txt.gz", "rt") as file:
        header = ["row_name", *next(csv.reader(file, delimiter=" ", skipinitialspace=True))]
        obs = pd.read_csv(file, sep=" ", names=header, dtype=str, skipinitialspace=True,
                          keep_default_na=False)
    cells = pd.read_csv(raw / f"{RAW_PREFIX}_cell.annotations.txt.gz", sep="\t",
                        header=None, names=["cell", "sample"], dtype=str)
    if not obs["cell"].is_unique or not obs["row_name"].equals(obs["cell"]) or not (
        obs[["cell", "sample"]].equals(cells)
    ):
        raise ValueError("GEO phenotype rows do not align to matrix columns")
    genes = pd.read_csv(raw / f"{RAW_PREFIX}_gene.annotations.txt.gz", sep=r"\s+",
                        names=["gene_id", "gene_short_name"], header=0,
                        dtype=str, keep_default_na=False)
    if not genes["gene_id"].is_unique:
        raise ValueError("GEO gene identifiers are not unique")
    obs = obs.drop(columns=["row_name"]).set_index("cell", drop=False)
    return obs, genes


def pair_split(obs: pd.DataFrame, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    treated = obs.loc[~obs["is_control"], ["cell_type", "treatment"]]
    counts = treated.value_counts().rename("n_cells").sort_index().reset_index()
    counts["split"] = "train"
    lines = sorted(counts["cell_type"].unique())
    drugs = sorted(counts["treatment"].unique())
    if len(counts) != len(lines) * len(drugs):
        raise ValueError("Expected each drug in all cell lines")
    per_stratum = round(0.15 * len(drugs))
    if 2 * len(lines) * per_stratum > len(drugs):
        raise ValueError("Unable to leave out this many distinct drugs")

    # Each held-out drug is held out in exactly one line, with two known
    # training lines. Balancing the pair counts avoids favoring larger lines.
    permutation = np.random.default_rng(seed).permutation(drugs)
    for i, (split, line) in enumerate((split, line) for split in ("valid", "ood") for line in lines):
        chosen = permutation[i * per_stratum:(i + 1) * per_stratum]
        mask = counts["cell_type"].eq(line) & counts["treatment"].isin(chosen)
        counts.loc[mask, "split"] = split
    lookup = counts.set_index(["cell_type", "treatment"])["split"]
    labels = lookup.reindex(pd.MultiIndex.from_frame(obs[["cell_type", "treatment"]]))
    if np.any(labels.notna().to_numpy() & obs["is_control"].to_numpy()):
        raise ValueError("Drug and control annotations overlap")
    split_codes = labels.map({name: code for code, name in SPLIT_NAMES.items()})
    split_codes.loc[obs["is_control"].to_numpy()] = 1
    if split_codes.isna().any():
        raise ValueError("Some labeled cells lack a split")
    return split_codes.to_numpy(dtype=np.uint8), counts


def hvg_scores(sums: np.ndarray, squares: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Seurat-style 20-bin dispersion from train-only normalized count moments."""
    mean = sums / n
    variance = np.maximum(squares / n - mean ** 2, 0) * (n / (n - 1))
    dispersion = np.full(mean.shape, np.nan)
    measured = (mean > 0) & (variance > 0)
    dispersion[measured] = np.log(variance[measured] / mean[measured])
    means_binned = pd.cut(np.log1p(mean[measured]), bins=20)
    raw = pd.Series(dispersion[measured])
    by_bin = raw.groupby(means_binned, observed=True)
    std = by_bin.transform("std").to_numpy()
    center = by_bin.transform("mean").to_numpy()
    score = np.full(mean.shape, -np.inf)
    scored = np.zeros(len(raw))
    good = std > 0
    scored[good] = (raw.to_numpy()[good] - center[good]) / std[good]
    scored[np.isnan(std)] = 1.0
    score[measured] = scored
    return mean, score


def make_condition(obs: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    rows = np.arange(len(obs))
    treated = ~obs["is_control"].to_numpy()
    drug_vocab = sorted(obs.loc[treated, "treatment"].unique())
    line_vocab = sorted(obs["cell_type"].unique())
    time_vocab = sorted(obs["time_point"].unique())
    names = (drug_vocab + ["dose_log10_scaled"] + [f"cell_line={x}" for x in line_vocab] +
             [f"time_point={x}" for x in time_vocab])
    condition = np.zeros((len(obs), len(names)), dtype=np.float32)
    condition[rows[treated], pd.Categorical(obs.loc[treated, "treatment"],
                                             categories=drug_vocab).codes] = 1.0
    condition[:, len(drug_vocab)] = np.log10(np.maximum(obs["dose_numeric"].to_numpy(), 1)) / 4
    line_codes = pd.Categorical(obs["cell_type"], categories=line_vocab).codes.astype(np.intp)
    time_codes = pd.Categorical(obs["time_point"], categories=time_vocab).codes.astype(np.intp)
    condition[rows, len(drug_vocab) + 1 + line_codes] = 1.0
    condition[rows, len(drug_vocab) + 1 + len(line_vocab) + time_codes] = 1.0
    return condition, names


def prepare(raw: Path, out: Path, n_hvg: int, seed: int) -> None:
    if np.dtype(np.int32).byteorder not in ("=", "<") or not np.little_endian:
        raise RuntimeError("This data converter requires a little-endian system")
    obs_raw, genes = read_annotations(raw)
    kept = (obs_raw["cell_type"].ne("NA") & obs_raw["treatment"].ne("")).to_numpy()
    obs = obs_raw.loc[kept].copy()
    obs["dose_numeric"] = pd.to_numeric(obs["dose"], errors="raise").astype("int32")
    obs["is_control"] = obs["vehicle"].eq("TRUE") | obs["treatment"].eq("S0000")
    if (obs["is_control"] != obs["dose_numeric"].eq(0)).any():
        raise ValueError("Vehicle and zero-dose annotations disagree")
    split_codes, pairs = pair_split(obs, seed)
    obs["split"] = pd.Categorical([SPLIT_NAMES[i] for i in split_codes])
    labels = (obs["cell_type"] + "|" + obs["treatment"] + "|dose=" +
              obs["dose_numeric"].astype(str) + "|time=" + obs["time_point"])
    obs["bacteria_label"] = pd.Categorical(labels.mask(obs["is_control"], "none"))
    n_train = int((split_codes == 1).sum())
    split_counts = {str(k): int(v) for k, v in obs["split"].value_counts().items()}
    pair_counts = {str(k): int(v) for k, v in pairs["split"].value_counts().items()}
    print(f"Balanced cell line/drug pair counts: {pair_counts}; cells: {split_counts}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sciplex3_pair_ood_", dir=out.parent) as scratch:
        work = Path(scratch)
        binary = work / "sciplex3_stream_counts"
        cpp = Path(__file__).with_name("sciplex3_stream_counts.cpp")
        subprocess.run(
            [*shlex.split(os.environ.get("CXX", "g++")), "-std=c++17", "-O3",
             str(cpp), "-o", str(binary), "-lz"],
            check=True,
        )
        mask = np.zeros(len(obs_raw), dtype=np.uint8)
        mask[kept] = split_codes
        mask_path = work / "split"
        mask.tofile(mask_path)
        prefix = work / "matrix"
        matrix = raw / f"{RAW_PREFIX}_UMI.count.matrix.gz"
        args = [str(matrix), str(len(genes)), str(len(obs_raw)), str(mask_path),
                str(prefix), str(TARGET_SUM)]

        def run_pass(stage: str) -> None:
            n_entries = int(subprocess.check_output([str(binary), stage, *args], text=True).strip())
            if n_entries != EXPECTED_ENTRIES:
                raise ValueError(f"Invalid number of GEO matrix coordinates: {n_entries}")

        print("Pass 1/2: collecting normalized gene moments using only training cells", flush=True)
        run_pass("moments")
        mean, score = hvg_scores(np.fromfile(f"{prefix}.sums", dtype="<f8"),
                                 np.fromfile(f"{prefix}.squares", dtype="<f8"), n_train)
        if not 0 < n_hvg <= np.isfinite(score).sum():
            raise ValueError("Invalid number of HVGs")
        # Sort selected indices to retain the raw annotation column order.
        selected = np.sort(np.argsort(-score, kind="stable")[:n_hvg])
        columns = np.full(len(genes), -1, dtype="<i4")
        columns[selected] = np.arange(n_hvg)
        columns.tofile(f"{prefix}.columns")

        print(f"Pass 2/2: extracting raw counts on the selected {n_hvg} genes", flush=True)
        run_pass("extract")
        row_sizes = np.fromfile(f"{prefix}.row_sizes", dtype="<u4")[kept]
        index = np.empty(len(row_sizes) + 1, dtype=np.int32)
        index[0] = 0
        np.cumsum(row_sizes, out=index[1:])
        counts = sp.csr_matrix((np.fromfile(f"{prefix}.counts", dtype="<i4"),
                                np.fromfile(f"{prefix}.indices", dtype="<i4"), index),
                               shape=(len(obs), n_hvg))
        obs["hvg_library_size"] = np.asarray(counts.sum(axis=1)).ravel()
        normalized = counts.astype(np.float32)
        scale = (TARGET_SUM / np.maximum(obs["hvg_library_size"].to_numpy(), 1)).astype(np.float32)
        # Bound temporary storage instead of creating another full sparse matrix.
        for start in range(0, len(obs), 8192):
            stop = min(start + 8192, len(obs))
            normalized.data[index[start]:index[stop]] *= np.repeat(scale[start:stop], row_sizes[start:stop])
        np.log1p(normalized.data, out=normalized.data)
        var = genes.iloc[selected].copy()
        var["train_normalized_mean"] = mean[selected]
        var["train_hvg_score"] = score[selected]
        var.index = (var["gene_short_name"] + "|" + var["gene_id"]).rename("gene")
        cond, names = make_condition(obs)
        # Persist plate identifiers and all other source phenotype annotations.
        a = ad.AnnData(X=normalized, obs=obs, var=var)
        a.layers["counts"] = counts
        a.obsm["bacteria_raw"] = cond
        a.uns["bacteria_names"] = names
        a.uns["sciplex3_source"] = "GSE139944/GSM4150378"
        a.uns["hvg_selection"] = (
            "training cells only, 20 bins of log1p mean; standardized log variance/mean "
            "after full-library normalization to 10,000"
        )
        a.uns["count_normalization"] = "log1p(counts / selected gene library size * 10000)"
        a.uns["split_seed"] = seed
        a.uns["heldout_unit"] = "cell_type × treatment across all doses, times and replicates"
        a.uns["heldout_pair_control_policy"] = "All vehicle cells are train references; no held-out treated cell is train."
        print(f"Writing {a.shape[0]} cells and {a.shape[1]} genes", flush=True)
        temp_output = work / "output.h5ad"
        a.write_h5ad(temp_output, compression="lzf")
        a.file.close()
        counts_per_group = (
            obs.loc[~obs["is_control"]].groupby(["split", *TARGET_COLUMNS], observed=True).size()
        )
        manifest = {
            "source": a.uns["sciplex3_source"],
            "raw_cells": len(obs_raw),
            "excluded_unlabeled_cells": int((~kept).sum()),
            "labeled_cells": len(obs),
            "control_cells": int(obs["is_control"].sum()),
            "n_genes_raw": len(genes),
            "n_hvg": n_hvg,
            "hvg_selection": a.uns["hvg_selection"],
            "normalization": a.uns["count_normalization"],
            "seed": seed,
            "split_cells": split_counts,
            "split_pairs": pair_counts,
            "split_conditions": {
                str(k): int(v) for k, v in counts_per_group.groupby(level=0, observed=True).size().items()
            },
            "smallest_group_cells": {
                str(k): int(v) for k, v in counts_per_group.groupby(level=0, observed=True).min().items()
            },
            "zero_selected_library_cells": int(obs["hvg_library_size"].eq(0).sum()),
            "conditioning_fields": ["treatment (one hot)", "dose (scaled log10)",
                                    "cell_type (one hot)", "time_point (one hot)"],
        }
        pairs.to_csv(work / "pairs.csv", index=False)
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (work / "genes.txt").write_text("\n".join(a.var_names) + "\n")
        (work / "conditions.csv").write_text(counts_per_group.rename("n_cells").to_csv())
        os.replace(work / "pairs.csv", out.with_suffix(".pairs.csv"))
        os.replace(work / "manifest.json", out.with_suffix(".manifest.json"))
        os.replace(work / "genes.txt", out.with_suffix(".genes.txt"))
        os.replace(work / "conditions.csv", out.with_suffix(".conditions.csv"))
        os.replace(temp_output, out)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/sciplex3_raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/sciplex3_pair_ood_500hvg.h5ad")
    parser.add_argument("--n-hvg", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.raw_dir, args.output, args.n_hvg, args.seed)
