"""Score held-out Sci-Plex 3 cell distributions with one shared reference space."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
H5AD = ROOT / "data/sciplex3_pair_ood_500hvg.h5ad"
OUT = ROOT / "checking_data/outputs/sciplex3_pair_ood"
SCORE_COLUMNS = ["energy", "mmd2", "mean_r2", "mean_pearson", "std_rmse", "delta_pearson"]
CONDITION_COLUMNS = ["cell_type", "treatment", "dose_numeric", "time_point"]
RESPONSE_STRATA = ("weak", "medium", "strong")
BOOTSTRAP_METRICS = ("energy", "mmd2", "delta_pearson")
BOOTSTRAP_DRAWS = 2_000


def load_inputs(path):
    with h5py.File(path, "r") as handle:
        obs = ad.io.read_elem(handle["obs"])
        genes = ad.io.read_elem(handle["var"]).index.to_numpy(dtype=str)
        expression = ad.io.read_elem(handle["X"])
    return obs, genes, expression


def fit_reference(obs, expression, n_components, seed):
    train = np.flatnonzero(obs["split"].eq("train").to_numpy())
    rng = np.random.default_rng(seed)
    subset = rng.choice(train, min(40_000, len(train)), replace=False)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(expression[subset].toarray())

    # Calibrate one kernel scale from pairs of real training cells that share
    # an exact treatment; differences between cell lines are not the bandwidth.
    treated = obs["split"].eq("train") & ~obs["is_control"]
    idx = np.flatnonzero(treated.to_numpy())
    groups = obs.iloc[idx].groupby("bacteria_label", observed=True).indices
    pairs = [rng.choice(idx[rows], (8, 2), replace=True) for rows in groups.values() if len(rows) > 1]
    pairs = np.concatenate(pairs)
    left, right = (pca.transform(expression[pairs[:, k]].toarray()) for k in (0, 1))
    sq = np.square(left - right).sum(axis=1)
    median = float(np.median(sq[sq > 0]))
    if not np.isfinite(median) or median <= 0:
        raise ValueError("Training observations did not give a positive bandwidth")
    return pca, median, len(subset)


def within(sq, n, gammas):
    """Mean pairwise distance and RBF kernel including their diagonals."""
    offdiag = (n - 1) / n
    d = float(np.sqrt(sq).mean()) * offdiag
    k = np.asarray([np.exp(-gamma * sq).mean() for gamma in gammas]) * offdiag + 1 / n
    return d, k


def distances(p, q, gammas, ref_within=None):
    if p.ndim != 2 or p.shape != q.shape or len(p) < 2:
        raise ValueError("Distances require two equal-size samples of at least two cells")
    dd_p, k_p = within(pdist(p, "sqeuclidean"), len(p), gammas) if ref_within is None else ref_within
    dd_q, k_q = within(pdist(q, "sqeuclidean"), len(q), gammas)
    cross = cdist(p, q, "sqeuclidean")
    energy = 2 * np.sqrt(cross).mean() - dd_p - dd_q
    k_cross = np.asarray([np.exp(-gamma * cross).mean() for gamma in gammas])
    mmd2 = (k_p + k_q - 2 * k_cross).mean()
    return max(0.0, float(energy)), max(0.0, float(mmd2))


def correlation(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 1e-12 else np.nan


def score(real, real_pca, pred, pred_pca, control_mean, gammas, ref_within):
    t, p = real.mean(axis=0), pred.mean(axis=0)
    energy, mmd2 = distances(real_pca, pred_pca, gammas, ref_within)
    return {
        "energy": energy,
        "mmd2": mmd2,
        "mean_r2": r2_score(t, p),
        "mean_pearson": correlation(t, p),
        "std_rmse": float(np.sqrt(np.mean(np.square(real.std(0) - pred.std(0))))),
        "delta_pearson": correlation(t - control_mean, p - control_mean),
    }


def load_predictions(path, genes, expected):
    with np.load(path, allow_pickle=False) as bundle:
        pred = bundle["expression"].astype(np.float32, copy=False)
        if not np.array_equal(bundle["gene_names"], genes):
            raise ValueError(f"Gene order differs: {path}")
        labels = bundle["labels"].astype(str)
    if pred.shape != (len(labels), len(genes)) or not np.isfinite(pred).all() or np.min(pred) < -1e-6:
        raise ValueError(f"Predictions must be finite and nonnegative in log1p space: {path}")
    if np.max(pred) > np.log1p(10_000) + 1e-3:
        raise ValueError(f"Predictions exceed the CP10k expression range: {path}")
    mass = np.expm1(pred).sum(axis=1)
    if not np.all(np.isclose(mass, 10_000, atol=1) | np.isclose(mass, 0, atol=1e-5)):
        raise ValueError(f"Predictions are not log1p CP10k: {path}")
    groups = pd.Series(labels).groupby(labels).indices
    if set(groups) != set(expected):
        missing = set(expected) - set(groups)
        extra = set(groups) - set(expected)
        raise ValueError(f"Prediction label coverage differs in {path}: {len(missing)} missing, {len(extra)} extra")
    return pred, groups


def apply_delta(x, delta):
    # Shift a real vehicle population in log expression and restore CP10k.
    abundance = np.expm1(np.clip(x + delta, 0, np.log1p(10_000)))
    total = abundance.sum(axis=1, keepdims=True)
    abundance *= 10_000 / np.maximum(total, 1)
    return np.log1p(abundance)


def train_controls_and_effects(obs, expression):
    train = obs["split"].eq("train").to_numpy()
    control = obs["is_control"].to_numpy()
    columns = ["cell_type", "time_point"]
    keys = [tuple(map(str, x)) for x in obs[columns].itertuples(index=False, name=None)]
    rows = {}
    for i in np.flatnonzero(train & control):
        rows.setdefault(keys[i], []).append(i)
    counts = {ctx: expression[idx].toarray() for ctx, idx in rows.items()}
    means = {ctx: values.mean(0) for ctx, values in counts.items()}

    idx = np.flatnonzero(train & ~control)
    group_keys = ["cell_type", "treatment", "dose_numeric", "time_point"]
    effects = {}
    for (line, drug, dose, time), group in obs.iloc[idx].groupby(group_keys, observed=True).indices.items():
        line, drug, dose, time = str(line), str(drug), float(dose), str(time)
        effects[(line, drug, dose, time)] = np.asarray(expression[idx[group]].mean(0)).ravel() - means[line, time]
    return counts, means, effects


def transferred_effect(effects, line, drug, dose, time):
    matched = [v for (l, d, s, t), v in effects.items() if l != line and (d, s, t) == (drug, dose, time)]
    if matched:
        return np.mean(matched, axis=0), False
    # Only A549 has 72 h in this screen; 24 h is the documented fallback.
    matched = [v for (l, d, s, t), v in effects.items() if l != line and (d, s, t) == (drug, dose, "24")]
    if not matched:
        raise ValueError(f"No training source for {line}, {drug}, {dose}, {time}")
    return np.mean(matched, axis=0), True


def paired_bootstrap(pair_metrics, methods, seed):
    if "bacdiff" not in methods:
        return {}
    rng = np.random.default_rng(seed)
    output = {}
    for name in methods:
        if name == "bacdiff":
            continue
        df = pair_metrics.pivot(index=["cell_type", "treatment"], columns="method", values=["energy", "mmd2"])
        selected = []
        for _, per_line in df.groupby(level=0):
            deltas = np.stack([(per_line[k]["bacdiff"] - per_line[k][name]).to_numpy() for k in ("energy", "mmd2")], axis=1)
            selected.append(deltas[rng.integers(0, len(deltas), (2_000, len(deltas)))])
        boot = np.concatenate(selected, axis=1).mean(axis=1)
        obs = {k: float((df[k]["bacdiff"] - df[k][name]).mean()) for k in ("energy", "mmd2")}
        output[name] = {k: {"delta": obs[k], "ci95": np.percentile(boot[:, i], [2.5, 97.5]).tolist()}
                        for i, k in enumerate(("energy", "mmd2"))}
    return output


def split_groups(obs, split):
    rows = np.flatnonzero(obs["split"].eq(split).to_numpy())
    selected = obs.iloc[rows]
    groups = selected.groupby("bacteria_label", observed=True, sort=True)
    if selected.empty or selected["is_control"].any() or (
        groups[CONDITION_COLUMNS].nunique().max().max() != 1
    ):
        raise ValueError(f"{split} groups must each contain exactly one treated condition")
    return rows, selected, groups.indices


def condition_strength(rows, selected, groups, expression, control_means):
    strengths = {}
    for label, idx in groups.items():
        context = tuple(map(str, selected.iloc[idx[0]][["cell_type", "time_point"]]))
        mean = np.asarray(expression[rows[idx]].mean(axis=0)).ravel()
        strengths[str(label)] = float(np.sqrt(np.square(mean - control_means[context]).mean()))
    return pd.Series(strengths, name="response_strength", dtype=float).rename_axis("condition")


def classify_responses(obs, expression, control_means, split, selection):
    valid = condition_strength(*split_groups(obs, "valid"), expression, control_means)
    thresholds = np.quantile(valid.to_numpy(), [1 / 3, 2 / 3])
    actual = valid if split == "valid" else condition_strength(*selection, expression, control_means)
    bins = np.searchsorted(thresholds, actual.to_numpy(), side="left")
    condition_info = actual.to_frame().assign(
        response_stratum=np.asarray(RESPONSE_STRATA)[bins],
    )
    definition = {
        "formula": "sqrt(mean_gene((mean_condition_log1p_CP10k - mean_matched_train_vehicle_log1p_CP10k)^2))",
        "control_match": ["cell_type", "time_point"],
        "all_available_cells": True,
        "replicates": "Pooled within each condition and matched training vehicle context.",
        "calibration_split": "valid",
        "calibration_conditions": len(valid),
        "calibration_quantiles": [1 / 3, 2 / 3],
        "thresholds": thresholds.tolist(),
        "intervals": {"weak": "x <= t1", "medium": "t1 < x <= t2", "strong": "x > t2"},
        "test_labels_use_measured_response": True,
    }
    return condition_info, definition


def subset_summary(detail):
    by_method_stratum = detail.groupby(["response_stratum", "method"])
    summary = by_method_stratum[SCORE_COLUMNS].mean()
    summary["n_conditions"] = by_method_stratum.size()
    pairs = detail[["response_stratum", "method", "cell_type", "treatment"]].drop_duplicates()
    summary["n_pairs"] = pairs.groupby(["response_stratum", "method"]).size()
    return summary[["n_conditions", "n_pairs", *SCORE_COLUMNS]].reindex(
        RESPONSE_STRATA, level="response_stratum",
    )


def centroid_effects(detail, methods, seed, reference="centroid"):
    """Paired condition deltas averaged within each cell line/drug pair."""
    if reference not in methods:
        return {}
    methods = sorted(set(methods) - {reference})
    if not methods:
        return {}
    strata = {"all": detail}
    if "response_stratum" in detail:
        strata.update({s: detail.loc[detail["response_stratum"].eq(s)] for s in RESPONSE_STRATA})
    out = {}
    pair = ["cell_type", "treatment"]
    for stratum, rows in strata.items():
        pivot = rows.pivot(index=[*pair, "condition"], columns="method", values=list(BOOTSTRAP_METRICS))
        out[stratum] = {}
        for name in methods:
            values = {}
            for metric in BOOTSTRAP_METRICS:
                if rows.empty:
                    values[metric] = {"delta": None, "ci95": None, "n_pairs": 0, "n_conditions": 0}
                    continue
                delta = (pivot[metric][name] - pivot[metric][reference]).dropna()
                by_pair = delta.groupby(level=pair).mean()
                if by_pair.empty:
                    values[metric] = {"delta": None, "ci95": None, "n_pairs": 0, "n_conditions": 0}
                    continue
                rng = np.random.default_rng(seed)
                boot_sum = np.zeros(BOOTSTRAP_DRAWS)
                for _, line in by_pair.groupby(level="cell_type"):
                    sampled = rng.integers(0, len(line), size=(BOOTSTRAP_DRAWS, len(line)))
                    boot_sum += line.to_numpy()[sampled].sum(axis=1)
                values[metric] = {
                    "delta": float(by_pair.mean()),
                    "ci95": np.quantile(boot_sum / len(by_pair), [0.025, 0.975]).tolist(),
                    "n_pairs": len(by_pair),
                    "n_conditions": len(delta),
                }
            out[stratum][name] = values
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=H5AD)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--split", choices=("valid", "ood"), default="ood")
    parser.add_argument("--response-strata", action="store_true",
                        help="Stratify by measured effect strength using validation tertiles.")
    parser.add_argument("--pred", action="append", required=True, metavar="NAME=NPZ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca", type=int, default=50)
    parser.add_argument("--max-cells", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.out is None:
        args.out = OUT / ("eval" if args.split == "ood" else "eval_valid")
    if args.repeats < 1 or args.max_cells < 2:
        parser.error("Use at least one repeat and at least two cells per group")
    specified = [x.split("=", 1) for x in args.pred]
    if any(len(parts) != 2 or not all(parts) for parts in specified):
        parser.error("Use --pred NAME=NPZ")
    models = dict(specified)
    if len(models) != len(specified) or set(models) & {"vehicle", "dose_matched_transfer", "real_vs_real"}:
        parser.error("Prediction model names must be unique and cannot be built-in references")

    obs, genes, expression = load_inputs(args.data)
    indices, selected, groups = split_groups(obs, args.split)
    predictions = {name: load_predictions(path, genes, groups) for name, path in models.items()}
    pca, median, fit_size = fit_reference(obs, expression, args.pca, args.seed)
    gammas = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0]) / median
    controls, control_means, effects = train_controls_and_effects(obs, expression)
    if args.response_strata:
        response_info, response_definition = classify_responses(
            obs, expression, control_means, args.split, (indices, selected, groups),
        )

    print(f"Scoring {len(groups)} {args.split} conditions, {len(indices):,} cells and {len(models)} model(s)", flush=True)
    rng = np.random.default_rng(args.seed + 1)
    rows, fallback = [], 0
    for label, idx in groups.items():
        n = min(args.max_cells, len(idx) // 2)
        if n < 2 or any(len(group_idx[label]) < n for _, group_idx in predictions.values()):
            raise ValueError(f"Insufficient real or predicted cells in {label}")
        line, drug, dose, time = selected.iloc[idx[0]][CONDITION_COLUMNS]
        line, drug, dose, time = str(line), str(drug), float(dose), str(time)
        target = expression[indices[idx]].toarray()
        target_pca = pca.transform(target)
        sources = controls[line, time]
        shift, used_fallback = transferred_effect(effects, line, drug, dose, time)
        fallback += int(used_fallback)
        pred_by_name = {name: pred[group_idx[label]] for name, (pred, group_idx) in predictions.items()}
        for repeat in range(args.repeats):
            real_a, real_b = rng.choice(len(idx), 2 * n, replace=False).reshape(2, n)
            draw = rng.random(len(idx) + args.max_cells)
            src = sources[rng.choice(len(sources), n, replace=len(sources) < n)]
            compared = {
                "real_vs_real": target[real_b],
                "vehicle": src,
                "dose_matched_transfer": apply_delta(src, shift),
            }
            for name, pred in pred_by_name.items():
                # Common random ranks keep draws independent of model order.
                ranks = draw[:len(pred)] if len(pred) <= len(draw) else np.random.default_rng(
                    np.random.SeedSequence([args.seed, int(idx[0]), repeat])
                ).random(len(pred))
                compared[name] = pred[np.argpartition(ranks, n - 1)[:n]]
            rp = target_pca[real_a]
            ref_within = within(pdist(rp, "sqeuclidean"), n, gammas)
            for name, pred in compared.items():
                rows.append({"condition": label, "cell_type": line, "treatment": drug, "dose": dose, "time": time,
                             "method": name, "repeat": repeat, "n_real": len(idx), "n_draw": n,
                             **score(target[real_a], rp, pred, pca.transform(pred),
                                     control_means[line, time], gammas, ref_within)})

    args.out.mkdir(parents=True, exist_ok=True)
    draws = pd.DataFrame(rows)
    keys = ["method", "condition", "cell_type", "treatment", "dose", "time", "n_real", "n_draw"]
    detail = draws.groupby(keys, as_index=False)[SCORE_COLUMNS].mean()
    pairs = detail.groupby(["method", "cell_type", "treatment"], as_index=False)[SCORE_COLUMNS].mean()
    summary = detail.groupby("method")[SCORE_COLUMNS].mean()
    if args.response_strata:
        detail = detail.join(response_info, on="condition", validate="many_to_one")
        subset_summary(detail).to_csv(args.out / "response_strength_metrics.csv")
    detail.to_csv(args.out / "condition_metrics.csv", index=False)
    pairs.to_csv(args.out / "pair_metrics.csv", index=False)
    summary.to_csv(args.out / "comparison.csv")
    pair_count = int(selected[["cell_type", "treatment"]].drop_duplicates().shape[0])
    summary_doc = {
        "split": args.split,
        "evaluated_cells": len(indices),
        "evaluated_conditions": len(groups),
        "evaluated_pairs": pair_count,
        "genes": len(genes),
        "pca_dims": args.pca,
        "pca_train_cells": fit_size,
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "seed": args.seed,
        "repeats": args.repeats,
        "min_sample": int(detail["n_draw"].min()),
        "max_sample": int(detail["n_draw"].max()),
        "mmd_rbf_gamma": gammas.tolist(),
        "mmd_calibration_median_training_sqdistance": median,
        "transfer_24h_fallback_conditions": fallback,
        "condition_macro": summary.to_dict(orient="index"),
        "pair_macro": pairs.groupby("method")[SCORE_COLUMNS].mean().to_dict(orient="index"),
        "bacdiff_minus_baseline_pair_estimates": paired_bootstrap(pairs, list(models), args.seed + 2),
    }
    if args.split == "ood":
        summary_doc.update(ood_cells=len(indices), ood_conditions=len(groups), ood_pairs=pair_count)
    if args.response_strata:
        summary_doc["response_strength"] = response_definition
    contrasts = centroid_effects(detail, list(models), args.seed + 3)
    if contrasts:
        summary_doc["centroid_pair_bootstrap"] = {
            "draws": BOOTSTRAP_DRAWS,
            "direction": "method minus centroid",
            "unit": "paired condition differences averaged by cell_type and treatment",
            "stratification": "resample treatment pairs within each cell_type on the evaluated split",
            "method_minus_centroid_pair_estimates": contrasts,
        }
        contrast_rows = []
        for stratum, methods in contrasts.items():
            for method, metrics in methods.items():
                for metric, values in metrics.items():
                    lo, hi = values["ci95"] or (None, None)
                    contrast_rows.append({
                        "response_stratum": stratum, "method": method, "metric": metric,
                        "delta": values["delta"], "ci95_low": lo, "ci95_high": hi,
                        "n_pairs": values["n_pairs"], "n_conditions": values["n_conditions"],
                    })
        pd.DataFrame(contrast_rows).to_csv(args.out / "centroid_pair_effects.csv", index=False)
    if {"ot", "random"} <= models.keys():
        ot_contrasts = centroid_effects(detail, ("random", "ot"), args.seed + 3, reference="random")
        summary_doc["ot_minus_random_pair_bootstrap"] = {
            "draws": BOOTSTRAP_DRAWS,
            "direction": "ot minus random",
            "unit": "paired condition differences averaged by cell_type and treatment",
            "stratification": "resample treatment pairs within each cell_type on the evaluated split",
            "ot_minus_random_pair_estimates": {
                stratum: methods["ot"] for stratum, methods in ot_contrasts.items()
            },
        }
    # Pearson of zero response can be undefined (e.g. for small vehicle draws).
    (args.out / "metrics.json").write_text(json.dumps(summary_doc, indent=2) + "\n")
    print(summary.to_string(float_format=lambda x: f"{x:.5f}"), flush=True)


if __name__ == "__main__":
    main()
