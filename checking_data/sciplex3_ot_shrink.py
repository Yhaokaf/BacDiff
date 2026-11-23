"""Select a vehicle mixture on Sci-Plex 3 validation and apply the frozen rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from sciplex3_gate_features import source_features
from score_sciplex3_pair_ood import (
    CONDITION_COLUMNS, H5AD, correlation, distances, fit_reference,
    load_predictions, split_groups, train_controls_and_effects, within,
)


N_CELLS = 256
# Include small reductions so the same amplitude guard is feasible for
# fixed proportions as well as adaptive rules.
FRACTIONS = np.asarray([*np.arange(8) / 8, 0.90625, 0.9375, 0.96875, 1.0])
GUARDS = {"all_delta_pearson_drop": 0.005, "strong_delta_pearson_drop": 0.01,
          "strong_aligned_effect_relative_drop": 0.10}


def load_benchmark(path, validation=False):
    with h5py.File(path, "r") as handle:
        obs = ad.io.read_elem(handle["obs"])
        genes = ad.io.read_elem(handle["var"]).index.to_numpy(dtype=str)
        matrix = ad.io.sparse_dataset(handle["X"])
        train_rows = np.flatnonzero(obs["split"].eq("train").to_numpy())
        train = obs.iloc[train_rows]
        train_x = matrix[train_rows]
        valid_x = matrix[split_groups(obs, "valid")[0]] if validation else None
    controls, control_means, effects = train_controls_and_effects(train, train_x)
    return obs, genes, train, train_x, valid_x, controls, control_means, effects


def condition_inputs(selected, groups, effects):
    rows = []
    for label, idx in groups.items():
        line, drug, dose, time = selected.iloc[idx[0]][CONDITION_COLUMNS]
        line, drug, dose, time = str(line), str(drug), float(dose), str(time)
        rows.append({"condition": str(label), "cell_type": line, "treatment": drug, "dose": dose,
                     "time": time, **source_features(effects, line, drug, dose, time)})
    return pd.DataFrame(rows).set_index("condition")


def left_bin(values, thresholds):
    return np.searchsorted(thresholds, np.asarray(values), side="left")


def source_panel(controls, condition, rng):
    values = controls[condition["cell_type"], condition["time"]]
    cells = rng.choice(len(values), N_CELLS, replace=len(values) < N_CELLS)
    return values[cells], rng.permutation(N_CELLS)


def mix(ot, vehicle, permutation, fraction):
    n = int(np.rint(N_CELLS * fraction))
    if n == N_CELLS:
        return ot
    mixed = vehicle.copy()
    keep = permutation[:n]
    mixed[keep] = ot[keep]
    return mixed


def prediction_groups(path, genes, groups):
    expr, index = load_predictions(path, genes, groups)
    if any(len(rows) != N_CELLS for rows in index.values()):
        raise ValueError(f"Expected {N_CELLS} OT cells per condition in {path}")
    return expr, index


def score_weights(valid, meta, groups, expr, indices, controls, control_means, pca, gammas, seed):
    mix_rng = np.random.default_rng([seed, 111])
    eval_rng = np.random.default_rng([seed, 888])
    scores = np.empty((len(meta), len(FRACTIONS), 3), dtype=float)
    strength = []
    aligned = np.empty((len(meta), len(FRACTIONS)), dtype=float)
    for pos, (label, row) in enumerate(meta.iterrows()):
        real = valid[groups[label]].toarray()
        ot = expr[indices[label]]
        vehicle, order = source_panel(controls, row, mix_rng)
        control_mean = control_means[row["cell_type"], row["time"]]
        truth = real.mean(axis=0) - control_mean
        norm = float(truth @ truth)
        strength.append(np.sqrt(norm / len(truth)))
        real_pc, ot_pc, vehicle_pc = (pca.transform(x) for x in (real, ot, vehicle))
        n = min(128, len(real) // 2)
        fixed = []
        for _ in range(3):
            a = eval_rng.choice(len(real), n, replace=False)
            b = eval_rng.choice(N_CELLS, n, replace=False)
            ref = within(pdist(real_pc[a], "sqeuclidean"), n, gammas)
            fixed.append((real_pc[a], real[a].mean(axis=0) - control_mean, b, ref))
        for j, weight in enumerate(FRACTIONS):
            predicted = mix(ot, vehicle, order, weight)
            encoded = mix(ot_pc, vehicle_pc, order, weight)
            aligned[pos, j] = float((predicted.mean(axis=0) - control_mean) @ truth) / max(norm, 1e-12)
            results = [(*distances(ref_pc, encoded[b], gammas, ref),
                        correlation(d, predicted[b].mean(axis=0) - control_mean))
                       for ref_pc, d, b, ref in fixed]
            scores[pos, j] = np.mean(results, axis=0)
    return scores, np.asarray(strength), aligned


def choose_policies(meta, scores, measured, aligned):
    probs = [1 / 3, 2 / 3]
    measured_thresholds = np.quantile(measured, probs)
    strong = measured > measured_thresholds[1]
    counts = meta.groupby(["cell_type", "treatment"], observed=True)["dose"].transform("size").to_numpy()
    pair_weight = 1 / counts
    thresholds = {"donor": np.quantile(meta["source_mean_effect_rms"], probs).tolist(),
                  "dose": np.quantile(meta["dose"], probs).tolist()}
    bins = {"donor": left_bin(meta["source_mean_effect_rms"], thresholds["donor"]),
            "dose": left_bin(meta["dose"], thresholds["dose"])}

    def evaluate(family, indices):
        idx = np.asarray(indices, dtype=int)
        selected = np.full(len(meta), idx[0]) if family == "constant" else idx[bins[family]]
        e, mmd, d = scores[np.arange(len(meta)), selected].T
        base = {"family": family, "weights": FRACTIONS[idx].tolist(), "mean_ot_share": float(FRACTIONS[selected].mean()),
                "pair_energy": float(np.average(e, weights=pair_weight)), "energy": float(e.mean()),
                "mmd2": float(mmd.mean()), "delta_pearson": float(np.nanmean(d)),
                "strong_delta_pearson": float(np.nanmean(d[strong])),
                "strong_aligned_effect": float(aligned[np.arange(len(meta)), selected][strong].mean())}
        return base

    original = evaluate("constant", [len(FRACTIONS) - 1])
    minimum = {"delta_pearson": original["delta_pearson"] - GUARDS["all_delta_pearson_drop"],
               "strong_delta_pearson": original["strong_delta_pearson"] - GUARDS["strong_delta_pearson_drop"],
               "strong_aligned_effect": original["strong_aligned_effect"] -
               GUARDS["strong_aligned_effect_relative_drop"] * abs(original["strong_aligned_effect"])}
    candidates = [evaluate("constant", [k]) for k in range(len(FRACTIONS))]
    for family in ("donor", "dose"):
        candidates.extend(evaluate(family, [i, j, len(FRACTIONS) - 1]) for j in range(len(FRACTIONS)) for i in range(j + 1))
    for candidate in candidates:
        candidate["eligible"] = all(candidate[key] >= value - 1e-12 for key, value in minimum.items())
    ranking = lambda row: (row["pair_energy"], row["mmd2"], -row["mean_ot_share"])
    picked = {family: min((r for r in candidates if r["eligible"] and r["family"] == family), key=ranking)
              for family in ("donor", "dose", "constant")}
    return dict(thresholds=thresholds, measured_validation_thresholds=measured_thresholds.tolist(),
                original=original, minimum=minimum, selected=picked), candidates


def tune(args):
    obs, genes, train, train_x, valid_x, controls, means, effects = load_benchmark(args.data, validation=True)
    _, valid, groups = split_groups(obs, "valid")
    meta = condition_inputs(valid, groups, effects)
    expr, indices = prediction_groups(args.ot, genes, groups)
    pca, bandwidth, _ = fit_reference(train, train_x, 50, args.seed)
    del train_x
    gammas = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0]) / bandwidth
    scores, measured, aligned = score_weights(valid_x, meta, groups, expr, indices, controls, means,
                                               pca, gammas, args.seed)
    policy, candidates = choose_policies(meta, scores, measured, aligned)
    p = policy["selected"]
    budget = float(np.rint(N_CELLS * p["donor"]["mean_ot_share"]) / N_CELLS)
    methods = {"donor_gate": {"feature": "source_mean_effect_rms", "thresholds": policy["thresholds"]["donor"],
                              "weights": p["donor"]["weights"]},
               "dose_gate": {"feature": "dose", "thresholds": policy["thresholds"]["dose"], "weights": p["dose"]["weights"]},
               "constant": {"weights": p["constant"]["weights"]},
               "equal_budget": {"weights": [budget]}}
    policy.update(version=1, seed=args.seed, validation_input=str(args.ot), validation_conditions=len(meta),
                  candidates=len(candidates), validation_repeats=3, validation_max_draw=128, guards=GUARDS, methods=methods)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).to_csv(args.out / "validation_candidates.csv", index=False)
    meta.assign(measured_rms=measured, original_aligned_effect=aligned[:, -1]).to_csv(args.out / "validation_features.csv")
    target = args.out / "policy.json"
    temporary = target.with_suffix(".partial")
    temporary.write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    print(pd.DataFrame(p).T[["weights", "mean_ot_share", "pair_energy", "delta_pearson", "strong_delta_pearson", "strong_aligned_effect"]].to_string(), flush=True)


def apply(args):
    policy = json.loads(args.policy.read_text())
    if policy["version"] != 1:
        raise ValueError("Unrecognized policy version")
    obs, genes, _, train_x, _, controls, _, effects = load_benchmark(args.data)
    del train_x
    _, selected, groups = split_groups(obs, args.split)
    meta = condition_inputs(selected, groups, effects)
    expr, indices = prediction_groups(args.ot, genes, groups)
    rng = np.random.default_rng([policy["seed"], 111 if args.split == "valid" else 555])
    panels = [source_panel(controls, row, rng) for _, row in meta.iterrows()]
    labels = np.repeat(meta.index.to_numpy(dtype=str), N_CELLS)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.methods or list(policy["methods"]):
        rule = policy["methods"][name]
        values = np.asarray(rule["weights"], dtype=float)
        gates = values[left_bin(meta[rule["feature"]], rule["thresholds"])] if "feature" in rule else np.full(len(meta), values[0])
        if not np.all(np.isfinite(gates) & (0 <= gates) & (gates <= 1)):
            raise ValueError(f"Invalid mixture weights for {name}")
        predictions = np.concatenate([mix(expr[indices[label]], vehicle, perm, fraction)
                                      for label, (vehicle, perm), fraction in zip(meta.index, panels, gates)])
        name_base = f"{name}_{args.split}"
        tmp = args.out / f"{name_base}.partial.npz"
        np.savez_compressed(tmp, expression=predictions, labels=labels, gene_names=genes)
        tmp.replace(args.out / f"{name_base}.npz")
        meta[f"{name}_ot_fraction"] = np.rint(gates * N_CELLS) / N_CELLS
        print(f"{args.split} {name}: {len(predictions):,} profiles, OT share {meta[f'{name}_ot_fraction'].mean():.4f}", flush=True)
    meta.to_csv(args.out / f"routing_{args.split}.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    fit = sub.add_parser("tune")
    run = sub.add_parser("apply")
    for command in (fit, run):
        command.add_argument("--data", type=Path, default=H5AD)
        command.add_argument("--ot", required=True, type=Path)
        command.add_argument("--out", required=True, type=Path)
    fit.add_argument("--seed", type=int, default=42)
    run.add_argument("--policy", required=True, type=Path)
    run.add_argument("--split", choices=("valid", "ood"), default="ood")
    run.add_argument("--methods", nargs="+", choices=("donor_gate", "dose_gate", "constant", "equal_budget"))
    args = parser.parse_args()
    (tune if args.phase == "tune" else apply)(args)


if __name__ == "__main__":
    main()
