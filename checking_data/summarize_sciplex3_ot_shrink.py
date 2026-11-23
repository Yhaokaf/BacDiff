"""Summarize paired Sci-Plex 3 gate comparisons from the shared scorer."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from score_sciplex3_pair_ood import (
    BOOTSTRAP_METRICS,
    CONDITION_COLUMNS,
    H5AD,
    RESPONSE_STRATA,
    centroid_effects,
    load_inputs,
    load_predictions,
    split_groups,
    train_controls_and_effects,
)


COMPARISONS = {
    "donor_gate": ("ot", "constant", "equal_budget", "dose_gate"),
    "gate43": ("ot43", "constant43"),
    "gate44": ("ot44", "constant44"),
}
AMPLITUDE_COLUMNS = ("aligned_amplitude", "response_rmse", "true_effect_rms", "pred_effect_rms")


def paired_contrasts(detail: pd.DataFrame, seed: int) -> pd.DataFrame:
    methods = {name: group for name, group in detail.groupby("method", sort=False)}
    rows = []
    for method, references in COMPARISONS.items():
        if method not in methods:
            continue
        for reference in references:
            if reference not in methods:
                continue
            source = methods[method]
            comparator = methods[reference]
            key = ["condition", "response_stratum", "cell_type", "treatment"]
            if not source[key].sort_values(key).reset_index(drop=True).equals(
                comparator[key].sort_values(key).reset_index(drop=True)
            ):
                raise ValueError(f"Conditions differ between {method} and {reference}")
            contrasts = centroid_effects(
                pd.concat([source, comparator]), [method, reference], seed, reference=reference,
            )
            for stratum, by_method in contrasts.items():
                for metric, stats in by_method[method].items():
                    lo, hi = stats["ci95"] or (None, None)
                    rows.append({
                        "response_stratum": stratum, "method": method, "reference": reference,
                        "metric": metric, "delta": stats["delta"], "ci95_low": lo,
                        "ci95_high": hi, "n_pairs": stats["n_pairs"],
                        "n_conditions": stats["n_conditions"],
                    })
    if not rows:
        raise ValueError("No planned gate comparison has predictions from both methods")
    return pd.DataFrame(rows)


def effect_amplitude(target_mean, predicted_mean, control_mean):
    target = np.asarray(target_mean, dtype=np.float64) - control_mean
    predicted = np.asarray(predicted_mean, dtype=np.float64) - control_mean
    observed_sq = float(target @ target)
    return {
        "aligned_amplitude": float(predicted @ target / observed_sq) if observed_sq > 0 else np.nan,
        "response_rmse": float(np.sqrt(np.square(predicted - target).mean())),
        "true_effect_rms": float(np.sqrt(observed_sq / len(target))),
        "pred_effect_rms": float(np.sqrt(np.square(predicted).mean())),
    }


def amplitude_tables(detail, predictions, data, split):
    obs, genes, expression = load_inputs(data)
    absolute_rows, selected, groups = split_groups(obs, split)
    strata = detail[["condition", "response_stratum"]].drop_duplicates().set_index("condition")
    if set(strata.index) != set(groups) or not strata.index.is_unique:
        raise ValueError(f"Scored conditions do not match the {split} split")
    _, control_means, _ = train_controls_and_effects(obs, expression)
    rows = []
    for name, path in predictions.items():
        values, prediction_groups = load_predictions(path, genes, groups)
        for label, idx in groups.items():
            line, drug, dose, time = selected.iloc[idx[0]][CONDITION_COLUMNS]
            line, drug, dose, time = str(line), str(drug), float(dose), str(time)
            target = np.asarray(expression[absolute_rows[idx]].mean(axis=0, dtype=np.float64)).ravel()
            pred_idx = prediction_groups[label]
            pred = values[pred_idx].mean(axis=0, dtype=np.float64)
            rows.append({
                "method": name, "condition": label, "response_stratum": strata.at[label, "response_stratum"],
                "cell_type": line, "treatment": drug, "dose": dose, "time": time,
                "n_real": len(idx), "n_pred": len(pred_idx),
                **effect_amplitude(target, pred, control_means[line, time]),
            })
    per_condition = pd.DataFrame(rows)
    with_overall = pd.concat([per_condition.assign(response_stratum="all"), per_condition])
    keys = ["method", "response_stratum"]
    grouped = with_overall.groupby(keys, sort=False)
    summary = grouped[list(AMPLITUDE_COLUMNS)].mean()
    summary["n_amplitude_defined"] = grouped["aligned_amplitude"].count()
    summary["n_conditions"] = grouped.size()
    summary["n_pairs"] = with_overall[keys + ["cell_type", "treatment"]].drop_duplicates().groupby(keys).size()
    return per_condition, summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=H5AD)
    parser.add_argument("--split", choices=("valid", "ood"), default="ood")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pred", action="append", default=[], metavar="NAME=NPZ",
                        help="Also score full-population response amplitude for these predictions.")
    args = parser.parse_args()
    predictions = {}
    for spec in args.pred:
        name, eq, path = spec.partition("=")
        if not eq or not name or not path or name in predictions:
            parser.error("Use unique --pred NAME=NPZ entries")
        predictions[name] = Path(path)
    detail = pd.read_csv(args.condition_metrics)
    required = {"method", "condition", "response_stratum", "cell_type", "treatment", *BOOTSTRAP_METRICS}
    if missing := required - set(detail.columns):
        raise ValueError(f"Required scorer columns are missing: {sorted(missing)}")
    if not detail["response_stratum"].isin(RESPONSE_STRATA).all():
        raise ValueError("Use scoring output with --response-strata")
    contrasts = paired_contrasts(detail, seed=args.seed + 3)
    if predictions:
        per_condition, summary = amplitude_tables(detail, predictions, args.data, args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    contrasts.to_csv(args.out / "paired_contrasts.csv", index=False)
    if predictions:
        per_condition.to_csv(args.out / "condition_effect_amplitude.csv", index=False)
        summary.to_csv(args.out / "effect_amplitude_summary.csv", index=False)
    print(contrasts.query("response_stratum == 'all'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
