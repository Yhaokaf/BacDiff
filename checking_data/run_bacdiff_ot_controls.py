"""Run controlled BacDiff OT ablations without changing the main OT artifacts.

The controls are:

* ``no_posterior_noise``: use deterministic treated VAE means as diffusion
  targets;
* ``expr_aux``: retain the original target sampling and add a small decoder
  expression loss to the DDPM noise loss;
* ``latent64``: retrain the VAE with a 64-dimensional latent and then use the
  same OT/DDPM protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn.utils import clip_grad_norm_

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from bacdiff.cli.sciplex3_pair import (
    contexts,
    fit_vae,
    load_fit_data,
    metadata,
    normalize,
)
from bacdiff.cli.sciplex3_pair_ablation import (
    inference_panel,
    model_for,
    prepare as prepare_frozen,
    sample,
    save_checkpoint,
)
from bacdiff.data.sciplex3_ablation_validation import make_validation_scorer
from bacdiff.data.sciplex3_pairing import ConditionPairer
from bacdiff.models.model import BacDiffModel
from bacdiff.utils.config import load_config
from bacdiff.utils.runtime import resolve_device, set_seed


ARMS = ("no_posterior_noise", "expr_aux", "latent64")


def read_options(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def latent_config(base: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base))
    cfg["model"]["latent_dim"] = 64
    cfg["run_name"] = "latent64"
    return cfg


def train_latent_vae(cfg: dict[str, Any], output: Path, device: torch.device) -> Path:
    """Train only the VAE for the 64D arm and emit a frozen-source checkpoint."""
    checkpoint = output / "latent64_vae.pt"
    if checkpoint.exists():
        return checkpoint
    (output / "latent64_vae_run").mkdir(parents=True, exist_ok=True)
    set_seed(cfg["training"]["seed"])
    obs, genes, features = metadata(cfg)
    context_names, context_codes = np.unique(contexts(obs), return_inverse=True)
    fit_data = load_fit_data(cfg, obs, genes, features, context_codes, device)
    model = BacDiffModel(len(genes), len(features), cfg).to(device)
    epoch = fit_vae(model, fit_data, cfg, output / "latent64_vae_run")
    state = {
        "model": model.state_dict(),
        "config": cfg,
        "genes": genes,
        "features": features,
        "vae_epoch": epoch,
    }
    save_checkpoint(state, checkpoint)
    del fit_data, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint


def prepare_latent64(base_options: dict[str, Any], output: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    base_cfg = load_config(base_options["model_config"])
    cfg = latent_config(base_cfg)
    latent_root = output / "latent64"
    latent_root.mkdir(parents=True, exist_ok=True)
    cfg_path = latent_root / "latent64_config.yaml"
    prepared = latent_root / "prepared.pt"
    if cfg_path.exists() and prepared.exists():
        cfg = load_config(cfg_path)
        options = dict(base_options)
        options["model_config"] = str(cfg_path)
        options["frozen_checkpoint"] = str(latent_root / "latent64_vae.pt")
        options["output_dir"] = str(latent_root)
        return cfg, options
    source = train_latent_vae(cfg, latent_root, device)
    options = dict(base_options)
    options["model_config"] = base_options["model_config"]
    options["frozen_checkpoint"] = str(source)
    options["output_dir"] = str(output / "latent64")
    # prepare_frozen reads the model configuration itself, so temporarily use
    # the latent config path written beside the source checkpoint.
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    options["model_config"] = str(cfg_path)
    if not (latent_root / "prepared.pt").exists():
        prepare_frozen(cfg, options, latent_root, device)
    return cfg, options


def load_data(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def pair_inputs(data: dict[str, Any], cfg: dict[str, Any], options: dict[str, Any], device: torch.device):
    mean, logvar = (data[key].to(device) for key in ("train_mean", "train_logvar"))
    scale, anchors = (data[key].to(device) for key in ("scale", "anchors"))
    ctx, controls, codes, features = (data[key].to(device) for key in
                                       ("train_context", "controls", "condition_ids", "condition_features"))
    treated = torch.nonzero(~controls, as_tuple=True)[0]
    pairer = ConditionPairer(
        mean / scale,
        ctx,
        controls,
        codes,
        minibatch_size=options["pair_batch_size"],
        ot_epsilon=options["ot_epsilon"],
        ot_iterations=options["ot_iterations"],
    )
    if not torch.equal(pairer.treated_rows, treated):
        raise ValueError("OT pairer changed treated-row order")
    return mean, logvar, scale, anchors, ctx, controls, codes, features, treated, pairer


def expression_aux_loss(model: BacDiffModel, base: torch.Tensor, delta_pred: torch.Tensor,
                        target_counts: torch.Tensor, depth: torch.Tensor, target_sum: float) -> torch.Tensor:
    """Compare decoded predicted clean state with the treated target expression.

    The target depth is the observed treated cell depth during training. This
    auxiliary term is only used to align the frozen decoder's expression
    coordinates; it does not change the DDPM target or the OT pairing.
    """
    mu, _ = model.decoder(base + delta_pred, depth)
    predicted = normalize(mu, target_sum)
    target = normalize(target_counts, target_sum)
    return torch.mean((predicted - target) ** 2)


def fit_control(data: dict[str, Any], cfg: dict[str, Any], options: dict[str, Any],
                output: Path, arm: str, device: torch.device) -> None:
    settings = cfg["training"]
    seed = int(data["seed"])
    set_seed(seed)
    model = model_for(data, cfg, device)
    mean, logvar, scale, anchors, ctx, controls, codes, features, treated, pairer = pair_inputs(data, cfg, options, device)
    params = list(model.bacteria_encoder.parameters()) + list(model.diffusion.parameters())
    optimizer = torch.optim.AdamW(params, lr=settings["diffusion_lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["diffusion_epochs"], eta_min=1e-4,
    )
    order_rng = torch.Generator(device=device).manual_seed(seed + 70)
    validation = inference_panel(data, cfg, "valid", seed + 300, data["validation"]["n_draws"])
    validate = make_validation_scorer(data["validation"])
    run = output / arm
    run.mkdir(parents=True, exist_ok=True)
    history, best, chosen = [], None, None
    start = time.perf_counter()
    for epoch in range(1, settings["diffusion_epochs"] + 1):
        model.train()
        # The VAE and decoder are frozen. Keep the decoder deterministic so
        # the expression auxiliary loss is not evaluated through dropout.
        model.rna_encoder.eval()
        model.decoder.eval()
        anchor = mean[pairer.control_rows("ot", seed + 100_003 * epoch)]
        total = torch.zeros((), device=device)
        total_aux = torch.zeros((), device=device)
        for idx in torch.randperm(len(treated), generator=order_rng, device=device).split(settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            sel = treated[idx]
            base = anchor[idx]
            # Draw the posterior noise for every arm so the only difference
            # in the no-noise control is its amplitude, not later RNG draws.
            posterior_noise = torch.randn_like(mean[sel])
            noise_scale = 0.0 if arm == "no_posterior_noise" else 1.0
            z_target = mean[sel] + noise_scale * posterior_noise * (0.5 * logvar[sel]).exp()
            residual = (z_target - base) / scale
            condition = torch.cat([base, model.bacteria_encoder(features[codes[sel]])], dim=-1)
            diffusion = model.diffusion.diffusion_loss(residual, condition)
            loss = diffusion["diffusion_loss"]
            aux = torch.zeros((), device=device)
            if arm == "expr_aux":
                # The selected treated cells are loaded lazily from the H5AD
                # once per batch below; their observed depth is not used at
                # inference and is only a training target for this auxiliary.
                target_counts, target_depth = read_training_counts(data, cfg, sel, device)
                aux = expression_aux_loss(model, base, diffusion["delta_pred"] * scale,
                                          target_counts, target_depth, cfg["data"]["target_sum"])
                loss = loss + float(options["expr_aux_weight"]) * aux
            loss.backward()
            clip_grad_norm_(params, settings["grad_clip_norm"])
            optimizer.step()
            total += diffusion["diffusion_loss"].detach() * len(idx)
            total_aux += aux.detach() * len(idx)
        scheduler.step()
        row = {"epoch": epoch, "train_noise_mse": total.item() / len(treated),
               "train_expr_aux": total_aux.item() / len(treated)}
        status = ""
        if epoch % options["validate_every"] == 0 or epoch == settings["diffusion_epochs"]:
            metrics = validate(sample(model, data, validation, cfg, "ot", device, seed + 310))
            key = (metrics["energy"], metrics["mmd2"], -metrics["delta_pearson"])
            row.update({f"val_{key_name}": value for key_name, value in metrics.items()})
            if not np.isfinite(key).all():
                raise RuntimeError("Nonfinite validation metrics")
            if best is None or key < best:
                best, chosen = key, row.copy()
                save_checkpoint({"epoch": epoch, "arm": arm, "config": cfg, "model": model.state_dict(),
                                 "source_sha256": data["source_sha256"], "metrics": metrics}, run / "best_model.pt")
            status = f" Energy {key[0]:.5f}; MMD2 {key[1]:.5f}; delta Pearson {-key[2]:.5f}; best {chosen['epoch']}"
        history.append(row)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        print(f"{arm} epoch {epoch:02d}/{settings['diffusion_epochs']} noise {row['train_noise_mse']:.6f} aux {row['train_expr_aux']:.6f}{status}", flush=True)
    metadata_run = {
        "arm": arm, "seed": seed, "source_checkpoint": data["source_checkpoint"],
        "source_sha256": data["source_sha256"], "vae_epoch": data["vae_epoch"],
        "latent_dim": int(cfg["model"]["latent_dim"]), "epochs": settings["diffusion_epochs"],
        "optimizer_steps": settings["diffusion_epochs"] * (-(-len(treated) // settings["batch_size"])),
        "ot": {"batch_size": options["pair_batch_size"], "epsilon": options["ot_epsilon"],
               "iterations": options["ot_iterations"]},
        "target_posterior_noise": arm != "no_posterior_noise",
        "expr_aux_weight": float(options["expr_aux_weight"]) if arm == "expr_aux" else 0.0,
        "chosen": chosen, "seconds": time.perf_counter() - start,
    }
    write_json(run / "run.json", metadata_run)


def read_training_counts(data: dict[str, Any], cfg: dict[str, Any], rows: torch.Tensor, device: torch.device):
    """Read count rows for an auxiliary loss without adding them to cached data."""
    # The prepared cache retains original train order. Build the h5ad row map
    # once per process and reuse it; this avoids storing another 500-column
    # training matrix in the prepared checkpoint.
    if not hasattr(read_training_counts, "_train_counts"):
        obs, _, _ = metadata(cfg)
        train_rows = np.flatnonzero(obs[cfg["data"]["split_key"]].eq("train").to_numpy())
        from anndata.io import sparse_dataset

        values = np.empty((len(train_rows), len(data["genes"])), dtype=np.float32)
        with h5py.File(cfg["data"]["merged_h5ad"], "r") as handle:
            counts = sparse_dataset(handle["layers/counts"])
            for start in range(0, len(train_rows), 16_384):
                idx = train_rows[start:start + 16_384]
                first, stop = int(idx[0]), int(idx[-1]) + 1
                values[start:start + len(idx)] = counts[first:stop][idx - first].toarray()
        read_training_counts._train_counts = torch.from_numpy(values)
    tensor = read_training_counts._train_counts[rows.detach().cpu().numpy()].to(device)
    return tensor, tensor.sum(dim=-1, keepdim=True).clamp_min(1.0)


@torch.inference_mode()
def predict_control(data: dict[str, Any], cfg: dict[str, Any], output: Path, arm: str,
                    split: str, device: torch.device) -> None:
    path = output / arm
    best = torch.load(path / "best_model.pt", map_location="cpu", weights_only=True)
    model = model_for(data, cfg, device)
    model.load_state_dict(best["model"])
    seed = data["seed"] + (500 if split == "ood" else 400)
    panel = inference_panel(data, cfg, split, seed)
    expression = sample(model, data, panel, cfg, "ot", device, seed + 10)
    name = "predictions.npz" if split == "ood" else "predictions_valid.npz"
    tmp = path / f"{name}.partial.npz"
    np.savez_compressed(tmp, expression=expression, labels=panel["labels"], gene_names=np.asarray(data["genes"], dtype=str))
    tmp.replace(path / name)
    print(f"{arm} epoch {best['epoch']}: generated {len(expression):,} {split} cells", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit", "predict"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--config", default="configs/config_sciplex3_pair_controls.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("valid", "ood"), default="ood")
    args = parser.parse_args()
    options = read_options(args.config)
    output = args.output or Path(options["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(options["model_config"])
    device = resolve_device(base_cfg["training"]["device"])
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.set_float32_matmul_precision("high")

    if args.arm == "latent64":
        cfg, arm_options = prepare_latent64(options, output, device)
        prepared = output / "latent64" / "prepared.pt"
    else:
        cfg = base_cfg
        arm_options = options
        prepared = Path("checking_data/outputs/sciplex3_pair_ablation/prepared.pt")
        if args.phase == "prepare":
            raise ValueError("32D controls reuse checking_data/outputs/sciplex3_pair_ablation/prepared.pt")
    if args.phase == "prepare":
        # latent64 is prepared above; this branch exists for a consistent CLI.
        return
    data = load_data(prepared)
    if args.phase == "fit":
        fit_control(data, cfg, arm_options, output, args.arm, device)
    else:
        predict_control(data, cfg, output, args.arm, args.split, device)


if __name__ == "__main__":
    main()
