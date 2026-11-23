"""Fit and sample BacDiff on the full Sci-Plex 3 held-out pair benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

from bacdiff.models.losses import kl_standard_normal
from bacdiff.models.model import BacDiffModel
from bacdiff.utils.config import load_config, resolve_run_dir
from bacdiff.utils.runtime import resolve_device, set_seed


def metadata(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
    with h5py.File(config["data"]["merged_h5ad"], "r") as handle:
        obs = ad.io.read_elem(handle["obs"])
        genes = list(ad.io.read_elem(handle["var"]).index)
        features = list(ad.io.read_elem(handle["uns/bacteria_names"]))
    if len(genes) != config["data"]["n_genes"] or len(genes) != len(set(genes)):
        raise ValueError("Expected distinct benchmark genes in their fixed order")
    return obs, genes, features


def contexts(obs: pd.DataFrame) -> np.ndarray:
    return obs["cell_type"].astype(str).str.cat(obs["time_point"].astype(str), sep="|").to_numpy(dtype=str)


def normalize(counts: torch.Tensor, target_sum: float) -> torch.Tensor:
    return torch.log1p(counts * (target_sum / counts.sum(dim=-1, keepdim=True).clamp_min(1)))


def load_fit_data(config, obs, genes, features, context_codes, device):
    split = obs[config["data"]["split_key"]].astype(str).to_numpy()
    if set(split) != {"train", "valid", "ood"}:
        raise ValueError("Unexpected split labels")
    result = {}
    with h5py.File(config["data"]["merged_h5ad"], "r") as handle:
        sparse_counts = ad.io.sparse_dataset(handle["layers/counts"])
        encoded_conditions = handle["obsm/bacteria_raw"]
        for part in ("train", "valid"):
            indices = np.flatnonzero(split == part)
            counts = np.empty((len(indices), len(genes)), dtype=np.float32)
            cond = np.empty((len(indices), len(features)), dtype=np.float32)
            for start in range(0, len(indices), 16384):
                idx = indices[start:start + 16384]
                sl = slice(int(idx[0]), int(idx[-1]) + 1)
                keep = idx - idx[0]
                counts[start:start + len(idx)] = sparse_counts[sl][keep].toarray()
                cond[start:start + len(idx)] = encoded_conditions[sl][keep]
            result[part] = (
                torch.as_tensor(counts, device=device),
                torch.as_tensor(cond, device=device),
                torch.as_tensor(context_codes[indices], device=device),
                torch.as_tensor(obs["is_control"].to_numpy(dtype=bool)[indices], device=device),
            )
            print(f"Loaded {len(indices):,} {part} cells", flush=True)
    return result


def fit_vae(model, fit_data, cfg, run_dir):
    train_counts, _, _, _ = fit_data["train"]
    valid_counts, _, _, _ = fit_data["valid"]
    target_sum = cfg["data"]["target_sum"]
    settings = cfg["training"]
    params = list(model.rna_encoder.parameters()) + list(model.decoder.parameters())
    opt = torch.optim.AdamW(params, lr=settings["vae_lr"], weight_decay=1e-5)
    batch_size = settings["batch_size"]
    best = float("inf")
    best_epoch = 0
    history = []
    device = train_counts.device
    cache = run_dir / "vae.pt"
    if cache.exists():
        state = torch.load(cache, map_location=device, weights_only=True)
        model.rna_encoder.load_state_dict(state["encoder"])
        model.decoder.load_state_dict(state["decoder"])
        print(f"Restored RNA VAE, best validation epoch {state['epoch']}", flush=True)
        return state["epoch"]

    def objective(counts, training):
        out = model.rna_encoder(normalize(counts, target_sum), sample=training)
        mu, theta = model.decoder(out["z"], counts.sum(dim=1, keepdim=True))
        nll = model.decoder.reconstruction_nll(counts, mu, theta)
        kl = kl_standard_normal(out["z_mean"], out["z_logvar"])
        return nll, nll + settings["kl_weight"] * kl

    for epoch in range(1, settings["vae_epochs"] + 1):
        model.train()
        total = torch.zeros((), device=device)
        for idx in torch.randperm(len(train_counts), device=device).split(batch_size):
            opt.zero_grad(set_to_none=True)
            nll, loss = objective(train_counts[idx], True)
            loss.backward()
            clip_grad_norm_(params, settings["grad_clip_norm"])
            opt.step()
            total += nll.detach() * len(idx)
        model.eval()
        with torch.no_grad():
            valid = sum(objective(batch, False)[0] * len(batch)
                        for batch in valid_counts.split(batch_size)).item() / len(valid_counts)
        train = total.item() / len(train_counts)
        if not np.isfinite([train, valid]).all():
            raise RuntimeError("Nonfinite VAE loss")
        history.append({"epoch": epoch, "train_nb": train, "validation_nb": valid})
        pd.DataFrame(history).to_csv(run_dir / "vae_history.csv", index=False)
        print(f"RNA VAE {epoch:02d}: train NB {train:.4f}; validation NB {valid:.4f}", flush=True)
        if valid < best:
            best, best_epoch = valid, epoch
            tmp = cache.with_suffix(".tmp")
            torch.save({"encoder": model.rna_encoder.state_dict(), "decoder": model.decoder.state_dict(),
                        "epoch": epoch}, tmp)
            tmp.replace(cache)
        elif epoch - best_epoch >= settings["patience"]:
            break
    state = torch.load(cache, map_location=device, weights_only=True)
    model.rna_encoder.load_state_dict(state["encoder"])
    model.decoder.load_state_dict(state["decoder"])
    return best_epoch


@torch.no_grad()
def encode_split(model, counts, config):
    model.eval()
    encoded = [model.rna_encoder(normalize(batch, config["data"]["target_sum"]), sample=False)
               for batch in counts.split(config["training"]["batch_size"])]
    return (torch.cat([row["z_mean"] for row in encoded]),
            torch.cat([row["z_logvar"] for row in encoded]))


def fit(config, run_dir):
    settings = config["training"]
    set_seed(settings["seed"])
    device = resolve_device(settings["device"])
    torch.set_float32_matmul_precision("high")
    obs, genes, features = metadata(config)
    context_names, context_codes = np.unique(contexts(obs), return_inverse=True)
    fit_data = load_fit_data(config, obs, genes, features, context_codes, device)
    model = BacDiffModel(len(genes), len(features), config).to(device)
    vae_epoch = fit_vae(model, fit_data, config, run_dir)

    train_counts, train_cond, train_contexts, controls = fit_data["train"]
    valid_counts, valid_cond, valid_contexts, valid_controls = fit_data["valid"]
    train_means, train_logvars = encode_split(model, train_counts, config)
    valid_means, valid_logvars = encode_split(model, valid_counts, config)
    anchors, depths = [], []
    for i, name in enumerate(context_names):
        keep = controls & (train_contexts == i)
        if not int(keep.sum()):
            raise ValueError(f"No training controls for {name}")
        anchors.append(train_means[keep].mean(dim=0))
        depths.append(train_counts[keep].sum(dim=1).clamp_min(1).cpu())
    anchors = torch.stack(anchors)
    del fit_data, train_counts, valid_counts

    def treated(means, logvars, cond, code, is_control):
        keep = ~is_control
        return means[keep] - anchors[code[keep]], logvars[keep], cond[keep], code[keep]

    train_z, train_lv, train_cond, train_contexts = treated(train_means, train_logvars, train_cond, train_contexts, controls)
    valid_z, valid_lv, valid_cond, valid_contexts = treated(valid_means, valid_logvars, valid_cond, valid_contexts, valid_controls)
    scale = (train_z.var(dim=0, unbiased=False) + train_lv.exp().mean(dim=0)).sqrt().clamp_min(0.05)
    params = list(model.bacteria_encoder.parameters()) + list(model.diffusion.parameters())
    opt = torch.optim.AdamW(params, lr=settings["diffusion_lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=settings["diffusion_epochs"], eta_min=1e-4)
    batch_size = settings["batch_size"]
    val_rng = torch.Generator(device=device).manual_seed(settings["seed"] + 1)
    val_t = torch.randint(model.diffusion.timesteps, (len(valid_z),), generator=val_rng, device=device)
    val_latent = valid_z + torch.randn(valid_z.shape, generator=val_rng, device=device) * (0.5 * valid_lv).exp()
    val_noise = torch.randn(valid_z.shape, generator=val_rng, device=device)
    val_noisy, _ = model.diffusion.q_sample(val_latent / scale, val_t, val_noise)

    best = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, settings["diffusion_epochs"] + 1):
        model.train()
        total = torch.zeros((), device=device)
        for idx in torch.randperm(len(train_z), device=device).split(batch_size):
            opt.zero_grad(set_to_none=True)
            z = train_z[idx]
            delta = (z + torch.randn_like(z) * (0.5 * train_lv[idx]).exp()) / scale
            h = model.bacteria_encoder(train_cond[idx])
            condition = torch.cat([anchors[train_contexts[idx]], h], dim=-1)
            loss = model.diffusion.diffusion_loss(delta, condition)["diffusion_loss"]
            loss.backward()
            clip_grad_norm_(params, settings["grad_clip_norm"])
            opt.step()
            total += loss.detach() * len(idx)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            score = torch.zeros((), device=device)
            for start in range(0, len(valid_z), batch_size):
                sl = slice(start, start + batch_size)
                h = model.bacteria_encoder(valid_cond[sl])
                pred = model.diffusion.predict_epsilon(val_noisy[sl], val_t[sl],
                                                       torch.cat([anchors[valid_contexts[sl]], h], dim=-1))
                score += (pred - val_noise[sl]).square().sum()
            valid = score.item() / valid_z.numel()
        train = total.item() / len(train_z)
        if not np.isfinite([train, valid]).all():
            raise RuntimeError("Nonfinite diffusion loss")
        history.append({"epoch": epoch, "train_noise_mse": train, "validation_noise_mse": valid})
        pd.DataFrame(history).to_csv(run_dir / "diffusion_history.csv", index=False)
        print(f"Diffusion {epoch:02d}: train MSE {train:.6f}; validation MSE {valid:.6f}", flush=True)
        if valid < best:
            best, best_epoch = valid, epoch
            tmp = run_dir / "best_model.tmp"
            torch.save({"model": model.state_dict(), "config": config, "genes": genes, "features": features,
                        "anchors": anchors.cpu(), "residual_scale": scale.cpu(),
                        "contexts": context_names.tolist(), "control_depths": depths,
                        "vae_epoch": vae_epoch, "diffusion_epoch": best_epoch}, tmp)
            tmp.replace(run_dir / "best_model.pt")
        elif epoch - best_epoch >= settings["patience"]:
            break
    print(f"Selected epochs: VAE {vae_epoch}, diffusion {best_epoch}", flush=True)


@torch.no_grad()
def generate(config, run_dir, split):
    set_seed(config["training"]["seed"] + (2 if split == "ood" else 3))
    device = resolve_device(config["training"]["device"])
    torch.set_float32_matmul_precision("high")
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    obs, genes, features = metadata(config)
    if genes != ckpt["genes"] or features != ckpt["features"]:
        raise ValueError("Checkpoint and benchmark feature order disagree")
    model = BacDiffModel(len(genes), len(features), ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mask = (obs[config["data"]["split_key"]].astype(str).to_numpy() == split) & ~obs["is_control"].to_numpy(dtype=bool)
    part = obs.loc[mask, ["bacteria_label", "cell_type", "time_point"]].copy()
    part["row"] = np.flatnonzero(mask)
    part["context"] = contexts(part)
    grid = part.sort_values("bacteria_label").drop_duplicates("bacteria_label")
    if part.groupby("bacteria_label", observed=True)["context"].nunique().max() != 1:
        raise ValueError("A target label spans multiple control contexts")
    with h5py.File(config["data"]["merged_h5ad"]) as handle:
        rows = grid["row"].to_numpy()
        order = np.argsort(rows)
        cond = handle["obsm/bacteria_raw"][rows[order]][np.argsort(order)]
    context_ids = grid["context"].map({k: i for i, k in enumerate(ckpt["contexts"])}).to_numpy()
    if np.any(pd.isna(context_ids)):
        raise ValueError("No training reference for a target context")
    n = config["generation"]["cells_per_condition"]
    context_ids = torch.as_tensor(context_ids, device=device).repeat_interleave(n)
    condition_rows = torch.as_tensor(cond, device=device).repeat_interleave(n, dim=0)
    depth = torch.empty(len(context_ids), 1, device=device)
    for i, values in enumerate(ckpt["control_depths"]):
        is_context = context_ids == i
        pick = torch.randint(len(values), (int(is_context.sum()),), device=device)
        depth[is_context, 0] = values[pick]

    batch_size = config["generation"]["batch_size"]
    batches = []
    for start in range(0, len(context_ids), batch_size):
        sl = slice(start, start + batch_size)
        base = ckpt["anchors"][context_ids[sl]]
        h = model.bacteria_encoder(condition_rows[sl])
        delta = model.diffusion.sample_delta(torch.cat([base, h], dim=-1), num_samples=1,
                                             guidance_scale=config["diffusion"]["guidance_scale"])
        mu, theta = model.decoder(base + delta * ckpt["residual_scale"], depth[sl])
        counts = model.decoder.sample_counts(mu, theta)
        batches.append(normalize(counts, config["data"]["target_sum"]).cpu().numpy())
    expression = np.concatenate(batches)
    if not np.isfinite(expression).all():
        raise RuntimeError("Nonfinite generated expression")
    out = run_dir / ("predictions.npz" if split == "ood" else f"predictions_{split}.npz")
    tmp = out.with_name(f"{out.stem}.tmp.npz")
    np.savez_compressed(tmp, expression=expression, labels=np.repeat(grid["bacteria_label"].to_numpy(dtype=str), n),
                        gene_names=np.array(genes, dtype=str))
    tmp.replace(out)
    print(f"Wrote {len(expression):,} independent cells for {len(grid)} {split} groups to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config_sciplex3_pair_ood.yaml")
    parser.add_argument("--mode", choices=["fit", "generate", "both"], default="both")
    parser.add_argument("--split", choices=["valid", "ood"], default="ood")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {"fit", "both"}:
        fit(config, run_dir)
    if args.mode in {"generate", "both"}:
        generate(config, run_dir, args.split)


if __name__ == "__main__":
    main()
