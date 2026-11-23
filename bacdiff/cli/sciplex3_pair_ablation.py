"""Compare a vehicle centroid, random cell pairs and OT cell pairs in BacDiff."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

from bacdiff.cli.sciplex3_pair import contexts, metadata, normalize
from bacdiff.models.model import BacDiffModel
from bacdiff.utils.config import load_config
from bacdiff.utils.runtime import resolve_device, set_seed


ARMS = ("centroid", "random", "ot")


def take_dense(dataset, rows):
    """Read arbitrary row order from an HDF5 dense matrix."""
    unique, inverse = np.unique(rows, return_inverse=True)
    return np.asarray(dataset[unique][inverse], dtype=np.float32)


def save_checkpoint(value, path):
    tmp = path.with_suffix(".partial")
    torch.save(value, tmp)
    tmp.replace(path)


def model_for(data, cfg, device):
    model = BacDiffModel(len(data["genes"]), len(data["features"]), cfg).to(device)
    model.rna_encoder.load_state_dict(data["encoder"])
    model.decoder.load_state_dict(data["decoder"])
    model.rna_encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    return model


@torch.inference_mode()
def prepare(cfg, options, output, device):
    from anndata.io import sparse_dataset

    from bacdiff.data.sciplex3_ablation_validation import build_validation_reference

    source_path = Path(options["frozen_checkpoint"])
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    obs, genes, features = metadata(cfg)
    if source["genes"] != genes or source["features"] != features or source["config"]["model"] != cfg["model"]:
        raise ValueError("Frozen VAE was trained with different model settings or features")
    with source_path.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    data = {"genes": genes, "features": features, "source_sha256": source_hash,
            "source_checkpoint": str(source_path), "vae_epoch": source["vae_epoch"],
            "data_config": cfg["data"], "model_config": cfg["model"],
            "encoder": source["model"], "decoder": source["model"]}
    for key, prefix in (("encoder", "rna_encoder."), ("decoder", "decoder.")):
        data[key] = {name.removeprefix(prefix): value for name, value in source["model"].items() if name.startswith(prefix)}
    del source
    split = obs[cfg["data"]["split_key"]].astype(str).to_numpy()
    train_rows = np.flatnonzero(split == "train")
    valid_rows = np.flatnonzero(split == "valid")
    train_obs = obs.iloc[train_rows]
    context_names, train_ctx = np.unique(contexts(train_obs), return_inverse=True)
    control_mask = torch.from_numpy(train_obs["is_control"].to_numpy(dtype=bool, copy=True))
    condition_ids, condition_names = pd.factorize(train_obs["bacteria_label"], sort=True)
    _, condition_first = np.unique(condition_ids, return_index=True)
    model = model_for(data, cfg, device).eval()
    means, logvars, control_depth = [], [], []
    with h5py.File(cfg["data"]["merged_h5ad"], "r") as handle:
        counts = sparse_dataset(handle["layers/counts"])
        for start in range(0, len(train_rows), 16_384):
            idx = train_rows[start:start + 16_384]
            first, stop = int(idx[0]), int(idx[-1]) + 1
            matrix = counts[first:stop][idx - first]
            ctl = control_mask[start:start + len(idx)].numpy()
            if ctl.any():
                control_depth.append(torch.from_numpy(np.asarray(matrix[ctl].sum(axis=1), dtype=np.float32).ravel()))
            for batch in torch.as_tensor(matrix.toarray(), device=device, dtype=torch.float32).split(cfg["training"]["batch_size"]):
                encoded = model.rna_encoder(normalize(batch, cfg["data"]["target_sum"]), sample=False)
                means.append(encoded["z_mean"].cpu())
                logvars.append(encoded["z_logvar"].cpu())
        train_mean, train_logvar = torch.cat(means), torch.cat(logvars)
        if not torch.isfinite(train_mean).all() or not torch.isfinite(train_logvar).all():
            raise ValueError("Nonfinite frozen encoder outputs")
        print(f"Encoded {len(train_mean):,} training cells with frozen VAE epoch {data['vae_epoch']}", flush=True)
        condition_features = torch.from_numpy(take_dense(handle["obsm/bacteria_raw"], train_rows[condition_first]))
        validation = build_validation_reference(handle, obs, train_rows, valid_rows, cfg["training"]["seed"], options["validation_cells"])
    train_ctx = torch.from_numpy(train_ctx)
    keep = ~control_mask
    anchors = torch.stack([train_mean[control_mask & train_ctx.eq(i)].mean(dim=0) for i in range(len(context_names))])
    residual = train_mean[keep] - anchors[train_ctx[keep]]
    scale = (residual.var(dim=0, unbiased=False) + train_logvar[keep].exp().mean(dim=0)).sqrt().clamp_min(0.05)
    data.update(train_mean=train_mean, train_logvar=train_logvar, train_context=train_ctx, controls=control_mask,
                control_depth=torch.cat(control_depth).clamp_min(1), condition_ids=torch.from_numpy(condition_ids),
                condition_features=condition_features, context_names=context_names.tolist(), anchors=anchors, scale=scale,
                validation=validation, validation_cells=options["validation_cells"], seed=cfg["training"]["seed"])
    if len(condition_features) != len(condition_names) or len(data["control_depth"]) != int(control_mask.sum()):
        raise ValueError("Prepared row counts do not match training metadata")
    save_checkpoint(data, output / "prepared.pt")
    print(f"Saved shared inputs with {len(validation['labels'])} validation conditions to {output / 'prepared.pt'}", flush=True)


def inference_panel(data, cfg, split, seed, draws=None):
    """Make shared input rows and independently draw decoder depth from training controls."""
    obs, genes, features = metadata(cfg)
    if genes != data["genes"] or features != data["features"]:
        raise ValueError("Benchmark feature order changed after preparation")
    keep = obs[cfg["data"]["split_key"]].eq(split).to_numpy() & ~obs["is_control"].to_numpy()
    selected = obs.loc[keep, ["bacteria_label", "cell_type", "time_point"]].copy()
    selected["row"] = np.flatnonzero(keep)
    selected["context"] = contexts(selected)
    grid = selected.sort_values("bacteria_label", kind="stable").drop_duplicates("bacteria_label")
    labels = grid["bacteria_label"].to_numpy(dtype=str)
    if draws is None:
        draws = np.full(len(labels), cfg["generation"]["cells_per_condition"], dtype=np.int64)
    else:
        if labels.tolist() != data["validation"]["labels"]:
            raise ValueError("Validation sampling labels do not match prediction order")
        draws = np.asarray(draws, dtype=np.int64)
    context_ids = grid["context"].map({key: i for i, key in enumerate(data["context_names"])}).to_numpy()
    if pd.isna(context_ids).any():
        raise ValueError("No matched training control context")
    with h5py.File(cfg["data"]["merged_h5ad"], "r") as handle:
        features = take_dense(handle["obsm/bacteria_raw"], grid["row"].to_numpy())
    control_rows = torch.nonzero(data["controls"], as_tuple=True)[0]
    pools = {i: torch.nonzero(data["train_context"][control_rows].eq(i), as_tuple=True)[0].numpy()
             for i in range(len(data["context_names"]))}
    source_rng, depth_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    sources, depths = [], []
    for ctx, n in zip(context_ids, draws):
        pool = pools[ctx]
        sources.append(source_rng.choice(pool, n, replace=len(pool) < n))
        depths.append(depth_rng.choice(pool, n, replace=True))
    return {"labels": np.repeat(labels, draws), "features": torch.from_numpy(np.repeat(features, draws, axis=0)),
            "contexts": torch.from_numpy(np.repeat(context_ids, draws)),
            "source": data["train_mean"][control_rows[np.concatenate(sources)]],
            "depth": data["control_depth"][np.concatenate(depths)].reshape(-1, 1)}


@torch.inference_mode()
def sample(model, data, panel, cfg, arm, device, seed):
    model.eval()
    row_batch = cfg["generation"]["batch_size"]
    bases = (data["anchors"][panel["contexts"]] if arm == "centroid" else panel["source"]).to(device)
    scale = data["scale"].to(device)
    features = panel["features"].to(device)
    depths = panel["depth"].to(device)

    def sample_seed(value):
        torch.random.default_generator.manual_seed(value)
        if device.type == "cuda":
            torch.cuda.manual_seed(value)

    latent, expression = [], []
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        sample_seed(seed)
        for base, feat in zip(bases.split(row_batch), features.split(row_batch)):
            cond = torch.cat([base, model.bacteria_encoder(feat)], dim=-1)
            delta = model.diffusion.sample_delta(cond, num_samples=1, guidance_scale=cfg["diffusion"]["guidance_scale"])
            latent.append(base + scale * delta)
        sample_seed(seed + 1)
        for state, depth in zip(latent, depths.split(row_batch)):
            mu, theta = model.decoder(state, depth)
            expression.append(normalize(model.decoder.sample_counts(mu, theta), cfg["data"]["target_sum"]).cpu().numpy())
    values = np.concatenate(expression)
    if not np.isfinite(values).all():
        raise RuntimeError("Generated nonfinite expression")
    return values


def fit(data, cfg, options, output, arm, device, *, training_seed=None):
    from bacdiff.data.sciplex3_ablation_validation import make_validation_scorer

    settings = cfg["training"]
    protocol_seed = data["seed"]
    training_seed = protocol_seed if training_seed is None else training_seed
    set_seed(training_seed)
    model = model_for(data, cfg, device)
    mean, logvar = (data[key].to(device) for key in ("train_mean", "train_logvar"))
    scale, anchors = (data[key].to(device) for key in ("scale", "anchors"))
    ctx, ctrl, codes, features = (data[key].to(device) for key in ("train_context", "controls", "condition_ids", "condition_features"))
    treated = torch.nonzero(~ctrl, as_tuple=True)[0]
    pairer = None
    if arm != "centroid":
        from bacdiff.data.sciplex3_pairing import ConditionPairer

        pairer = ConditionPairer(mean / scale, ctx, ctrl, codes, minibatch_size=options["pair_batch_size"],
                                 ot_epsilon=options["ot_epsilon"], ot_iterations=options["ot_iterations"])
        if not torch.equal(pairer.treated_rows, treated):
            raise ValueError("Paired control mapping does not use the training target order")
    epochs = settings["diffusion_epochs"]
    params = list(model.bacteria_encoder.parameters()) + list(model.diffusion.parameters())
    opt = torch.optim.AdamW(params, lr=settings["diffusion_lr"], weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    order_rng = torch.Generator(device=device).manual_seed(training_seed + 70)
    validation = inference_panel(data, cfg, "valid", protocol_seed + 300, data["validation"]["n_draws"])
    validate = make_validation_scorer(data["validation"])
    run = output / arm
    run.mkdir(parents=True, exist_ok=True)
    best, chosen = None, None
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        anchor = (anchors[ctx[treated]] if pairer is None else
                  mean[pairer.control_rows(arm, training_seed + 100_003 * epoch)])
        total = torch.zeros((), device=device)
        for idx in torch.randperm(len(treated), generator=order_rng, device=device).split(settings["batch_size"]):
            opt.zero_grad(set_to_none=True)
            sel = treated[idx]
            base = anchor[idx]
            z_target = mean[sel] + torch.randn_like(mean[sel]) * (0.5 * logvar[sel]).exp()
            residual = (z_target - base) / scale
            cond = torch.cat([base, model.bacteria_encoder(features[codes[sel]])], dim=-1)
            loss = model.diffusion.diffusion_loss(residual, cond)["diffusion_loss"]
            loss.backward()
            clip_grad_norm_(params, settings["grad_clip_norm"])
            opt.step()
            total += loss.detach() * len(idx)
        schedule.step()
        train_loss = total.item() / len(treated)
        if not np.isfinite(train_loss):
            raise RuntimeError("Nonfinite training loss")
        row = {"epoch": epoch, "train_noise_mse": train_loss}
        status = ""
        if epoch % options["validate_every"] == 0 or epoch == epochs:
            metrics = validate(sample(model, data, validation, cfg, arm, device, protocol_seed + 310))
            key = (metrics["energy"], metrics["mmd2"], -metrics["delta_pearson"])
            if not np.isfinite(key).all():
                raise RuntimeError("Nonfinite validation metrics")
            row.update({f"val_{k}": v for k, v in metrics.items()})
            if best is None or key < best:
                best, chosen = key, row.copy()
                save_checkpoint({"epoch": epoch, "arm": arm, "config": cfg, "model": model.state_dict(),
                                 "training_seed": training_seed, "protocol_seed": protocol_seed,
                                 "source_sha256": data["source_sha256"], "metrics": metrics}, run / "best_model.pt")
            status = f" Energy {key[0]:.5f}, MMD2 {key[1]:.5f}, delta Pearson {-key[2]:.5f}, best {chosen['epoch']}"
        history.append(row)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        print(f"{arm} epoch {epoch:02d}/{epochs} noise MSE {train_loss:.6f}{status} ({time.perf_counter() - start:.1f}s)", flush=True)
    run_meta = {"arm": arm, "seed": training_seed, "protocol_seed": protocol_seed,
                "config": options, "chosen": chosen, "epochs": epochs,
                "checkpoints_every": options["validate_every"], "source_checkpoint": data["source_checkpoint"],
                "source_sha256": data["source_sha256"], "optimizer_steps": epochs * (-(-len(treated) // settings["batch_size"])),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "seconds": time.perf_counter() - start}
    (run / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")


def predict(data, cfg, output, arm, split, device):
    path = output / arm
    best = torch.load(path / "best_model.pt", map_location="cpu", weights_only=True)
    if best["arm"] != arm or best["source_sha256"] != data["source_sha256"]:
        raise ValueError("Selected checkpoint came from a different arm or VAE")
    cfg = best["config"]
    model = model_for(data, cfg, device)
    model.load_state_dict(best["model"])
    seed = data["seed"] + (500 if split == "ood" else 400)
    panel = inference_panel(data, cfg, split, seed)
    expression = sample(model, data, panel, cfg, arm, device, seed + 10)
    name = "predictions.npz" if split == "ood" else "predictions_valid.npz"
    tmp = path / f"{name}.partial.npz"
    np.savez_compressed(tmp, expression=expression, labels=panel["labels"], gene_names=np.asarray(data["genes"], dtype=str))
    tmp.replace(path / name)
    print(f"{arm} epoch {best['epoch']}: generated {len(expression):,} {split} cells to {path / name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit", "predict"))
    parser.add_argument("--config", default="configs/config_sciplex3_pair_ablation.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared", type=Path, help="Shared input cache for fit and predict; defaults to OUTPUT/prepared.pt")
    parser.add_argument("--training-seed", type=int, help="Override only the fit RNG; validation and prediction keep the prepared seed")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--split", choices=("valid", "ood"), default="ood")
    args = parser.parse_args()
    if args.phase != "prepare" and args.arm is None:
        parser.error("--arm is required for fit and predict")
    if args.phase == "prepare" and args.prepared is not None:
        parser.error("--prepared is only for fit and predict; use --output for prepare")
    if args.training_seed is not None and (args.phase != "fit" or not 0 <= args.training_seed < 2**32):
        parser.error("--training-seed requires fit and an integer in [0, 2**32)")
    options = load_config(args.config)
    cfg = load_config(options["model_config"])
    output = args.output or Path(options["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg["training"]["device"])
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    if args.phase == "prepare":
        prepare(cfg, options, output, device)
        return
    prepared_path = args.prepared or output / "prepared.pt"
    data = torch.load(prepared_path, map_location="cpu", weights_only=True)
    if (cfg["data"], cfg["model"], cfg["training"]["seed"], options["validation_cells"]) != (
        data["data_config"], data["model_config"], data["seed"], data["validation_cells"]
    ):
        raise ValueError("Data, model, seed or validation panel changed; rerun prepare")
    if args.phase == "fit":
        fit(data, cfg, options, output, args.arm, device, training_seed=args.training_seed)
    else:
        predict(data, cfg, output, args.arm, args.split, device)


if __name__ == "__main__":
    main()
