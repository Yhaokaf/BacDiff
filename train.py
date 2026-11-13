from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data import create_dataloader, prepare_split, set_seed
from losses import compute_total_loss
from model import BacDiffModel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train BacDiff: NB-VAE + conditional residual diffusion.")
    parser.add_argument("--config", required=True, type=str, help="Path to config.yaml")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields in a batch to the target device."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of metric dictionaries."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def create_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Create the optimizer from config."""
    opt_cfg = config["optimizer"]
    name = str(opt_cfg["name"]).lower()
    lr = float(opt_cfg["learning_rate"])
    weight_decay = float(opt_cfg["weight_decay"])
    betas = tuple(float(x) for x in opt_cfg.get("betas", [0.9, 0.999]))

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    raise ValueError(f"Unsupported optimizer: {opt_cfg['name']}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Create the optional learning-rate scheduler."""
    sched_cfg = config["scheduler"]
    sched_type = str(sched_cfg["type"]).lower()

    if sched_type == "none":
        return None
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"]["epochs"]),
            eta_min=float(sched_cfg.get("min_lr", 1e-6)),
        )
    if sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 10)),
            gamma=float(sched_cfg.get("gamma", 0.5)),
        )
    if sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 5)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )
    raise ValueError(f"Unsupported scheduler type: {sched_cfg['type']}")


def sanity_check_batch(model: BacDiffModel, batch: dict[str, Any], device: torch.device) -> None:
    """Run one batch through the model and print key shapes."""
    print("[sanity] batch shapes:")
    for key in ["x_input", "x_counts", "bac_raw", "bac_total_load"]:
        if key in batch and isinstance(batch[key], torch.Tensor):
            print(f"  - {key}: {tuple(batch[key].shape)}")
    if "is_none" in batch:
        print(f"  - is_none: {tuple(batch['is_none'].shape)}, none_count={batch['is_none'].sum().item()}")

    batch = move_batch_to_device(batch, device)
    model.eval()
    with torch.no_grad():
        outputs = model(batch, mode="eval")
    print(f"[sanity] z: {tuple(outputs.z.shape)}")
    print(f"[sanity] z_base: {tuple(outputs.z_base.shape)}")
    print(f"[sanity] h_bac: {tuple(outputs.h_bac.shape)}")
    print(f"[sanity] mu: {tuple(outputs.mu.shape)}")

    tensors_to_check = [outputs.z, outputs.h_bac, outputs.delta_pred, outputs.mu, outputs.theta]
    if any(torch.isnan(t).any() for t in tensors_to_check):
        raise ValueError("NaN detected during initial batch sanity check.")


def run_epoch(
    model: BacDiffModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    device: torch.device,
    config: dict[str, Any],
    split: str,
) -> dict[str, float]:
    """Run one training or validation epoch."""
    training = optimizer is not None
    model.train(training)
    metrics_rows: list[dict[str, float]] = []

    use_amp = bool(config["training"]["use_amp"]) and device.type == "cuda"
    grad_clip = float(config["training"]["grad_clip_norm"])
    iterator = tqdm(loader, desc=split, leave=False)

    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast(enabled=use_amp):
                outputs = model(batch, mode="train" if training else "eval")
                loss, batch_metrics = compute_total_loss(outputs, batch, config["loss_weights"])

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss detected on split={split}: {float(loss.detach().cpu())}")

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()

        metrics_rows.append(batch_metrics)
        iterator.set_postfix(loss=f"{batch_metrics['total_loss']:.4f}", recon=f"{batch_metrics['recon_loss']:.4f}")

    return mean_metrics(metrics_rows)


def step_scheduler(
    scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scheduler_type: str,
    val_loss: float,
) -> None:
    """Step the scheduler using the correct interface."""
    if scheduler is None:
        return
    if scheduler_type == "plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()


def save_checkpoint(
    path: Path,
    model: BacDiffModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: dict[str, Any],
    gene_names: list[str],
    bacteria_names: list[str],
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "config": config,
        "gene_names": gene_names,
        "bacteria_names": bacteria_names,
        "mean_z_base": model.mean_z_base.cpu(),
    }
    torch.save(checkpoint, path)


def save_loss_curve(history_df: pd.DataFrame, path: Path) -> None:
    """Save a loss-curve plot from training history."""
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_total_loss"], label="train_total_loss")
    plt.plot(history_df["epoch"], history_df["val_total_loss"], label="val_total_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    """Entry point for model training with a single train/val/test split."""
    args = parse_args()
    config = load_config(args.config)

    run_name = str(config["run_name"])
    run_dir = Path(config["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config.setdefault("output", {})
    config["output"]["root_dir"] = str(run_dir)
    config["output"].setdefault("checkpoint_name", "best_model.pt")
    config["output"].setdefault("history_name", "train_log.csv")

    with open(run_dir / "used_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    seed = int(config["training"]["seed"])
    set_seed(seed, deterministic=bool(config["training"]["deterministic"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare single 7:1:2 split
    train_dataset, val_dataset, test_dataset, split_info = prepare_split(config)

    n_genes = train_dataset.x_input.shape[1]
    n_bacteria = train_dataset.bac_raw.shape[1]

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    max_epochs = int(config["training"]["epochs"])
    early_cfg = config["early_stopping"]
    patience = int(early_cfg["patience"])
    min_delta = float(early_cfg.get("min_delta", 0.0))

    print(f"\n{'='*60}")
    print(f"Training: train={split_info['train_size']}, val={split_info['val_size']}, test={split_info['test_size']}")
    print(f"  Train none: {split_info['train_none']}, Val none: {split_info['val_none']}, Test none: {split_info['test_none']}")
    print(f"{'='*60}")

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Create model
    model = BacDiffModel(
        input_dim=n_genes,
        bacteria_dim=n_bacteria,
        config=config,
    ).to(device)

    # Compute initial mean_z_base from train none cells
    _compute_and_store_mean_z_base(model, train_loader, device)

    # Sanity check
    first_batch = next(iter(train_loader))
    sanity_check_batch(model, first_batch, device)

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    scheduler_type = str(config["scheduler"]["type"]).lower()

    use_amp = bool(config["training"]["use_amp"]) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int]] = []

    best_model_path = run_dir / "best_model.pt"
    log_csv_path = run_dir / "train_log.csv"
    loss_curve_path = run_dir / "loss_curve.png"

    for epoch in range(1, max_epochs + 1):
        # Recompute mean_z_base at start of each epoch
        _compute_and_store_mean_z_base(model, train_loader, device)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            split="train",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                scaler=scaler,
                device=device,
                config=config,
                split="val",
            )

        step_scheduler(scheduler, scheduler_type, val_metrics["total_loss"])

        row: dict[str, float | int] = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history_rows.append(row)
        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(log_csv_path, index=False)
        save_loss_curve(history_df, loss_curve_path)

        # Save best checkpoint
        improved = val_metrics["total_loss"] < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_metrics["total_loss"],
                config=config,
                gene_names=[],
                bacteria_names=[],
            )
        else:
            epochs_without_improvement += 1

        print(
            f"  epoch={epoch:03d} "
            f"train_loss={train_metrics['total_loss']:.4f} "
            f"recon={train_metrics['recon_loss']:.4f} "
            f"kl={train_metrics['kl_loss']:.4f} "
            f"diff={train_metrics['diffusion_loss']:.4f} "
            f"none_diff={train_metrics['none_diffusion_loss']:.4f} | "
            f"val_loss={val_metrics['total_loss']:.4f} "
            f"recon={val_metrics['recon_loss']:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch}. Best={best_epoch}, loss={best_val_loss:.6f}")
            break

    # Summary
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val_loss: {best_val_loss:.6f}")
    print(f"  Checkpoint: {best_model_path}")


@torch.no_grad()
def _compute_and_store_mean_z_base(
    model: BacDiffModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """Compute mean z_base from none cells in the train loader and store in model."""
    model.eval()
    z_means = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        is_none = batch["is_none"]
        if is_none.any():
            x_input = batch["x_input"][is_none]
            encoded = model.rna_encoder(x_input=x_input, sample=False)
            z_means.append(encoded["z_mean"].cpu())
    if z_means:
        all_z = torch.cat(z_means, dim=0)
        model.mean_z_base.copy_(all_z.mean(dim=0).to(device))
    model.train()


if __name__ == "__main__":
    main()
