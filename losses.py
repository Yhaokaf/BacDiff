from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from model import ModelOutput, NBDecoder


def nb_reconstruction_loss(counts: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Negative Binomial reconstruction loss."""
    return NBDecoder.reconstruction_nll(counts, mu, theta)


def kl_standard_normal(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between q(z|x) and the unit Gaussian prior."""
    kl = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp())
    return kl.sum(dim=-1).mean()


def diffusion_denoising_loss(model_outputs: ModelOutput) -> torch.Tensor:
    """Return the DDPM denoising objective already computed by the model."""
    return model_outputs.diffusion_loss


def smoothness_loss(model_outputs: ModelOutput) -> torch.Tensor:
    """Encourage similar conditions and similar base states to yield similar delta."""
    z_base = model_outputs.z_base
    condition = model_outputs.h_bac
    delta = model_outputs.delta_pred

    batch_size = z_base.shape[0]
    if batch_size < 2:
        return z_base.new_tensor(0.0)

    z_dist = torch.cdist(z_base, z_base, p=2)
    c_dist = torch.cdist(condition, condition, p=2) if condition.shape[1] > 0 else torch.zeros_like(z_dist)
    d_dist = torch.cdist(delta, delta, p=2)

    z_scale = z_dist.detach().mean().clamp_min(1e-6)
    c_scale = c_dist.detach().mean().clamp_min(1e-6) if condition.shape[1] > 0 else z_scale
    weights = torch.exp(-(z_dist / z_scale) - (c_dist / c_scale))
    weights = weights * (1.0 - torch.eye(batch_size, device=weights.device))

    numerator = (weights * d_dist).sum()
    denominator = weights.sum().clamp_min(1e-6)
    return numerator / denominator


def sparsity_loss(delta: torch.Tensor) -> torch.Tensor:
    """Encourage sparse perturbation residuals via an L1 penalty."""
    return delta.abs().mean()


def compute_total_loss(
    model_outputs: ModelOutput,
    batch: dict[str, torch.Tensor],
    loss_weights: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine all training objectives into one scalar loss.

    Only compute losses on bacteria cells (is_none=False), since
    none cells don't have meaningful delta/diffusion.
    """
    counts = batch["x_counts"]  # NB loss requires raw counts (non-negative integers)
    bacteria_mask = ~model_outputs.is_none
    none_mask = model_outputs.is_none

    # Reconstruction loss: on both bacteria and none cells
    if bacteria_mask.any():
        recon_loss_bacteria = nb_reconstruction_loss(
            counts[bacteria_mask], model_outputs.mu[bacteria_mask], model_outputs.theta[bacteria_mask]
        )
    else:
        recon_loss_bacteria = torch.tensor(0.0, device=counts.device)

    if none_mask.any():
        recon_loss_none = nb_reconstruction_loss(
            counts[none_mask], model_outputs.mu[none_mask], model_outputs.theta[none_mask]
        )
    else:
        recon_loss_none = torch.tensor(0.0, device=counts.device)

    recon_loss = recon_loss_bacteria + recon_loss_none

    # KL divergence: on all cells (encourage z ~ N(0,I) for everyone)
    kl_loss = kl_standard_normal(model_outputs.z_mean, model_outputs.z_logvar)

    # Diffusion loss (already computed only on bacteria cells)
    diff_loss = diffusion_denoising_loss(model_outputs)

    # Smoothness loss
    smooth = smoothness_loss(model_outputs)

    # Sparsity loss: only on bacteria cells
    if bacteria_mask.any():
        sparse = sparsity_loss(model_outputs.delta_pred[bacteria_mask])
    else:
        sparse = torch.tensor(0.0, device=counts.device)

    # None-cell diffusion loss: teaches model zero bacteria => zero delta
    none_diff_loss = model_outputs.none_diffusion_loss

    total_loss = (
        float(loss_weights.get("recon", 1.0)) * recon_loss
        + float(loss_weights.get("kl", 0.01)) * kl_loss
        + float(loss_weights.get("diffusion", 1.0)) * diff_loss
        + float(loss_weights.get("none_diffusion", 0.5)) * none_diff_loss
        + float(loss_weights.get("smoothness", 0.01)) * smooth
        + float(loss_weights.get("sparsity", 0.001)) * sparse
    )

    if not torch.isfinite(total_loss):
        raise ValueError("Loss became NaN or Inf during training.")

    metrics = {
        "loss": float(total_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "diffusion_loss": float(diff_loss.detach().cpu()),
        "none_diffusion_loss": float(none_diff_loss.detach().cpu()),
        "smoothness_loss": float(smooth.detach().cpu()),
        "sparsity_loss": float(sparse.detach().cpu()),
    }
    return total_loss, metrics
