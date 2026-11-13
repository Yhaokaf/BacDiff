from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_layers: int,
    dropout: float,
    use_layer_norm: bool,
) -> nn.Sequential:
    """Build a simple MLP with optional LayerNorm."""
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1.")

    layers: list[nn.Module] = []
    current = in_dim
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(current, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current = hidden_dim
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class RNAEncoder(nn.Module):
    """MLP encoder producing a single latent z from RNA input."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        layers: int,
        dropout: float,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.backbone = _make_mlp(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        """Sample from a Gaussian posterior using the reparameterization trick."""
        if not sample:
            return mean
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x_input: torch.Tensor,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Encode RNA input into a single latent distribution."""
        hidden = self.backbone(x_input)
        z_mean = self.z_mean(hidden)
        z_logvar = self.z_logvar(hidden).clamp(min=-8.0, max=8.0)
        z = self.reparameterize(z_mean, z_logvar, sample=sample)
        return {
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
        }


class BacteriaEncoder(nn.Module):
    """MLP encoder for bacterial abundance features."""

    def __init__(
        self,
        bacteria_dim: int,
        hidden_dim: int,
        out_dim: int,
        layers: int,
        dropout: float,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()
        self.network = _make_mlp(
            in_dim=bacteria_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, bac_raw: torch.Tensor) -> torch.Tensor:
        """Encode bacterial abundance into a condition embedding."""
        return self.network(bac_raw)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for DDPM timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Map integer timesteps to sinusoidal embeddings."""
        device = t.device
        half_dim = max(self.dim // 2, 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * (-math.log(10000.0) / max(half_dim - 1, 1)))
        angles = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[1] < self.dim:
            pad = self.dim - emb.shape[1]
            emb = torch.cat([emb, torch.zeros((emb.shape[0], pad), device=device, dtype=emb.dtype)], dim=-1)
        return emb


class LatentDiffusion(nn.Module):
    """DDPM-style latent diffusion operating on delta = z_total - z_base."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        time_embed_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        use_layer_norm: bool,
        condition_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.timesteps = timesteps
        self.condition_dropout = condition_dropout

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-12))

        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.epsilon_network = _make_mlp(
            in_dim=latent_dim + condition_dim + time_embed_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_layers=layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def _extract(self, values: torch.Tensor, t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Gather per-timestep coefficients and reshape for broadcasting."""
        return values.gather(0, t).reshape(-1, *([1] * (len(target_shape) - 1)))

    def _maybe_dropout_condition(self, condition: torch.Tensor, force_unconditional: bool = False) -> torch.Tensor:
        """Apply classifier-free guidance style condition dropout."""
        if force_unconditional:
            return torch.zeros_like(condition)
        if self.training and self.condition_dropout > 0.0:
            keep_mask = (torch.rand(condition.shape[0], 1, device=condition.device) > self.condition_dropout).float()
            return condition * keep_mask
        return condition

    def q_sample(
        self,
        delta: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a noisy delta_t from q(delta_t | delta_0)."""
        if noise is None:
            noise = torch.randn_like(delta)
        mean = self._extract(self.sqrt_alphas_cumprod, t, delta.shape) * delta
        std = self._extract(self.sqrt_one_minus_alphas_cumprod, t, delta.shape)
        delta_t = mean + std * noise
        return delta_t, noise

    def predict_epsilon(
        self,
        noisy_delta_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        force_unconditional: bool = False,
    ) -> torch.Tensor:
        """Predict DDPM epsilon for a noisy latent residual."""
        conditioned = self._maybe_dropout_condition(condition, force_unconditional=force_unconditional)
        t_embed = self.time_embedding(t)
        return self.epsilon_network(torch.cat([noisy_delta_t, conditioned, t_embed], dim=-1))

    def _predict_x0_from_epsilon(
        self,
        delta_t: torch.Tensor,
        t: torch.Tensor,
        epsilon_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the clean delta estimate from noisy delta and predicted epsilon."""
        alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, delta_t.shape)
        one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, delta_t.shape)
        return (delta_t - one_minus_alpha_bar_t * epsilon_hat) / alpha_bar_t.clamp_min(1e-8)

    def diffusion_loss(self, delta: torch.Tensor, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the DDPM denoising loss on latent residuals."""
        t = torch.randint(0, self.timesteps, (delta.shape[0],), device=delta.device)
        noisy_delta_t, epsilon = self.q_sample(delta, t)
        epsilon_hat = self.predict_epsilon(noisy_delta_t, t, condition)
        loss = F.mse_loss(epsilon_hat, epsilon)
        delta_pred = self._predict_x0_from_epsilon(noisy_delta_t, t, epsilon_hat)
        return {
            "t": t,
            "noise": epsilon,
            "noisy_delta_t": noisy_delta_t,
            "epsilon_hat": epsilon_hat,
            "delta_pred": delta_pred,
            "diffusion_loss": loss,
        }

    @torch.no_grad()
    def _p_sample(
        self,
        delta_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Single DDPM reverse step."""
        epsilon_cond = self.predict_epsilon(delta_t, t, condition, force_unconditional=False)
        if guidance_scale != 1.0 and self.condition_dropout > 0.0:
            epsilon_uncond = self.predict_epsilon(delta_t, t, condition, force_unconditional=True)
            epsilon_hat = epsilon_uncond + guidance_scale * (epsilon_cond - epsilon_uncond)
        else:
            epsilon_hat = epsilon_cond

        beta_t = self._extract(self.betas, t, delta_t.shape)
        alpha_t = self._extract(self.alphas, t, delta_t.shape)
        alpha_bar_t = self._extract(self.alphas_cumprod, t, delta_t.shape)
        mean = (1.0 / torch.sqrt(alpha_t)) * (delta_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * epsilon_hat)
        if int(t[0].item()) == 0:
            return mean
        noise = torch.randn_like(delta_t)
        variance = self._extract(self.posterior_variance, t, delta_t.shape)
        return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample_delta(
        self,
        condition: torch.Tensor,
        num_samples: int = 1,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample clean delta from the learned DDPM prior."""
        samples: list[torch.Tensor] = []
        for _ in range(max(num_samples, 1)):
            delta_t = torch.randn(condition.shape[0], self.latent_dim, device=condition.device)
            for step in reversed(range(self.timesteps)):
                t = torch.full((condition.shape[0],), step, device=condition.device, dtype=torch.long)
                delta_t = self._p_sample(delta_t, t, condition, guidance_scale=guidance_scale)
            samples.append(delta_t.unsqueeze(1))
        return torch.cat(samples, dim=1).mean(dim=1)


class NBDecoder(nn.Module):
    """Negative Binomial decoder for RNA count reconstruction."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int,
        dropout: float,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.network = _make_mlp(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_theta = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        z_hat: torch.Tensor,
        library_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent state into NB mean and dispersion."""
        hidden = self.network(z_hat)
        proportions = torch.softmax(self.mean_head(hidden), dim=-1)
        mu = proportions * library_size.clamp_min(1.0)
        theta = F.softplus(self.log_theta).unsqueeze(0).expand_as(mu) + 1e-4
        return mu.clamp_min(1e-6), theta

    @staticmethod
    def reconstruction_nll(counts: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Compute the Negative Binomial reconstruction NLL."""
        eps = 1e-8
        mu = mu.clamp_min(eps)
        theta = theta.clamp_min(eps)
        log_prob = (
            torch.lgamma(counts + theta)
            - torch.lgamma(theta)
            - torch.lgamma(counts + 1.0)
            + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
            + counts * (torch.log(mu + eps) - torch.log(theta + mu + eps))
        )
        return -log_prob.sum(dim=-1).mean()

    @torch.no_grad()
    def sample_counts(self, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Sample count realizations from the decoder's NB distribution."""
        probs = theta / (theta + mu)
        distribution = torch.distributions.NegativeBinomial(total_count=theta, probs=probs)
        return distribution.sample()


@dataclass
class ModelOutput:
    z: torch.Tensor              # all cells' latent
    z_mean: torch.Tensor
    z_logvar: torch.Tensor
    z_base: torch.Tensor         # mean z of none cells (broadcast)
    z_total: torch.Tensor        # z of bacteria cells
    delta_true: torch.Tensor     # z_total - z_base
    h_bac: torch.Tensor          # bacteria condition embedding
    diffusion_t: torch.Tensor
    diffusion_noise: torch.Tensor
    epsilon_hat: torch.Tensor
    diffusion_loss: torch.Tensor
    none_diffusion_loss: torch.Tensor  # diffusion loss for none cells (target delta=0)
    delta_pred: torch.Tensor
    z_hat: torch.Tensor          # z_base + delta_pred
    mu: torch.Tensor
    theta: torch.Tensor
    is_none: torch.Tensor        # mask for none cells


class BacDiffModel(nn.Module):
    """NB-VAE + conditional residual diffusion for bacteria-associated latent shifts.

    Training path:
      - Encode all cells → z
      - z_base = mean z of "none" cells in the batch
      - z_total = z of bacteria cells
      - delta = z_total - z_base
      - diffusion(z_base, bacteria) → delta_hat
      - z_hat = z_base + delta_hat → decode → mu, theta

    Inference path:
      - Encode test cells with bacteria → z_total
      - z_base = mean z of training "none" cells
      - diffusion(z_base, bacteria) → delta_hat → z_hat → decode
    """

    def __init__(
        self,
        input_dim: int,
        bacteria_dim: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        diffusion_cfg = config["diffusion"]

        self.input_dim = input_dim
        self.bacteria_dim = bacteria_dim
        self.latent_dim = int(model_cfg["latent_dim"])
        self.bacteria_hidden_dim = int(model_cfg["bacteria_hidden_dim"])

        self.rna_encoder = RNAEncoder(
            input_dim=input_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            latent_dim=self.latent_dim,
            layers=int(model_cfg["encoder_layers"]),
            dropout=float(model_cfg["dropout"]),
            use_layer_norm=bool(model_cfg["use_layer_norm"]),
        )
        self.bacteria_encoder = BacteriaEncoder(
            bacteria_dim=bacteria_dim,
            hidden_dim=self.bacteria_hidden_dim,
            out_dim=self.bacteria_hidden_dim,
            layers=int(model_cfg["bacteria_layers"]),
            dropout=float(model_cfg["dropout"]),
            use_layer_norm=bool(model_cfg["use_layer_norm"]),
        )
        self.diffusion = LatentDiffusion(
            latent_dim=self.latent_dim,
            condition_dim=self.latent_dim + self.bacteria_hidden_dim,
            timesteps=int(diffusion_cfg["timesteps"]),
            beta_start=float(diffusion_cfg["beta_start"]),
            beta_end=float(diffusion_cfg["beta_end"]),
            time_embed_dim=int(diffusion_cfg["time_embed_dim"]),
            hidden_dim=int(diffusion_cfg["hidden_dim"]),
            layers=int(diffusion_cfg["layers"]),
            dropout=float(diffusion_cfg["dropout"]),
            use_layer_norm=bool(diffusion_cfg["use_layer_norm"]),
            condition_dropout=float(diffusion_cfg.get("condition_dropout", 0.0)),
        )
        self.decoder = NBDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            output_dim=input_dim,
            layers=int(model_cfg["decoder_layers"]),
            dropout=float(model_cfg["dropout"]),
            use_layer_norm=bool(model_cfg["use_layer_norm"]),
        )

        # Buffer for precomputed mean z_base from "none" cells (set at eval time)
        self.register_buffer("mean_z_base", torch.zeros(self.latent_dim))

    def forward(self, batch: dict[str, torch.Tensor], mode: str = "train") -> ModelOutput:
        """Run the full model forward pass."""
        sample_latent = mode == "train"
        x_input = batch["x_input"]
        bac_raw = batch["bac_raw"]
        library_size = batch["library_size"]
        is_none = batch["is_none"]  # bool tensor [B]

        # Encode all cells → z
        encoded = self.rna_encoder(x_input=x_input, sample=sample_latent)
        z = encoded["z"]            # [B, latent_dim]
        z_mean = encoded["z_mean"]  # [B, latent_dim]

        # Compute z_base from none cells in this batch
        if is_none.any():
            z_base = z[is_none].mean(dim=0, keepdim=True).expand_as(z)  # [B, latent_dim]
        else:
            # Fallback: use precomputed mean_z_base (for eval when batch has no none cells)
            z_base = self.mean_z_base.unsqueeze(0).expand_as(z)

        # Delta: only meaningful for bacteria cells, but we compute for all
        delta_true = z - z_base  # [B, latent_dim]

        # Bacteria condition
        h_bac = self.bacteria_encoder(bac_raw)  # [B, bacteria_hidden_dim]

        # Diffusion condition: z_base + h_bac
        condition = torch.cat([z_base, h_bac], dim=-1)

        # For bacteria cells, compute diffusion loss
        bacteria_mask = ~is_none
        if bacteria_mask.any():
            delta_bacteria = delta_true[bacteria_mask]
            condition_bacteria = condition[bacteria_mask]
            diffusion_outputs = self.diffusion.diffusion_loss(delta_bacteria, condition_bacteria)
            delta_pred = torch.zeros_like(delta_true)
            delta_pred[bacteria_mask] = diffusion_outputs["delta_pred"]
            diffusion_loss = diffusion_outputs["diffusion_loss"]
            diffusion_t = diffusion_outputs["t"]
            diffusion_noise = diffusion_outputs["noise"]
            epsilon_hat = diffusion_outputs["epsilon_hat"]
        else:
            # No bacteria cells in batch (shouldn't happen in normal training)
            delta_pred = torch.zeros_like(delta_true)
            diffusion_loss = torch.tensor(0.0, device=z.device)
            diffusion_t = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
            diffusion_noise = torch.zeros_like(delta_true)
            epsilon_hat = torch.zeros_like(delta_true)

        # For none cells, compute diffusion loss with target delta=0
        # This teaches the model that zero bacteria => zero perturbation
        none_diffusion_loss = torch.tensor(0.0, device=z.device)
        if is_none.any() and self.training:
            delta_none_target = torch.zeros(is_none.sum(), self.latent_dim, device=z.device)
            condition_none = condition[is_none]
            none_diff_outputs = self.diffusion.diffusion_loss(delta_none_target, condition_none)
            none_diffusion_loss = none_diff_outputs["diffusion_loss"]

        # Reconstruct
        z_hat = z_base + delta_pred
        mu, theta = self.decoder(z_hat, library_size=library_size)

        return ModelOutput(
            z=z,
            z_mean=z_mean,
            z_logvar=encoded["z_logvar"],
            z_base=z_base,
            z_total=z,
            delta_true=delta_true,
            h_bac=h_bac,
            diffusion_t=diffusion_t,
            diffusion_noise=diffusion_noise,
            epsilon_hat=epsilon_hat,
            diffusion_loss=diffusion_loss,
            none_diffusion_loss=none_diffusion_loss,
            delta_pred=delta_pred,
            z_hat=z_hat,
            mu=mu,
            theta=theta,
            is_none=is_none,
        )

    @torch.no_grad()
    def predict(
        self,
        x_input: torch.Tensor,
        bac_raw: torch.Tensor,
        library_size: torch.Tensor,
        is_none: torch.Tensor | None = None,
        num_diffusion_samples: int = 1,
        guidance_scale: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inference: predict gene expression for all cells.

        - Bacteria cells: encode → mean_z_base + diffusion(z_base, bacteria) → decode
        - None cells: encode → use own z → decode (no diffusion)

        Args:
            x_input: [N, genes] RNA input
            bac_raw: [N, bacteria_dim] bacteria abundance
            library_size: [N, 1] library size
            is_none: [N] bool tensor, True for none cells. If None, all treated as bacteria.
            num_diffusion_samples: number of diffusion samples for bacteria cells
        """
        guidance = float(self.config["diffusion"].get("guidance_scale", 1.0) if guidance_scale is None else guidance_scale)

        # Encode all cells
        encoded = self.rna_encoder(x_input=x_input, sample=False)
        z_total = encoded["z"]  # [N, latent_dim]

        if is_none is None:
            is_none = torch.zeros(z_total.shape[0], dtype=torch.bool, device=z_total.device)

        none_mask = is_none
        bact_mask = ~is_none

        # Initialize outputs
        N = z_total.shape[0]
        z_base = torch.zeros_like(z_total)
        delta_hat = torch.zeros_like(z_total)
        z_hat = torch.zeros_like(z_total)

        # Bacteria cells: mean_z_base + diffusion
        if bact_mask.any():
            z_base_bact = self.mean_z_base.unsqueeze(0).expand(bact_mask.sum(), -1)
            z_base[bact_mask] = z_base_bact

            h_bac = self.bacteria_encoder(bac_raw[bact_mask])
            condition = torch.cat([z_base_bact, h_bac], dim=-1)
            delta_bact = self.diffusion.sample_delta(condition, num_samples=num_diffusion_samples, guidance_scale=guidance)
            delta_hat[bact_mask] = delta_bact
            z_hat[bact_mask] = z_base_bact + delta_bact

        # None cells: use own encoder z directly
        if none_mask.any():
            z_hat[none_mask] = z_total[none_mask]

        mu, theta = self.decoder(z_hat, library_size=library_size)

        return {
            "z_base": z_base,
            "z_total": z_total,
            "delta_pred": delta_hat,
            "z_hat": z_hat,
            "recon_mu": mu,
            "recon_theta": theta,
        }

    @torch.no_grad()
    def counterfactual_forward(
        self,
        batch: dict[str, torch.Tensor],
        new_bacteria_batch: dict[str, torch.Tensor],
        num_diffusion_samples: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Counterfactual inference: predict expression with new bacteria condition.

        Args:
            batch: Original batch with x_input, bac_raw, library_size, is_none
            new_bacteria_batch: New bacteria condition (bac_raw, bac_presence, bac_composition, bac_total_load)
            num_diffusion_samples: Number of diffusion samples

        Returns:
            Dictionary with recon_mu, delta_cf, and other outputs
        """
        x_input = batch["x_input"]
        library_size = batch["library_size"]
        is_none = batch["is_none"]

        # Encode RNA to get z_base
        encoded = self.rna_encoder(x_input=x_input, sample=False)
        z_total = encoded["z"]

        # Get z_base: use mean_z_base for bacteria cells, own z for none cells
        N = z_total.shape[0]
        z_base = torch.zeros_like(z_total)
        none_mask = is_none
        bact_mask = ~is_none

        # Bacteria cells: use mean_z_base
        if bact_mask.any():
            z_base_bact = self.mean_z_base.unsqueeze(0).expand(bact_mask.sum(), -1)
            z_base[bact_mask] = z_base_bact

        # None cells: use own z
        if none_mask.any():
            z_base[none_mask] = z_total[none_mask]

        # Initialize delta_cf
        N = z_total.shape[0]
        delta_cf = torch.zeros_like(z_total)

        # Bacteria cells: apply diffusion with new bacteria condition
        if bact_mask.any():
            # Encode new bacteria condition
            bac_raw_new = new_bacteria_batch["bac_raw"]
            h_bac_new = self.bacteria_encoder(bac_raw_new)

            # Diffusion condition: z_base + new bacteria
            condition_new = torch.cat([z_base[bact_mask], h_bac_new], dim=-1)

            # Sample delta from diffusion
            delta_bact = self.diffusion.sample_delta(
                condition_new,
                num_samples=num_diffusion_samples,
                guidance_scale=float(self.config["diffusion"].get("guidance_scale", 1.0)),
            )
            delta_cf[bact_mask] = delta_bact

        # None cells: delta = 0 (no bacterial perturbation)
        # z_hat = z_base + delta
        z_hat = z_base + delta_cf

        # Decode
        mu, theta = self.decoder(z_hat, library_size=library_size)

        return {
            "z_base": z_base,
            "delta_cf": delta_cf,
            "z_hat": z_hat,
            "recon_mu": mu,
            "recon_theta": theta,
        }

    @torch.no_grad()
    def compute_mean_z_base(self, none_x_input: torch.Tensor) -> None:
        """Compute and store mean z_base from "none" cells.

        Call this at the start of each epoch (on train set none cells)
        or once before evaluation (on train set none cells).
        """
        encoded = self.rna_encoder(x_input=none_x_input, sample=False)
        self.mean_z_base.copy_(encoded["z_mean"].mean(dim=0))
