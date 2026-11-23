"""Sample training vehicle references for exact Sci-Plex 3 conditions."""

from __future__ import annotations

import math

import torch


_INDEX_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


class ConditionPairer:
    """Pair unpaired treated cells and vehicles already loaded from the training split.

    All constructor tensors share the original training row order and device.
    An exact treated condition code must identify a single cell line and time
    context. Vehicle condition codes are ignored. The caller provides latent
    means already normalized by a scale fitted using only training rows.
    """

    @torch.no_grad()
    def __init__(
        self,
        train_means: torch.Tensor,
        train_contexts: torch.Tensor,
        train_is_control: torch.Tensor,
        condition_codes: torch.Tensor,
        *,
        minibatch_size: int = 32,
        ot_epsilon: float = 0.1,
        ot_iterations: int = 32,
        max_batches_per_chunk: int = 256,
    ) -> None:
        if train_means.ndim != 2 or not train_means.shape[1] or not train_means.is_floating_point():
            raise ValueError("train_means must be a floating-point matrix with at least one feature")
        n_rows = len(train_means)
        device = train_means.device
        if any(x.shape != (n_rows,) or x.device != device for x in (train_contexts, train_is_control, condition_codes)):
            raise ValueError("Training metadata must be 1D and aligned with train_means on the same device")
        if train_is_control.dtype != torch.bool:
            raise ValueError("train_is_control must be boolean")
        if train_contexts.dtype not in _INDEX_DTYPES or condition_codes.dtype not in _INDEX_DTYPES:
            raise ValueError("Context and exact condition codes must be integers")
        if min(minibatch_size, ot_iterations, max_batches_per_chunk) < 1 or not math.isfinite(ot_epsilon) or ot_epsilon <= 0:
            raise ValueError("Pairing settings must be positive and OT epsilon must be finite")

        self._means = train_means.detach().float()
        if not bool(torch.isfinite(self._means).all()):
            raise ValueError("Training latent means must be finite")
        self._epsilon = ot_epsilon
        self._iterations = ot_iterations
        self._chunk_size = max_batches_per_chunk

        self.treated_rows = torch.where(~train_is_control)[0]
        vehicles = torch.where(train_is_control)[0]
        if not self.treated_rows.numel() or not vehicles.numel():
            raise ValueError("Pairing needs both treated cells and vehicles in the training split")
        context_names, row_context = torch.unique(train_contexts, sorted=True, return_inverse=True)
        condition_names, self._groups, group_counts = torch.unique(
            condition_codes[self.treated_rows], sorted=True, return_inverse=True, return_counts=True
        )

        group_starts = group_counts.cumsum(0) - group_counts
        sorted_targets = torch.argsort(self._groups, stable=True)
        target_context = row_context[self.treated_rows]
        group_context = target_context[sorted_targets[group_starts]]
        if not torch.equal(target_context, group_context[self._groups]):
            raise ValueError("An exact treated condition code spans multiple cell line/time contexts")

        vehicle_context = row_context[vehicles]
        vehicle_counts = torch.bincount(vehicle_context, minlength=context_names.numel())
        if not bool((vehicle_counts[group_context] > 0).all()):
            raise ValueError("A treated context has no vehicle cells in the training split")
        self._vehicles = vehicles[torch.argsort(vehicle_context, stable=True)]

        # Each slot addresses the corresponding position after sorting shuffled
        # treatment rows by group; a final partial batch is masked in Sinkhorn.
        group_batches = (group_counts + minibatch_size - 1) // minibatch_size
        batch_group = torch.repeat_interleave(torch.arange(condition_names.numel(), device=device), group_batches)
        first_batch = (group_batches.cumsum(0) - group_batches)[batch_group]
        batch_starts = group_starts[batch_group] + (torch.arange(len(batch_group), device=device) - first_batch) * minibatch_size
        slots = batch_starts[:, None] + torch.arange(minibatch_size, device=device)
        self._active = slots < (group_starts + group_counts)[batch_group, None]
        self._slots = slots.masked_fill(~self._active, 0)
        self._real_slots = torch.where(self._active.flatten())[0]
        self._batch_sizes = self._active.sum(1, keepdim=True)
        batch_context = group_context[batch_group]
        self._vehicle_sizes = vehicle_counts[batch_context, None]
        self._vehicle_starts = (vehicle_counts.cumsum(0) - vehicle_counts)[batch_context, None]

    def _draw(self, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.treated_rows.device
        rng = torch.Generator(device=device).manual_seed(seed)
        shuffle = torch.randperm(len(self.treated_rows), device=device, generator=rng)
        order = shuffle[torch.argsort(self._groups[shuffle], stable=True)]
        targets = order[self._slots]
        vehicle_uniform = torch.rand(self._slots.shape, generator=rng, device=device, dtype=torch.float32)
        vehicle_pick = (vehicle_uniform * self._vehicle_sizes).long().minimum(self._vehicle_sizes - 1)
        candidates = self._vehicles[self._vehicle_starts + vehicle_pick]
        pair_uniform = torch.rand(self._slots.shape, generator=rng, device=device, dtype=torch.float32)
        return targets, candidates, pair_uniform

    def _ot_choice(
        self, target: torch.Tensor, source: torch.Tensor, active: torch.Tensor, batch_sizes: torch.Tensor, uniform: torch.Tensor
    ) -> torch.Tensor:
        # Scale each cost matrix by its mean real squared distance, so epsilon
        # is relative to the batch; never rescale individual latent features.
        cost = torch.cdist(target, source, compute_mode="donot_use_mm_for_euclid_dist").square_()
        mask = active[:, :, None] & active[:, None, :]
        cost.masked_fill_(~mask, 0)
        mean_cost = cost.sum(dim=(1, 2), keepdim=True) / batch_sizes[:, :, None].square()
        kernel = (-cost / mean_cost.clamp_min_(1e-12) / self._epsilon).clamp_min_(-30).exp_()
        kernel.masked_fill_(~mask, 0)

        # Balanced entropic Sinkhorn with uniform marginals on the real rows
        # and columns. Work up to global scale; flooring the positive kernel
        # and rescaling the dual avoid underflow in the float32 batch products.
        mass = active[:, :, None].to(kernel.dtype)
        transpose = kernel.transpose(1, 2)
        v = mass
        for _ in range(self._iterations):
            u = mass / kernel.bmm(v).clamp_min_(1e-20)
            v = mass / transpose.bmm(u).clamp_min_(1e-20)
            v = v / v.amax(dim=1, keepdim=True)

        # The target-row dual cancels when conditioning on a given treatment.
        cdf = (kernel * v.transpose(1, 2)).cumsum(-1)
        threshold = uniform[:, :, None] * cdf[:, :, -1:]
        return (cdf <= threshold).sum(-1).minimum(batch_sizes - 1)

    @torch.no_grad()
    def control_rows(self, method: str, seed: int) -> torch.Tensor:
        """Return original training vehicle row indices in ``treated_rows`` order.

        Equal seeds on the same device reproduce the treated minibatches,
        uniformly drawn context-matched vehicle candidates (with replacement),
        and categorical uniforms. Random pairing samples the uniform product
        coupling; OT samples each treated cell's conditional of the balanced
        minibatch plan. The resulting assignments can repeat a vehicle row.
        """
        if method not in {"random", "ot"}:
            raise ValueError("Pairing method must be 'random' or 'ot'")
        target, candidates, uniform = self._draw(seed)
        if method == "random":
            positions = (uniform * self._batch_sizes).long().minimum(self._batch_sizes - 1)
            paired = candidates.gather(1, positions)
        else:
            paired = torch.empty_like(candidates)
            for start in range(0, len(target), self._chunk_size):
                part = slice(start, start + self._chunk_size)
                positions = self._ot_choice(
                    self._means[self.treated_rows[target[part]]], self._means[candidates[part]],
                    self._active[part], self._batch_sizes[part], uniform[part],
                )
                paired[part] = candidates[part].gather(1, positions)

        output = torch.empty_like(self.treated_rows)
        return output.scatter_(0, target.flatten()[self._real_slots], paired.flatten()[self._real_slots])
