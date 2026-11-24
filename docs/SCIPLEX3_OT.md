# Sci-Plex 3 OT experiment

This experiment adapts BacDiff to an unpaired Sci-Plex 3 screen.

## Data and split

The preparation script reads the GEO GSM4150378 matrix and annotations. It
selects HVGs using training cells only, keeps the original count layer for the
negative-binomial decoder, and writes a normalized/log-transformed expression
matrix for scoring. The default split holds out complete `(cell line, drug)`
pairs. All doses, time points, and biological replicates for a pair remain in
the same split. Vehicle cells remain training references.

This split measures known-drug transfer to a held-out cell-line/drug pair. It
does not measure cold-start prediction for an entirely unseen drug, dose
extrapolation, or replicate/plate generalization.

## Training

There are no measured before/after cell pairs in Sci-Plex 3. The VAE is fit on
training vehicle and treated cells and then frozen. For each treated training
cell, the OT arm does the following:

1. Restricts candidate vehicle cells to the same cell-line and exposure-time
   context.
2. Computes a latent-space squared-distance cost using the frozen VAE means.
3. Solves a balanced entropic Sinkhorn coupling in a minibatch.
4. Samples a vehicle source from the treated row's conditional coupling.
5. Forms `z_treated - z_vehicle` and trains the conditional DDPM on that latent
   residual.

The candidate vehicle can be reused, and the assignment is a distribution-level
pseudo-pair. It must not be interpreted as the true trajectory of one cell.
At inference, OT is not run: the model receives a matched training vehicle
latent plus treatment metadata and generates treated profiles.

## Reproduction

```bash
python checking_data/prepare_sciplex3_pair_ood.py \
  --raw-dir data/sciplex3_raw \
  --output data/sciplex3_pair_ood_500hvg.h5ad

python -m bacdiff.cli.sciplex3_pair \
  --config configs/config_sciplex3_pair_ood.yaml --mode both --split ood
```

The matching ablation uses the same frozen VAE and compares `centroid`,
`random`, `ot`, and `otfm` source construction/objectives:

```bash
python -m bacdiff.cli.sciplex3_pair_ablation fit \
  --config configs/config_sciplex3_pair_ablation.yaml --arm ot
python -m bacdiff.cli.sciplex3_pair_ablation predict \
  --config configs/config_sciplex3_pair_ablation.yaml --arm ot --split ood

python -m bacdiff.cli.sciplex3_pair_ablation fit \
  --config configs/config_sciplex3_pair_ablation.yaml --arm otfm
python -m bacdiff.cli.sciplex3_pair_ablation predict \
  --config configs/config_sciplex3_pair_ablation.yaml --arm otfm --split ood
```

The `otfm` control keeps the same frozen VAE, source/target OT assignments,
posterior target sampling, condition encoder, NB decoder, validation scorer,
and optimizer-step budget as the `ot` residual-DDPM arm. It replaces only the
DDPM noise-prediction loss and reverse sampler. For a pseudo-pair with source
and target latent states `z0` and `z1`, context anchor `a`, and train-fitted
scale `s`, it uses

```text
r0 = (z0 - a) / s
r1 = (z1 - a) / s
rt = (1 - t) * r0 + t * r1 + sigma * epsilon
u  = r1 - r0
loss = mean((v_theta(rt, t, condition) - u) ** 2)
```

Here `t` is sampled uniformly from zero to one, `epsilon` is standard normal,
and `sigma=1.5` follows the CellFlow OTFM adapter used in this benchmark. At
inference, RK4 integrates the learned velocity field from the sampled vehicle
residual. The endpoint is added back to the context anchor and decoded by the
same frozen NB decoder. The velocity field is deterministic conditional on a
source cell; population variability therefore comes from the sampled vehicle
source and decoder sampling, unlike the DDPM's additional latent noise.

This is a controlled latent-space OTFM adaptation, not a reproduction of the
full CellFlow implementation: the representation, condition encoder, and
decoder remain BacDiff's. The validation-selected checkpoint is selected with
the same distribution metrics as the other arms; test metrics are not used for
model selection.

The scorer accepts generated `.npz` files with `--pred NAME=PATH` and writes
condition- and pair-level CSV summaries. It compares predicted treated
populations with real treated populations using a shared train-fitted PCA and
RBF kernel calibration. `Delta Pearson` is a population treatment-effect
direction diagnostic; it is not a single-cell paired accuracy measure.

For rigorous follow-up comparisons, score biological replicates separately and
add dose-OOD, whole-drug-OOD, and replicate/plate-OOD splits. The current OT
artifacts are intentionally not committed because they include multi-gigabyte
datasets and model outputs.
