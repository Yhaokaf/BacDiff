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
`random`, and `ot` source construction:

```bash
python -m bacdiff.cli.sciplex3_pair_ablation fit \
  --config configs/config_sciplex3_pair_ablation.yaml --arm ot
python -m bacdiff.cli.sciplex3_pair_ablation predict \
  --config configs/config_sciplex3_pair_ablation.yaml --arm ot --split ood
```

The scorer accepts generated `.npz` files with `--pred NAME=PATH` and writes
condition- and pair-level CSV summaries. It compares predicted treated
populations with real treated populations using a shared train-fitted PCA and
RBF kernel calibration. `Delta Pearson` is a population treatment-effect
direction diagnostic; it is not a single-cell paired accuracy measure.

For rigorous follow-up comparisons, score biological replicates separately and
add dose-OOD, whole-drug-OOD, and replicate/plate-OOD splits. The current OT
artifacts are intentionally not committed because they include multi-gigabyte
datasets and model outputs.
