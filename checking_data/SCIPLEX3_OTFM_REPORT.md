# Latent OTFM control on Sci-Plex 3

## Question

Does replacing the BacDiff OT residual DDPM with OT flow matching improve the
same known-drug, held-out cell-line/drug pair task?

## Controlled setup

The experiment reuses the 500 train-selected HVGs, pair-OOD split, frozen
32-dimensional VAE/negative-binomial decoder, OT pseudo-pairs, target posterior
sampling, treatment encoder, source/depth panels, validation scorer, and
optimizer budget from the BacDiff OT arm. Both models use training seed 42,
32 epochs, 4,032 optimizer steps, and the same 4096-sample optimizer batches.
OT pairs are formed in minibatches of 32 within an exact treated condition;
vehicle candidates are restricted to the same cell line and exposure time.

For OTFM, each matched vehicle/treated latent pair is centered on its
train-vehicle context anchor and divided by the same train-fitted residual
scale as BacDiff. The training path is

```text
r0 = (z_vehicle - context_anchor) / scale
r1 = (z_treated - context_anchor) / scale
rt = (1 - t) * r0 + t * r1 + 1.5 * epsilon
u  = r1 - r0
loss = mean((v_theta(rt, t, condition) - u) ** 2)
```

The path uses `t ~ Uniform(0, 1)` and standard-normal `epsilon`. At inference,
the same sampled training vehicle source initializes a 32-step RK4 integration
of the velocity field. The endpoint is decoded by the shared frozen decoder.
The OTFM checkpoint is selected on validation Energy, then MMD squared, then
Delta Pearson. No test score is used for checkpoint selection.

This is a latent-space OTFM adaptation using BacDiff's representation and
decoder. It is not a reproduction of all CellFlow components. Unlike DDPM,
the learned ODE is deterministic conditional on a sampled source cell, so its
cell-level variability comes from source sampling and the NB decoder rather
than an additional latent diffusion draw.

## Validation choices

We tested path noise only on validation. Full validation scoring used 64
predicted cells per condition and the same scorer as the OOD evaluation.

| Arm | Noise sigma | Energy | MMD squared | Delta Pearson |
|---|---:|---:|---:|---:|
| BacDiff OT DDPM | n/a | 1.40620 | 0.03657 | 0.05097 |
| OTFM | 0.0 | 2.47084 | 0.06198 | 0.03738 |
| OTFM | 0.5 | 1.96403 | 0.05322 | 0.05463 |
| OTFM | 1.5 | 1.96247 | 0.05383 | 0.05869 |

The primary validation criterion selected sigma 1.5. A separate integration
check for that checkpoint gave Energy 1.94190, 1.94608, 1.94525, and 1.94525
for 8, 16, 32, and 64 RK4 steps, respectively. This small range indicates the
test result is not explained by using too few integration steps.

## OOD results

The test contains 84 held-out cell-line/drug pairs and 364 exact drug/dose/time
conditions. Each method receives the same source and depth sampling panel; the
common scorer uses the same number of real and generated cells per condition.
Lower Energy and MMD indicate closer distributions; higher expression and
Delta Pearson values indicate better correlation under the existing scorer.

| Method | Energy | MMD squared | Mean expression R2 | Mean Pearson | Std RMSE | Delta Pearson |
|---|---:|---:|---:|---:|---:|---:|
| Vehicle | 0.68117 | 0.01733 | 0.92929 | 0.96626 | 0.24223 | -0.00275 |
| Same-dose transfer | 0.63687 | 0.01680 | 0.93067 | 0.96920 | 0.24088 | 0.03170 |
| BacDiff OT DDPM | 0.91375 | 0.02266 | 0.91426 | 0.96461 | 0.26413 | 0.06978 |
| CellFlow OTFM | 1.22785 | 0.02887 | 0.89150 | 0.97375 | 0.49859 | 0.07268 |
| BacDiff latent OTFM | 1.53898 | 0.04178 | 0.79898 | 0.91631 | 0.44703 | 0.04626 |
| Real vs. real reference | 0.32053 | 0.00927 | 0.96820 | 0.98516 | 0.22285 | 0.32050 |

Pair-macro results are similar: OTFM Energy is 1.55870 and MMD squared is
0.04222, versus 0.92308 and 0.02283 for OT DDPM. The pair-cluster bootstrap
resamples the 84 held-out pairs within each cell line (2,000 draws):

| OTFM minus OT DDPM | Difference | 95% pair-bootstrap interval |
|---|---:|---:|
| Energy | +0.63563 | [+0.55335, +0.71137] |
| MMD squared | +0.01939 | [+0.01744, +0.02122] |
| Delta Pearson | -0.02230 | [-0.04665, +0.00267] |

The first two differences indicate worse OOD distribution matching for this
OTFM adaptation. The Delta Pearson difference is uncertain and its interval
includes zero. Response-stratum Delta Pearson is 0.0020/0.0218/0.1357 for
OTFM and 0.0026/0.0338/0.2043 for OT DDPM across weak/medium/strong responses.

## Interpretation and limits

For this controlled run, replacing residual DDPM with latent OTFM did not
improve the held-out distribution predictions. It performed worse than the
current OT DDPM and the upstream CellFlow run under the shared scorer. This
does not show that flow matching is generally inferior: the result is one
split and one training seed, uses a 32D BacDiff VAE, and is an adapter rather
than the full CellFlow model. Pair bootstrap intervals measure variation over
held-out pairs, not training-seed or split uncertainty.

The experiment also does not establish that training must match an individual
vehicle to an individual treated cell. Both methods use OT pseudo-pairs, but
the OTFM path uses each sampled pair to define a velocity target rather than
as a fixed DDPM residual label. Follow-up should test multiple training seeds,
replicate-aware scoring, and a distribution-level objective before drawing a
method-level conclusion.

## Artifacts

Run files are stored outside the repository under
`/home1/lihx/bacdiff/checking_data/outputs/sciplex3_pair_otfm/`. The validation
sigma runs are under `sciplex3_pair_otfm_sigma0/` and
`sciplex3_pair_otfm_sigma05/`. Large datasets, checkpoints, and generated
predictions are excluded from Git.
