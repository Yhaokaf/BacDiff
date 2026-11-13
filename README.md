# BacDiff

BacDiff models bacterial perturbation effects on single-cell RNA expression with a variational encoder, conditional latent diffusion, and a Negative Binomial decoder. It includes data preparation, model training, held-out evaluation, and counterfactual dose scans.

## Model overview

The RNA encoder produces base and total cell embeddings. A bacteria-conditioned diffusion model predicts the latent shift, which the decoder maps back to gene expression. The lower panel illustrates bacteria and dose counterfactuals.

![BacDiff model architecture and counterfactual workflow](assets/bacdiff_model.png)

## Repository layout

```text
bacdiff/
  data/       AnnData dataset, merging, and conversion tools
  models/     Neural networks and loss functions
  utils/      Configuration, random seed, and device helpers
  cli/        Training, evaluation, inference, and plotting commands
configs/      Dataset-specific YAML configurations
notebooks/    Exploratory and data inspection notebooks
```

## Setup

Use Python 3.11 or newer in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Datasets and trained checkpoints are not included. Put the merged AnnData file at the path configured by `data.merged_h5ad`, or update that setting. The merged object must contain `obsm["bacteria_raw"]`, `uns["bacteria_names"]`, and `obs["bacteria_label"]`; controls use the label `none`. For configs with `preprocess.log1p_input: true`, `X` is treated as counts and normalized/log-transformed for model input. Reconstruction targets are read from `X`.

## Main workflow

Run commands from the repository root. The default config expects `data/OSCC_11_merged.h5ad`:

```bash
python -m bacdiff.cli.train --config configs/config.yaml
python -m bacdiff.cli.evaluate --config configs/config.yaml
python -m bacdiff.cli.infer_counterfactual --config configs/config.yaml --sample-n 128
python -m bacdiff.cli.visualize_results --config configs/config.yaml
```

For CPA-style Sci-Plex metrics, also provide the original annotated file:

```bash
python -m bacdiff.cli.evaluate_cpa_style --config configs/config_sciplex.yaml --original_h5ad /path/to/original_sciplex.h5ad
```

Other configs are provided for OSCC15, M5, M20, and Sci-Plex. Edit their data paths and training settings for your environment.

## Data preparation

Put paired GEX and 16S files in `data/`, then merge the OSCC15, M5, and M20 datasets:

```bash
python -m bacdiff.data.merge_all
```

The merger aligns cells by `obs_names`, rejects mismatched cell ID sets, filters rare bacterial features and labels, then writes merged files into `data/`. To convert a prepared Sci-Plex file:

```bash
python -m bacdiff.data.prepare_sciplex --input /path/to/sciplex.h5ad --output data/sciplex_merged.h5ad
```

## Outputs

Training writes checkpoints, the effective config, a CSV log, and a loss plot under `outputs/<run_name>/`. Evaluation writes metrics and plots under `eval/`; counterfactual inference writes result tables, arrays, and plots under `counterfactual/`.

Datasets, checkpoints, and generated outputs are excluded by `.gitignore`. Obtain source datasets according to their original licenses and access terms.
