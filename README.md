# BacDiff

BacDiff models bacterial perturbation effects on single-cell RNA expression with a variational encoder, a conditional latent diffusion model, and a Negative Binomial decoder. The repository contains the training, evaluation, data preparation, and counterfactual analysis code. Datasets and trained checkpoints are kept outside the public source tree.

## Project files

- `data.py`: AnnData loading, RNA preprocessing, bacteria normalization, and dataset splits.
- `model.py`: RNA and bacteria encoders, latent diffusion, and count decoder.
- `losses.py`: reconstruction and regularization objectives.
- `train.py`: model training and checkpoint writing.
- `evaluate.py`: held-out test evaluation.
- `infer_counterfactual.py`: condition and dose-scan inference.
- `visualize_results.py`: plots from counterfactual result tables.
- `merge_all.py`: merge paired GEX and 16S AnnData files.
- `data/prepare_sciplex.py`: convert Sci-Plex data to BacDiff's AnnData format.

## Setup

Use Python 3.11 or newer in a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `.h5ad` datasets are not included. Place a prepared merged dataset at the path configured in the selected YAML file, or edit `data.merged_h5ad` to point to its location. A merged AnnData object must contain:

- `obsm["bacteria_raw"]`: cells by bacterial features.
- `uns["bacteria_names"]`: feature names in the same order.
- `obs["bacteria_label"]`: condition labels, with controls labeled `none`.

The RNA matrix in `X` must match the selected `preprocess.log1p_input` setting. With `true`, it is treated as counts and normalized/log-transformed for model input. Reconstruction targets are taken from `X`.

## Run

From the repository root, choose a config and run training, evaluation, and counterfactual inference:

```bash
python train.py --config config.yaml
python evaluate.py --config config.yaml
python infer_counterfactual.py --config config.yaml --sample-n 128
python visualize_results.py --config config.yaml
```

The default config expects `data/OSCC_11_merged.h5ad` and writes runs beneath `outputs/`. Dataset-specific templates are provided for OSCC15, M5, M20, and Sci-Plex. Edit the paths and model settings for your local environment before running.

To merge GEX and 16S input files, put the expected input files under `data/` and run:

```bash
python merge_all.py
```

The merge script aligns cells by `obs_names`, filters rare bacterial features and labels, and writes merged `.h5ad` files under `data/`. The Sci-Plex converter takes explicit paths:

```bash
python data/prepare_sciplex.py --input /path/to/sciplex.h5ad --output data/sciplex_merged.h5ad
```

## Outputs

Training writes a checkpoint, the effective config, a CSV training log, and a loss plot under `outputs/<run_name>/`. Evaluation writes metrics and plots under `eval/`; counterfactual inference writes tables, arrays, and plots under `counterfactual/`.

Datasets, checkpoints, and generated outputs are excluded by `.gitignore`. Obtain and manage source datasets according to their original licenses and access terms.
