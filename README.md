# Amyloid-beta plaque classification

PyTorch Lightning pipeline for classifying amyloid-beta plaque types from microscopy images. Supports **supervised**, **semi-supervised**, and **self-supervised** training, with optional cross-validation and Optuna hyperparameter search.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run commands from the **repository root** so paths like `configs/` and `data/` resolve correctly. The entry point is `src/main.py`; if imports fail, add `src` to `PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Data

Experiments expect a `data/` folder (see `general_config.json` → `data`):

| Path | Role |
|------|------|
| `data/<data_table_file_name>` | CSV index of samples (paths, labels, splits) |
| `data/label_names.csv` | Maps numeric labels to class names |
| `data/labeled_images/` | Downscaled labeled plaque images |
| `data/sampled_unlabeled_images/` | Unlabeled images (semi- and self-supervised) |

To build a sampled table and image folders from raw HDF5/plaque exports, use the preprocessing script:

```bash
python src/data_preprocessing/data_sampler.py --data_folder data
```

Point `data.data_table_file_name`, `labeled_data_folder`, and `unlabeled_data_folder` in `configs/general_config.json` at whatever filenames and folders you create.

## Quick start

```bash
# Supervised: one train/val/test split
python src/main.py --train_mode supervised --run_mode single

# Semi-supervised (Pi-Model): stratified k-fold CV
python src/main.py --train_mode semi_supervised --run_mode cross_validate

# Self-supervised (SimCLR pretrain + finetune): Optuna search
python src/main.py --train_mode self_supervised --run_mode hyperparameter_tuning --n_trials 20
```

Optional: `--config_dir configs` (default) overrides where JSON configs are loaded from.

## Training methods

Choose the paradigm with `--train_mode`. Each mode uses a dedicated runner and config subtree; unused mode sections are removed at load time so invalid keys fail early.

### Supervised (`--train_mode supervised`)

Standard labeled training: ResNet (or SimpleCNN) backbone + linear/MLP classifier on labeled data only. Configure backbone and classifier under `general_config.architecture` and `configs/architectures/`.

### Semi-supervised (`--train_mode semi_supervised`)

Uses **labeled** and **unlabeled** data. Pick the algorithm in `configs/semi_supervised/semi_supervised_config.json`:

| `model_name` | Config file | Description |
|--------------|-------------|-------------|
| `pi_model` | `pi_model_config.json` | Π-model consistency regularization |
| `fixmatch` | `fixmatch_config.json` | FixMatch-style pseudo-labeling |
| `mean_teacher` | `mean_teacher_config.json` | Mean teacher EMA targets |

Shared semi-supervised knobs (e.g. consistency weight, ramp-up) live in `semi_supervised_config.json` → `training`. Method-specific options are in the matching `*_config.json`.

### Self-supervised (`--train_mode self_supervised`)

Two-stage pipeline:

1. **Pretraining** on unlabeled data (SimCLR or VAE).
2. **Supervised finetuning** on labeled data with the pretrained backbone.

Set `pretraining_method` in `configs/self_supervised/self_supervised_config.json` to `simclr` or `vae`. Method-specific hyperparameters are in `simclr_config.json` or `vae_config.json`. Pretraining checkpoint path and epoch count are under `self_supervised_config.json` → `pretraining`.

## Run modes

| `--run_mode` | Behavior |
|--------------|----------|
| `single` | One stratified train/val/test split; saves metrics, confusion matrix, and `checkpoints/best_model.ckpt` under the run folder |
| `cross_validate` | Stratified k-fold on labeled data (`test_size` in config defines fold count); aggregates test predictions across folds |
| `hyperparameter_tuning` | Optuna TPE study (`--n_trials`); each trial runs inner CV and optimizes **mean validation macro F1**; then runs full CV with best params |

Training defaults (all modes that use a validation set):

- Optimizer: **AdamW** (`general_config.training.optimizer`; learning rate and weight decay are tunable in HPO)
- Max epochs: **200** (`num_epochs`), with early stopping on validation loss and validation every **5** epochs (`early_stop_check_val_every_n_epoch`)
- Best checkpoint selected by **`checkpoint_monitor`** (default: `val_f1`)

Set `general_config.system.debug_mode` to `true` to skip writing fold checkpoints during CV (faster debugging).

## How configuration works

### Loading and merging

`Config.load_config(config_dir, train_mode, run_mode)` in `src/models/config.py`:

1. Recursively loads every `.json` under `config_dir/`, nesting files by folder and basename (e.g. `configs/general_config.json` → `config.general_config`, `configs/semi_supervised/pi_model_config.json` → `config.semi_supervised.pi_model_config`).
2. Reads `data/label_names.csv` and attaches `label_to_name` / `name_to_label`.
3. Sets `run_id` from `SLURM_JOB_ID` or a timestamp, and `general_config.system.device` to `cuda`, `mps`, or `cpu`.
4. **Filters by `train_mode`**: drops `supervised`, `semi_supervised`, or `self_supervised` sections that do not apply.
5. **Filters by active method**: for semi/self-supervised, keeps only the selected `model_name` or `pretraining_method` config file (unless HPO lists multiple methods).
6. **Non-HPO runs**: removes all `hyperparameter_tuning` blocks and keeps only the selected feature extractor and classifier entries under `architectures/`.

Access options with attribute syntax: `config.general_config.training.batch_size`.

Each run saves a resolved snapshot to `config.txt` (JSON) in the run directory.

### Config layout

```
configs/
├── general_config.json          # Data paths, training, system, global HPO search space
├── architectures/
│   ├── feature_extractors_config.json   # resnet18, simple_cnn, …
│   └── classifiers_config.json
├── supervised/
│   └── supervised_config.json   # (empty placeholder; extend if needed)
├── semi_supervised/
│   ├── semi_supervised_config.json
│   ├── pi_model_config.json
│   ├── fixmatch_config.json
│   └── mean_teacher_config.json
└── self_supervised/
    ├── self_supervised_config.json
    ├── simclr_config.json
    └── vae_config.json
```

### Hyperparameter tuning blocks

For `--run_mode hyperparameter_tuning`, any nested object named `hyperparameter_tuning` defines **categorical** search spaces: lists become Optuna choices (see `src/utils/hyperparameter_tuning_utils.py`).

Search spaces are merged from:

- `general_config.hyperparameter_tuning` (learning rate, batch size, freeze/unfreeze, …)
- `<mode>_config.hyperparameter_tuning` (e.g. consistency loss for Pi-Model, `pretraining.num_epochs` for SSL)
- `architectures.feature_extractors_config.<name>.hyperparameter_tuning`
- `architectures.classifiers_config.<name>.hyperparameter_tuning`
- Method-specific files (`simclr_config.json`, `vae_config.json`, etc.)

To search multiple semi-supervised or SSL methods in one study, list them in the parent config, e.g. `semi_supervised_config.hyperparameter_tuning.model_name: ["pi_model", "fixmatch"]` or `self_supervised_config.hyperparameter_tuning.pretraining_method: ["simclr", "vae"]`.

Artifacts: `runs/hyperparameter_tuning/<mode>/.../optuna_study.db`, `best_params.json`, and per-trial subfolders.

### Common settings to change

| Goal | Where to edit |
|------|----------------|
| CSV / image paths | `general_config.data` |
| Backbone / classifier | `general_config.architecture` + `architectures/*` |
| Batch size, epochs, early stopping | `general_config.training` |
| Semi-supervised algorithm | `semi_supervised/semi_supervised_config.json` → `model_name` |
| SSL pretraining method | `self_supervised/self_supervised_config.json` → `pretraining_method` |
| Reproducibility | `general_config.system.random_seed`, `seed_everything` |
| TensorBoard | `general_config.system.tensorboard_log` |

## Outputs

Runs are written under `general_config.data.runs_folder` (default: `runs/`), organized by run mode and training type, for example:

```
runs/single/supervised/resnet18/<run_id>/
runs/cross_validate/semi_supervised/pi_model/resnet18/<run_id>/
runs/hyperparameter_tuning/self_supervised/simclr/resnet18/
```

Typical contents: `config.txt`, `full_training_output.log`, training curves, classification reports, `checkpoints/best_model.ckpt`, and (for HPO) Optuna database exports.

## Cluster / background jobs

Example wrappers under `scripts/`:

```bash
bash scripts/run_single_job.sh              # tmux: semi_supervised, single
bash scripts/run_cross_validation_job.sh    # tmux: cross_validate
bash scripts/run_hyperparam_job.sh        # tmux: hyperparameter_tuning, 20 trials
```

`scripts/run_tmux_job.sh` activates a conda env, runs `src/main.py` with passed args, logs to `logs/`, and can email on completion (`scripts/template.env` for SMTP). Adjust `CONDA_ENV` and paths inside the script for your machine.

## Project layout

```
src/
  main.py                 # CLI
  supervised_runner.py
  semi_supervised_runner.py
  self_supervised_runner.py
  models/                 # Config, base runner, Lightning modules, datasets
  data_preprocessing/     # Sampling / downscaling pipeline
  utils/                  # Logging, plots, Optuna helpers
configs/                  # JSON experiment configuration
notebooks/                # Analysis and visualization
scripts/                  # Job launchers
```

## Development

```bash
bash scripts/ruff_reformat_code.sh
```

Install is defined in `requirements.txt` and `pyproject.toml` (package root: `src/`).
