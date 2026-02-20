# spatio-temporal-reading

A Python project for ***spatio-temporal reading analysis*** — preprocessing, exploring and plotting spatial-temporal data.

This repo includes:
- Preprocessing utilities for aligning fixations to text (character/token indices)
- Optional computation of **GPT-2 hidden states** for tokens
- Training + evaluation for next-fixation / saccade-duration prediction
- Baselines + plotting + qualitative visualizations (GIFs)

## Command Reference

All commands are launched from repo root:

```bash
python main.py <FLAGS> --data DATA_DIR [--output OUTPUT_DIR]
```

### 1) Preprocessing

Add fixation-level indices (`dx`, `dy`, `saccade`, `char_idx`) and NaN indicators:

```bash
python main.py --include-indices --data DATA_DIR
```

Writes:
- `hp_augmented_meco_100_1000_1_10_sacc_idx.csv`
- `hp_augmented_meco_100_1000_1_10_model.csv`
- `variances.json`

Add token indices + GPT-2 hidden states (for LM-augmented training):

```bash
python main.py --include-tokens --data DATA_DIR
```

Writes:
- `hp_eng_texts_100_1000_1_10_tokens.csv`
- `hp_augmented_meco_100_1000_1_10_model_tokens.csv`
- `hidden_states.npy`

### 2) Exploratory Plots

```bash
python main.py --make-plots --data DATA_DIR --output OUTPUT_DIR
```

Writes multiple analysis plots in `OUTPUT_DIR` (durations, regressions, forward/backward jumps, etc.).

### 3) Transformer Training

```bash
python main.py --train --data DATA_DIR --output OUTPUT_DIR
```

Writes:
- `best_model.pt` in `OUTPUT_DIR`

Useful options:
- `--filtering {filtered|raw}`: default `filtered`
- `--n-layers INT`: transformer blocks (default `3`)
- `--d-model INT`: hidden size (default `30`)
- `--heads INT`: number of heads argument (default `5`)
- `--n-components INT`: mixture components (default `20`)
- `--epochs INT`: epochs (default `50`)
- `--cov`: train covariance-aware variant
- `--augment`: use LM-augmented dataset (`*_tokens.csv` + `hidden_states.npy`)

### 4) Transformer Test Evaluation

```bash
python main.py --test --data DATA_DIR --checkpoint-path CHECKPOINT_PATH
```

Writes:
- `negloglikelihoods.npy` next to the checkpoint directory

Supports the same model switches as training (`--filtering`, `--cov`, `--augment`).

### 5) Baseline Grid Search Training

```bash
python main.py --train-baseline --data DATA_DIR
```

Runs baseline sweeps defined in `spatio_temporal_reading/launchers/baseline/grid_search_baseline.py` and writes results under model-specific folders inside `DATA_DIR`.

### 6) Baseline Test Evaluation

```bash
python main.py --test-baseline --data DATA_DIR --models-path MODELS_ROOT
```

For each baseline family in `MODELS_ROOT`, selects the best run by validation loss and writes:
- `negloglikelihoods.npy` in each model family directory.

### 7) Qualitative GIF Visualization

```bash
python main.py --visualize-model --data DATA_DIR --output OUTPUT_DIR --checkpoint-path CHECKPOINT_PATH
```

Intended outputs:
- `train_pos.gif`
- `train_sacc.gif`
- `val_pos.gif`
- `val_sacc.gif`

Optional:
- `--train_index INT` (default `0`)
- `--val_index INT` (default `0`)
