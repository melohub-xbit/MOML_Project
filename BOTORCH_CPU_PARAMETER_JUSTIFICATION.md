# BoTorch CPU Hyperparameter Settings: Why These Values

This document explains the exact choices used in `botorch_cpu_hpo.ipynb` and why they are practical on a no-GPU machine.

## 1) Objective Design

We optimize 3 objectives:

- Accuracy: maximize (`accuracy`)
- Inference latency: minimize (`inference_ms`)
- Model size: minimize (`param_count`)

BoTorch acquisition functions are formulated as maximization problems, so we transform:

- `y1 = accuracy`
- `y2 = -inference_ms`
- `y3 = -(param_count / 1_000_000)`

Why divide params by 1,000,000:

- Keeps magnitudes closer to other objectives.
- Improves numerical conditioning for GP fitting.

## 2) CPU Runtime Controls

### `TRAIN_SUBSET_SIZE = 3000`

Why:

- Strongly reduces per-trial training time on CPU.
- Still large enough to produce meaningful differences between configs.

How to tune:

- Faster but noisier: 1000-2000
- Slower but more stable: 5000-10000

### Epoch choices: `[3, 5, 8]`

Why:

- CPU-friendly range.
- Gives BO enough variation in train time vs. performance.

How to tune:

- If too slow: `[2, 3, 5]`
- If stable and you want better final quality: include `10` or `12`

## 3) Search Space Choices

### Architecture: `plain`, `residual`, `depthwise_separable`

Why:

- Each architecture occupies a different trade-off region.
- Improves Pareto diversity.

### `num_conv_layers = [1, 2, 3, 4]`

Why:

- Covers shallow to deeper CNNs.
- 4 layers is enough to see non-linear complexity effects without being too slow.

### `num_channels = [8, 16, 32, 64]`

Why:

- Power-of-2 widths are standard and efficient.
- Upper bound 64 is safer for CPU runtime than 128.

### `num_fc_units = [32, 64, 128, 256]`

Why:

- Captures small to moderately expressive classifier heads.

### `learning_rate = [1e-4, 3e-4, 1e-3, 3e-3]`

Why:

- Log-spaced coverage of common useful rates.
- Includes conservative and aggressive settings.

### `batch_size = [16, 32, 64]`

Why:

- Standard range for CPU memory and throughput trade-off.

### `dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4]`

Why:

- Captures no regularization to fairly strong regularization.

### `optimizer_type = [Adam, SGD]`

Why:

- Adam gives fast convergence early.
- SGD is a classic baseline and may generalize better in some settings.

### `input_resolution = [16, 32]`

Why:

- Directly models speed-accuracy trade-off through image size.

## 4) BoTorch Strategy Parameters

### `N_INIT = 8` (Sobol initial points)

Why:

- Enough diversity to fit a first GP on a 10D space.
- Small enough for CPU budget.

### `N_BO_ITERS = 12`

Why:

- Total 20 trials (`8 + 12`) is realistic on CPU.
- Better than pure random search at similar budget.

### `q = 1`

Why:

- Sequential candidate generation is simpler and stable on a single CPU machine.

### `num_restarts = 8`, `raw_samples = 256`

Why:

- Good balance between acquisition optimization quality and overhead.

How to tune:

- Faster: `num_restarts=4`, `raw_samples=128`
- Better acquisition optimization: `num_restarts=12`, `raw_samples=512`

### QMC samples in qNEHVI: `128`

Why:

- Moderate Monte Carlo accuracy without large overhead.

### Reference point: `CPU_REF_POINT = [0.0, -100.0, -20.0]`

Meaning in original units:

- Accuracy worst-case: `0.0`
- Inference worst-case: `100 ms`
- Params worst-case: `20M`

Why:

- Must be a clearly worse point than expected outcomes.
- Conservative enough to avoid invalid hypervolume regions.

## 5) Mixed Variable Handling

BoTorch in this notebook uses a continuous 10D unit cube, then maps each dimension to a discrete/categorical option.

Why this approach:

- Practical and robust for mixed search spaces.
- Avoids heavy custom mixed-variable modeling complexity.

Trade-off:

- Multiple unit-cube values can map to the same discrete config.
- For class project scale, this is usually acceptable.

## 6) Suggested Scaling Path

Start here (current notebook):

- `TRAIN_SUBSET_SIZE=3000`, `N_INIT=8`, `N_BO_ITERS=12`

If stable and you want stronger results:

1. Increase `TRAIN_SUBSET_SIZE` to 5000.
2. Increase `N_BO_ITERS` to 20.
3. Add one higher epoch option (for example `10`).

If runtime is too high:

1. Reduce `TRAIN_SUBSET_SIZE` to 2000.
2. Reduce `N_BO_ITERS` to 8.
3. Restrict architecture choices temporarily (for example remove `residual` for quick smoke tests).

## 7) Reproducibility

We use fixed seeds (`SEED = 42`) for:

- Python random
- NumPy
- PyTorch

This reduces variation across runs and helps compare tuning changes fairly.
