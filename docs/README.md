# Multi-Objective Optimization for Image Classification

**Pareto Front Analysis** — MOML Assignment  
Prof. Aswin Kannan | Niranjan Gopal, Divyam Sareen  
Due: April 28, 2026

---

## What This Project Does

We design lightweight CNN image classifiers and use **multi-objective optimization (MOO)** to find the best trade-offs between three conflicting goals:

| Objective | Direction | What It Measures |
|-----------|-----------|-----------------|
| **Accuracy** | Maximize | Top-1 test accuracy (%) |
| **Inference Speed** | Minimize | Avg. milliseconds per image (CPU) |
| **Model Size** | Minimize | Number of trainable parameters |

You can't have all three — a more accurate model is usually bigger and slower. MOO finds the **Pareto front**: the set of solutions where you can't improve one objective without hurting another.

We compare two MOO frameworks:
- **BoTorch** (Bayesian, qNEHVI) — sample-efficient, ~50–80 trials
- **Optuna** (Evolutionary, NSGA-II) — scalable, ~100–150 trials

---

## Datasets

| Dataset | Images | Classes | Resolution | Type |
|---------|--------|---------|------------|------|
| **CIFAR-10** | 60,000 | 10 (airplane, car, bird, …) | 32×32 | RGB color |
| **Fashion-MNIST** | 70,000 | 10 (T-shirt, sneaker, bag, …) | 28×28 | Grayscale |

Both are loaded **locally** from `data/` — no internet download needed.

---

## Search Space

The optimizer searches over these variables to find the best configurations:

| Variable | Range | Description |
|----------|-------|-------------|
| `arch_type` | plain, residual, depthwise_separable | CNN architecture family |
| `num_conv_layers` | 1–4 | Depth of the network |
| `num_channels` | 8–128 (powers of 2) | Width of convolutional layers |
| `num_fc_units` | 32–256 | Size of fully-connected layer |
| `learning_rate` | 1e-5 to 1e-2 (log scale) | Training step size |
| `batch_size` | 16, 32, 64 | Training batch size |
| `num_epochs` | 5–15 | Training duration |
| `dropout_rate` | 0.0–0.5 | Regularization strength |
| `optimizer_type` | SGD, Adam | Optimization algorithm |
| `input_resolution` | 16, 32 | Image size (downsampled) |

See [`ARCHITECTURES.md`](ARCHITECTURES.md) for a detailed explanation of the three CNN families.

---

## Project Structure

```
MOML/
├── data/                        # Datasets (local, no download needed)
│   ├── cifar-10-batches-py/     #   CIFAR-10 pickle batches
│   └── FashionMNIST/raw/        #   Fashion-MNIST IDX files
│
├── data_loader.py               # Dataset loading, transforms, CUDA-aware subsets
├── models.py                    # 3 CNN architectures (Plain, Residual, DepthSep)
├── train_eval.py                # Training loop + 3-objective evaluation
├── moo_optuna.py                # [TODO] Optuna NSGA-II optimization loop
├── moo_botorch.py               # [TODO] BoTorch qNEHVI optimization loop
├── pareto_analysis.py           # [TODO] Pareto front, hypervolume, spacing metrics
├── visualize.py                 # [TODO] 2D/3D scatter, parallel coordinates plots
│
├── results/                     # [Auto-created] Trial logs and Pareto fronts
├── PROJECT_DESCRIPTION.md       # Full assignment specification
├── ARCHITECTURES.md             # CNN architecture explanation (beginner-friendly)
├── README.md                    # This file
└── requirements.txt             # [TODO] Python dependencies
```

---

## Setup

### Prerequisites
- Python 3.12
- NVIDIA GPU (optional but recommended; tested on GTX 1650)

### Installation

```bash
# Create virtual environment
py -3.12 -m venv .venv

# Activate (PowerShell)
.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.4 (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install botorch optuna matplotlib numpy
```

### Verify CUDA

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True / NVIDIA GeForce GTX 1650
```

---

## CUDA-Aware Defaults

The code auto-detects your hardware and adjusts:

| Setting | GPU (CUDA) | CPU only |
|---------|-----------|----------|
| Training device | GPU | CPU |
| Train subset size | 20,000 samples | 10,000 samples |
| Est. time per trial | ~30s – 1 min | ~1 – 2 min |

Inference time is **always measured on CPU** (batch_size=1) for fair comparison.

---

## Quick Test

```bash
# Test data loading
python data_loader.py

# Test all 3 architectures (forward pass only)
python models.py

# Run full train+evaluate sanity check (trains 6 small models, ~5 min on GPU)
python train_eval.py
```

---

## Current Progress

- [x] Project setup and data loading (CIFAR-10 + Fashion-MNIST)
- [x] 3 CNN architectures implemented and tested
- [x] Training loop with 3-objective evaluation
- [x] CUDA/GPU support verified (GTX 1650)
- [ ] Optuna NSGA-II optimization loop
- [ ] BoTorch qNEHVI optimization loop
- [ ] Pareto front analysis and metrics
- [ ] Visualization plots
- [ ] Report
