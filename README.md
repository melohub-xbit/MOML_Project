# MoML — Multi-Objective Optimization for Image Classification

Pareto-front analysis over 3 conflicting objectives (accuracy / inference time / model size) on Fashion-MNIST, comparing **pymoo NSGA-II** (evolutionary) and **BoTorch qNEHVI** (Bayesian) head-to-head on the same search space and same trial budget.

Project layout:
- [`src/`](src/) — Python modules (loaders, models, training, two MOO drivers, framework-agnostic analyzer)
- [`notebooks/`](notebooks/) — `pymoo_optimization.ipynb` and `botorch_optimization.ipynb`, designed to run concurrently
- [`docs/`](docs/) — full project description, architecture explainer, original PDF spec
- [`data/`](data/) — committed CIFAR-10 + Fashion-MNIST raw files (no internet download needed)
- [`results/`](results/) — per-framework study outputs (`pymoo/`, `botorch/`)

See [`docs/README.md`](docs/README.md) for the project overview and [`CLAUDE.md`](CLAUDE.md) for working notes.

## Quick start

```bash
# Activate venv (Python 3.12, see docs/README.md for the install steps)
.venv\Scripts\Activate.ps1

# Run pymoo NSGA-II  (80 trials, ≤1.5h)
python src/moo_pymoo.py --dataset fashion_mnist --pop-size 20 --n-gen 4

# Run BoTorch qNEHVI (80 trials, ≤1.5h, can run concurrently with the pymoo notebook)
python src/moo_botorch.py --dataset fashion_mnist --total-trials 80 --n-init 16

# Analyze either study + cross-compare
python src/analyze_study.py \
    --study-dir results/pymoo/fashion_mnist/<study> \
    --compare   results/botorch/fashion_mnist/<study>/pareto_front.csv
```
