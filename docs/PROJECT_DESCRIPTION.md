# Multi-Objective Optimization for Image Classification
### Pareto Front Analysis | MOML Assignment

**Course:** Multi-Objective Machine Learning  
**Instructor:** Prof. Aswin Kannan  
**Team:** Niranjan Gopal, Divyam Sareen  
**Due:** April 28, 2026 | **Viva:** April 2, 2026

---

## Problem Statement

Design a lightweight CNN image classifier and perform multi-objective optimization (MOO) to simultaneously balance **classification accuracy**, **inference speed**, and **model compactness** — three inherently conflicting objectives. We apply two MOO frameworks (BoTorch and Optuna) and compare their Pareto fronts on two datasets.

---

## Datasets

| Dataset | Size | Classes | Input | Split |
|---------|------|---------|-------|-------|
| **CIFAR-10** | 60k images | 10 (airplane, car, bird, ...) | 32×32 RGB | 50k train / 10k test |
| **Fashion-MNIST** | 70k images | 10 (T-shirt, trouser, bag, ...) | 28×28 grayscale | 60k train / 10k test |

Both datasets are class-balanced and directly available via `torchvision.datasets`.

---

## Objectives (3, optimized simultaneously)

| # | Objective | Direction | Metric |
|---|-----------|-----------|--------|
| O1 | Classification Accuracy | **Maximize** | Top-1 accuracy on test set |
| O2 | Inference Time | **Minimize** | Avg. ms per sample (CPU, batch=1) |
| O3 | Model Size | **Minimize** | Total trainable parameters |

All three conflict: higher accuracy typically demands more parameters (larger O3) and slower inference (larger O2).

---

## Decision Variables (Search Space)

| Variable | Type | Range |
|----------|------|-------|
| `num_conv_layers` | Integer | [1, 4] |
| `num_channels` | Integer | [8, 128] (powers of 2) |
| `num_fc_units` | Integer | [32, 256] |
| `learning_rate` | Float (log) | [1e-5, 1e-2] |
| `batch_size` | Categorical | {16, 32, 64} |
| `num_epochs` | Integer | [5, 15] |
| `dropout_rate` | Float | [0.0, 0.5] |
| `optimizer_type` | Categorical | {SGD, Adam} |
| `input_resolution` | Categorical | {16, 32} pixels (downsampled) |

---

## Optimization Frameworks

### Framework 1: BoTorch (Bayesian MOO)
- **Algorithm:** `qNEHVI` (q-Noisy Expected Hypervolume Improvement)
- **Surrogate:** Gaussian Process over objective space
- **Strengths:** Sample-efficient, principled uncertainty quantification, strong for expensive black-box evaluations
- **Budget:** ~50–80 total evaluations (BoTorch is data-efficient)

### Framework 2: Optuna (Evolutionary MOO)
- **Algorithm:** `NSGAIISampler` (Non-dominated Sorting Genetic Algorithm II)
- **Strengths:** Scalable, easy to parallelize, widely used baseline for NAS-style problems
- **Budget:** ~100–150 trials (evolutionary methods need more evaluations)

### Comparison Goal
Run both frameworks on the same search space and compare:
- Pareto front shape and diversity
- Hypervolume indicator
- Wall-clock time to convergence
- Quality of non-dominated solutions found

---

## MOO Loop Design

```
for each trial (BoTorch suggestion or Optuna trial):
    1. Sample architecture config from search space
    2. Build CNN with sampled hyperparameters
    3. Train on dataset (subset for speed, e.g., 10k samples)
    4. Evaluate:
         - O1: accuracy on test set
         - O2: mean inference time over 500 test samples (CPU)
         - O3: count trainable parameters via sum(p.numel())
    5. Report objectives to optimizer
    6. Update surrogate / evolve population

Total wall-clock budget: < 2 hours per framework
```

---

## Deliverables

### Code (public GitHub repo)
- `train_eval.py` — CNN definition, training loop, objective evaluation
- `moo_botorch.py` — BoTorch qNEHVI MOO loop
- `moo_optuna.py` — Optuna NSGA-II MOO loop
- `pareto_analysis.py` — Pareto front extraction, hypervolume, spacing metric
- `analyze_study.py` — Pareto front extraction, metrics, plots
- `requirements.txt`, `README.md`

### Report (max 6 pages)
1. **Quantitative Analysis**
   - 2D pairwise Pareto scatter plots (O1 vs O2, O1 vs O3, O2 vs O3)
   - 3D Pareto-front scatter (all-minimize bowl orientation)
   - Hypervolume indicator (with defined reference point)
   - Spacing metric
   - Generational distance (BoTorch vs Optuna comparison)

2. **Qualitative Analysis**
   - Trade-off discussion (e.g., "gaining 5% accuracy triples model size")
   - One non-obvious trade-off observed
   - Strengths/weaknesses of BoTorch vs Optuna on this problem
   - Practically useful solutions from the Pareto front

3. **Pareto Points Table** (≥4 points per framework)

   | Solution | Accuracy (%) | Inference (ms) | Params (k) |
   |----------|-------------|----------------|------------|
   | A (fast) | ... | ... | ... |
   | B (small) | ... | ... | ... |
   | C (accurate) | ... | ... | ... |
   | D (balanced) | ... | ... | ... |

4. **Appendix:** One Pareto-optimal solution explained in full (architecture, hyperparameters, objective values)

---

## Grading Rubric

| Component | Marks | Weight |
|-----------|-------|--------|
| Pareto front quality (diversity, convergence, non-dominated solutions) | 10 | 25% |
| Report (qualitative & quantitative analysis, trade-offs, tabulation) | 10 | 25% |
| Code quality (reproducibility, modularity, implementation) | 10 | 25% |
| Viva voce (understanding of MOO concepts, trade-offs, results) | 10 | 25% |
| **Total** | **40** | **100%** |

---

## Timeline

| Date | Milestone |
|------|-----------|
| Apr 23 | Project setup, baseline CNN, data loading |
| Apr 24 | Optuna NSGA-II loop running on both datasets |
| Apr 25 | BoTorch qNEHVI loop running; Pareto extraction |
| Apr 26 | Visualizations, hypervolume/spacing metrics |
| Apr 27 | Report writing, code cleanup, GitHub push |
| **Apr 28** | **Submission deadline** |

---

## Key Implementation Notes

- Train on a **fixed random subset** (e.g., 10k samples from CIFAR-10) per trial to keep each evaluation under ~2 min
- Measure inference time on **CPU** with `batch_size=1` using `torch.no_grad()` + `time.perf_counter()`
- Normalize objectives before passing to BoTorch (GP requires roughly unit-scale inputs)
- Fix random seed per trial for reproducibility
- For Optuna: use `study.optimize(..., n_jobs=1)` to avoid GPU contention during timing measurements
- Reference point for hypervolume: `[0.0, 1000ms, 10M params]` (worst-case bounds)
