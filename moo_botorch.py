"""
moo_botorch.py

BoTorch-based multi-objective hyperparameter optimization for image classification.
Designed for remote GPU execution with full datasets and a 100-trial default budget.

Objectives (maximize in BoTorch space):
    y1 = accuracy
    y2 = -inference_ms
    y3 = -(param_count / 1_000_000)

Notes:
- Accuracy is evaluated on the full test split.
- Inference time follows project spec (CPU, batch_size=1), implemented in train_eval.py.
- Set train_subset_size=None for full training datasets.

Example:
    python moo_botorch.py --dataset cifar10 --total-trials 100 --n-init 20 --train-subset-size full --num-workers 8 --download-if-missing
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torchvision import datasets, transforms

from data_loader import DATA_ROOT, DEVICE, get_dataset_info
from train_eval import train_and_evaluate

# Temporary upstream warning from torchvision + NumPy 2.4+ when reading CIFAR pickle metadata.
if hasattr(np, "VisibleDeprecationWarning"):
    warnings.filterwarnings(
        "ignore",
        message=r"dtype\(\): align should be passed as Python or NumPy boolean",
        category=np.VisibleDeprecationWarning,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dataset_available(dataset_name: str, download_if_missing: bool) -> None:
    """Verify local dataset exists; optionally download once if missing."""
    tf = transforms.ToTensor()
    key = dataset_name.lower().replace("-", "_")

    def _load(download: bool) -> None:
        if key == "cifar10":
            datasets.CIFAR10(root=DATA_ROOT, train=True, download=download, transform=tf)
            datasets.CIFAR10(root=DATA_ROOT, train=False, download=download, transform=tf)
        elif key == "fashion_mnist":
            datasets.FashionMNIST(root=DATA_ROOT, train=True, download=download, transform=tf)
            datasets.FashionMNIST(root=DATA_ROOT, train=False, download=download, transform=tf)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    try:
        _load(download=False)
    except Exception:
        if not download_if_missing:
            raise RuntimeError(
                f"Dataset '{dataset_name}' was not found under {DATA_ROOT}. "
                "Run with --download-if-missing or place dataset files locally."
            )
        print(f"[INFO] Downloading missing dataset: {dataset_name} into {DATA_ROOT}")
        _load(download=True)


class SearchSpace:
    """Discrete/categorical search space with unit-cube encoding/decoding."""

    def __init__(self) -> None:
        self.arch_type = ["plain", "residual", "depthwise_separable"]
        self.num_conv_layers = [1, 2, 3, 4]
        self.num_channels = [8, 16, 32, 64, 128]
        self.num_fc_units = [32, 64, 128, 256]
        self.learning_rate = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        self.batch_size = [16, 32, 64]
        self.num_epochs = [5, 8, 10, 12, 15]
        self.dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.optimizer_type = ["SGD", "Adam"]
        self.input_resolution = [16, 32]

        self.dim = 10

    @staticmethod
    def _pick_from_unit(u: float, choices: list[Any]) -> Any:
        idx = int(np.clip(np.floor(u * len(choices)), 0, len(choices) - 1))
        return choices[idx]

    def decode(self, x_unit: np.ndarray) -> dict[str, Any]:
        x = np.asarray(x_unit, dtype=float).reshape(-1)
        return {
            "arch_type": self._pick_from_unit(x[0], self.arch_type),
            "num_conv_layers": self._pick_from_unit(x[1], self.num_conv_layers),
            "num_channels": self._pick_from_unit(x[2], self.num_channels),
            "num_fc_units": self._pick_from_unit(x[3], self.num_fc_units),
            "learning_rate": self._pick_from_unit(x[4], self.learning_rate),
            "batch_size": self._pick_from_unit(x[5], self.batch_size),
            "num_epochs": self._pick_from_unit(x[6], self.num_epochs),
            "dropout_rate": self._pick_from_unit(x[7], self.dropout_rate),
            "optimizer_type": self._pick_from_unit(x[8], self.optimizer_type),
            "input_resolution": self._pick_from_unit(x[9], self.input_resolution),
        }


def config_to_key(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True)


def objective_transform(raw: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(
        [
            float(raw["accuracy"]),
            -float(raw["inference_ms"]),
            -(float(raw["param_count"]) / 1_000_000.0),
        ],
        dtype=torch.double,
    )


def compute_ref_point(y: torch.Tensor) -> list[float]:
    """
    Build a conservative dynamic reference point in transformed objective space.
    A valid reference point should be slightly worse than observed points.
    """
    margin = torch.tensor([0.02, 5.0, 0.5], dtype=torch.double)
    ref = y.min(dim=0).values - margin
    return ref.tolist()


def pareto_mask(values: np.ndarray, maximize: list[bool]) -> np.ndarray:
    n = values.shape[0]
    keep = np.ones(n, dtype=bool)
    signed = values.copy()

    for j, is_max in enumerate(maximize):
        if not is_max:
            signed[:, j] *= -1.0

    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = np.all(signed >= signed[i], axis=1) & np.any(signed > signed[i], axis=1)
        if np.any(dominates_i):
            keep[i] = False
    return keep


def save_progress(
    results_rows: list[dict[str, Any]],
    x: torch.Tensor,
    y: torch.Tensor,
    out_dir: str,
    dataset_name: str,
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    all_csv = os.path.join(out_dir, f"botorch_{dataset_name}_trials.csv")
    state_pt = os.path.join(out_dir, f"botorch_{dataset_name}_state.pt")

    df = pd.DataFrame(results_rows)
    df.to_csv(all_csv, index=False)

    torch.save(
        {
            "X": x.detach().cpu(),
            "Y": y.detach().cpu(),
            "num_trials": len(results_rows),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        state_pt,
    )
    return all_csv, state_pt


def plot_results(df: pd.DataFrame, pareto_df: pd.DataFrame, out_dir: str, dataset_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"botorch_{dataset_name}_tradeoffs.png")

    colors = {
        "plain": "#e74c3c",
        "residual": "#3498db",
        "depthwise_separable": "#2ecc71",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for _, r in df.iterrows():
        p = float(r["param_count"]) / 1000.0
        a = float(r["accuracy"]) * 100.0
        t = float(r["inference_ms"])
        c = colors.get(r["arch_type"], "gray")

        axes[0].scatter(p, a, c=c, s=65, alpha=0.85, edgecolors="k", linewidths=0.3)
        axes[1].scatter(t, a, c=c, s=65, alpha=0.85, edgecolors="k", linewidths=0.3)
        axes[2].scatter(p, t, c=c, s=65, alpha=0.85, edgecolors="k", linewidths=0.3)

    for _, r in pareto_df.iterrows():
        p = float(r["param_count"]) / 1000.0
        a = float(r["accuracy"]) * 100.0
        t = float(r["inference_ms"])

        axes[0].scatter(p, a, facecolors="none", edgecolors="black", s=180, linewidths=1.4)
        axes[1].scatter(t, a, facecolors="none", edgecolors="black", s=180, linewidths=1.4)
        axes[2].scatter(p, t, facecolors="none", edgecolors="black", s=180, linewidths=1.4)

    axes[0].set_title("Accuracy vs Size")
    axes[0].set_xlabel("Params (K)")
    axes[0].set_ylabel("Accuracy (%)")

    axes[1].set_title("Accuracy vs Inference")
    axes[1].set_xlabel("Inference (ms)")
    axes[1].set_ylabel("Accuracy (%)")

    axes[2].set_title("Inference vs Size")
    axes[2].set_xlabel("Params (K)")
    axes[2].set_ylabel("Inference (ms)")

    fig.suptitle(f"BoTorch qNEHVI ({dataset_name}) - Pareto outlined", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def run_botorch(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dataset_available(args.dataset, download_if_missing=args.download_if_missing)

    space = SearchSpace()
    if args.n_init >= args.total_trials:
        raise ValueError("--n-init must be smaller than --total-trials")

    n_bo_iters = args.total_trials - args.n_init

    ds_info = get_dataset_info(args.dataset)
    print("=" * 90)
    print("BoTorch qNEHVI Run Configuration")
    print("=" * 90)
    print(f"Device               : {DEVICE}")
    print(f"Dataset              : {args.dataset}")
    print(f"Classes              : {ds_info['num_classes']}")
    print(f"Input channels       : {ds_info['input_channels']}")
    print(f"Trials               : {args.total_trials} (init={args.n_init}, bo={n_bo_iters})")
    print(f"Train subset size    : {args.train_subset_size}")
    print(f"DataLoader workers   : {args.num_workers}")
    print(f"Results directory    : {args.out_dir}")
    print("=" * 90)

    bounds = torch.stack(
        [torch.zeros(space.dim, dtype=torch.double), torch.ones(space.dim, dtype=torch.double)]
    )

    sobol = torch.quasirandom.SobolEngine(dimension=space.dim, scramble=True, seed=args.seed)
    x = sobol.draw(args.n_init).to(dtype=torch.double)

    y_list: list[torch.Tensor] = []
    results_rows: list[dict[str, Any]] = []
    seen_configs: set[str] = set()

    print("\n[Phase] Sobol initialization")
    for i in range(args.n_init):
        cfg = space.decode(x[i].detach().cpu().numpy())
        key = config_to_key(cfg)

        # If Sobol maps to a duplicate discrete config, resample until unique.
        while key in seen_configs:
            x[i] = sobol.draw(1).to(dtype=torch.double)[0]
            cfg = space.decode(x[i].detach().cpu().numpy())
            key = config_to_key(cfg)

        seen_configs.add(key)

        t0 = time.time()
        raw = train_and_evaluate(
            config=cfg,
            dataset_name=args.dataset,
            seed=args.seed + i,
            train_subset_size=args.train_subset_size,
            num_workers=args.num_workers,
        )
        elapsed = time.time() - t0

        y_i = objective_transform(raw)
        y_list.append(y_i)

        row = {
            "trial": i,
            "phase": "init",
            "dataset": args.dataset,
            **cfg,
            **raw,
            "objective_accuracy": float(y_i[0].item()),
            "objective_neg_infer_ms": float(y_i[1].item()),
            "objective_neg_params_m": float(y_i[2].item()),
            "wall_time_s": elapsed,
        }
        results_rows.append(row)

        print(
            f"init {i:03d} | acc={raw['accuracy']:.2%} | infer={raw['inference_ms']:.3f} ms | "
            f"params={raw['param_count']:,} | {elapsed:.1f}s"
        )

    y = torch.stack(y_list, dim=0)
    all_csv, state_pt = save_progress(results_rows, x, y, args.out_dir, args.dataset)
    print(f"[Saved] {all_csv}")
    print(f"[Saved] {state_pt}")

    print("\n[Phase] Bayesian optimization")
    for bo_iter in range(n_bo_iters):
        trial_id = args.n_init + bo_iter
        print(f"\nBO iter {bo_iter + 1}/{n_bo_iters} (trial {trial_id})")

        model = SingleTaskGP(x, y, outcome_transform=Standardize(m=y.shape[-1]))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        ref_point = compute_ref_point(y)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([args.mc_samples]))
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=x,
            sampler=sampler,
            prune_baseline=True,
        )

        try:
            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=args.num_restarts,
                raw_samples=args.raw_samples,
            )
            new_x = candidate.detach().view(1, -1)
        except Exception as exc:
            print(f"[WARN] optimize_acqf failed ({exc}); falling back to Sobol candidate.")
            new_x = sobol.draw(1).to(dtype=torch.double)

        cfg = space.decode(new_x[0].detach().cpu().numpy())
        key = config_to_key(cfg)

        attempts = 0
        while key in seen_configs and attempts < 40:
            new_x = sobol.draw(1).to(dtype=torch.double)
            cfg = space.decode(new_x[0].detach().cpu().numpy())
            key = config_to_key(cfg)
            attempts += 1

        seen_configs.add(key)

        t0 = time.time()
        raw = train_and_evaluate(
            config=cfg,
            dataset_name=args.dataset,
            seed=args.seed + trial_id,
            train_subset_size=args.train_subset_size,
            num_workers=args.num_workers,
        )
        elapsed = time.time() - t0

        new_y = objective_transform(raw)

        x = torch.cat([x, new_x], dim=0)
        y = torch.cat([y, new_y.view(1, -1)], dim=0)

        row = {
            "trial": trial_id,
            "phase": "bo",
            "dataset": args.dataset,
            **cfg,
            **raw,
            "objective_accuracy": float(new_y[0].item()),
            "objective_neg_infer_ms": float(new_y[1].item()),
            "objective_neg_params_m": float(new_y[2].item()),
            "wall_time_s": elapsed,
        }
        results_rows.append(row)

        print(
            f"bo   {trial_id:03d} | acc={raw['accuracy']:.2%} | infer={raw['inference_ms']:.3f} ms | "
            f"params={raw['param_count']:,} | {elapsed:.1f}s"
        )

        all_csv, state_pt = save_progress(results_rows, x, y, args.out_dir, args.dataset)
        print(f"[Saved] {all_csv}")

    df = pd.DataFrame(results_rows)
    vals = df[["accuracy", "inference_ms", "param_count"]].to_numpy(dtype=float)
    mask = pareto_mask(vals, maximize=[True, False, False])
    pareto_df = df.loc[mask].sort_values("accuracy", ascending=False).reset_index(drop=True)

    pareto_csv = os.path.join(args.out_dir, f"botorch_{args.dataset}_pareto.csv")
    pareto_df.to_csv(pareto_csv, index=False)

    fig_path = plot_results(df, pareto_df, args.out_dir, args.dataset)

    print("\n" + "=" * 90)
    print("Run complete")
    print("=" * 90)
    print(f"All trials : {all_csv}")
    print(f"Pareto CSV : {pareto_csv}")
    print(f"Plot       : {fig_path}")
    print(f"State file : {state_pt}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BoTorch qNEHVI optimization for MOML project")

    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "fashion_mnist"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--total-trials", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=20)

    parser.add_argument(
        "--train-subset-size",
        type=str,
        default="full",
        help="Use 'full' for full training set, 'auto' for project default, or an integer.",
    )
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--num-restarts", type=int, default=16)
    parser.add_argument("--raw-samples", type=int, default=512)
    parser.add_argument("--mc-samples", type=int, default=128)

    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results/botorch_gpu")

    args = parser.parse_args()

    if args.total_trials < 2:
        raise ValueError("--total-trials must be at least 2")
    if args.n_init < 1:
        raise ValueError("--n-init must be at least 1")

    # Parse train_subset_size
    tss = args.train_subset_size.strip().lower()
    if tss == "full":
        args.train_subset_size = None
    elif tss == "auto":
        args.train_subset_size = "auto"
    else:
        args.train_subset_size = int(tss)

    return args


if __name__ == "__main__":
    run_botorch(parse_args())
