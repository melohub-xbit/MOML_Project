"""
moo_pymoo.py -- pymoo NSGA-II multi-objective optimization for MOML project.

3 objectives:
    O1: Maximize accuracy
    O2: Minimize inference_ms (CPU, batch=1)
    O3: Minimize param_count

pymoo internally minimizes everything; accuracy is fed in as -accuracy and
restored on output.

Each completed trial is appended to trials.csv immediately so a crash mid-run
loses no work. The final population's non-dominated set is written to
pareto_front.csv. Run analyze_pymoo.py afterwards for metrics and plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableMating,
    MixedVariableSampling,
)
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Choice, Integer, Real
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from data_loader import DEVICE, PROJECT_ROOT, get_dataset_info
from train_eval import train_and_evaluate

DEFAULT_OUTPUT_ROOT = str(PROJECT_ROOT / "results" / "pymoo")

SEARCH_SPACE_KEYS = [
    "arch_type",
    "num_conv_layers",
    "num_channels",
    "num_fc_units",
    "learning_rate",
    "batch_size",
    "num_epochs",
    "dropout_rate",
    "optimizer_type",
    "input_resolution",
]


def make_search_space() -> dict:
    """Mixed-variable search space, deliberately reduced.

    Architecture decision variables are limited to conv-layer count and
    conv-channel width. arch_type and num_fc_units are pinned to single
    options so the only architectural levers NSGA-II pulls are the two
    that directly drive the param/inference trade-off via the conv stack:

      arch_type      : fixed to "plain"   (drop residual / depthwise_separable)
      num_fc_units   : fixed to 128       (FC head not a search dim)

    The remaining 7 dims are training/preprocessing hyperparameters
    (learning rate, batch size, epochs, dropout, optimizer, resolution)
    plus the two conv knobs above. Keeping per-trial time near 10-25s on a
    small GPU lets the full study fit the 2-hour wall-clock cap.
    """
    return {
        "arch_type": Choice(options=["plain"]),
        "num_conv_layers": Choice(options=[1, 2, 4]),
        "num_channels": Choice(options=[8, 16, 32]),
        "num_fc_units": Choice(options=[64,128,256]),
        "learning_rate": Real(bounds=(1e-5, 1e-2)),
        "batch_size": Choice(options=[64,128,256]),
        "num_epochs": Integer(bounds=(5, 10)),
        "dropout_rate": Real(bounds=(0.0, 0.5)),
        "optimizer_type": Choice(options=["SGD", "Adam"]),
        "input_resolution": Choice(options=[16, 32]),
    }


def _normalize_cfg(raw: dict[str, Any], max_resolution: int) -> dict[str, Any]:
    """Cast pymoo numpy types to plain Python and clamp resolution.

    Resolution > dataset's native size would force upsampling, which the
    transforms in data_loader don't apply -- clamp to the dataset's max.
    """
    cfg = {k: raw[k] for k in SEARCH_SPACE_KEYS}
    # PyTorch's BatchSampler rejects np.int64 (isinstance(int) check), so cast
    # every numeric field down to plain Python int / float.
    for k in ("num_conv_layers", "num_channels", "num_fc_units", "num_epochs", "batch_size", "input_resolution"):
        cfg[k] = int(cfg[k])
    for k in ("learning_rate", "dropout_rate"):
        cfg[k] = float(cfg[k])
    cfg["input_resolution"] = min(cfg["input_resolution"], max_resolution)
    return cfg


class ImageClsProblem(ElementwiseProblem):
    """One pymoo evaluation == one full train_and_evaluate trial."""

    def __init__(
        self,
        dataset_name: str,
        train_subset_size: int | None,
        base_seed: int,
        num_workers: int,
        trial_log_path: Path,
        max_resolution: int,
        use_amp: bool = False,
        inference_warmup: int = 50,
        inference_timed: int = 500,
    ) -> None:
        super().__init__(vars=make_search_space(), n_obj=3, n_ieq_constr=0)
        self.dataset_name = dataset_name
        self.train_subset_size = train_subset_size
        self.base_seed = base_seed
        self.num_workers = num_workers
        self.trial_log_path = Path(trial_log_path)
        self.max_resolution = max_resolution
        self.use_amp = use_amp
        self.inference_warmup = inference_warmup
        self.inference_timed = inference_timed
        self.n_evaluated = 0
        self._init_log()

    def _init_log(self) -> None:
        self.trial_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trial_log_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "trial_number",
                    "seed",
                    "wall_time_s",
                    "accuracy",
                    "inference_ms",
                    "param_count",
                    *SEARCH_SPACE_KEYS,
                ]
            )

    def _evaluate(self, x, out, *args, **kwargs):  # type: ignore[override]
        cfg = _normalize_cfg(x, self.max_resolution)
        trial_number = self.n_evaluated
        seed = self.base_seed + trial_number

        t0 = time.time()
        results = train_and_evaluate(
            config=cfg,
            dataset_name=self.dataset_name,
            seed=seed,
            train_subset_size=self.train_subset_size,
            show_progress=False,
            num_workers=self.num_workers,
            use_amp=self.use_amp,
            inference_warmup=self.inference_warmup,
            inference_timed=self.inference_timed,
        )
        elapsed = time.time() - t0
        self.n_evaluated += 1

        acc = float(results["accuracy"])
        ms = float(results["inference_ms"])
        params = float(results["param_count"])

        # pymoo minimizes everything -- flip accuracy.
        out["F"] = np.array([-acc, ms, params], dtype=float)

        # Stream the trial to disk now (crash-safety).
        with self.trial_log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    trial_number,
                    seed,
                    f"{elapsed:.3f}",
                    f"{acc:.6f}",
                    f"{ms:.6f}",
                    int(params),
                    *[cfg[k] for k in SEARCH_SPACE_KEYS],
                ]
            )

        # Compact stdout marker so a long run is still legible.
        print(
            f"  trial {trial_number:03d} | acc={acc:.4f} | ms={ms:7.3f} | "
            f"params={int(params):>9,} | {elapsed:5.1f}s | "
            f"{cfg['arch_type']}/{cfg['num_conv_layers']}L/{cfg['num_channels']}ch/"
            f"{cfg['num_fc_units']}fc/bs{cfg['batch_size']}"
        )


class GenLogger(Callback):
    """Print a one-line status at the end of each NSGA-II generation."""

    def __init__(self, total_gens: int, t0: float) -> None:
        super().__init__()
        self.total_gens = total_gens
        self.t0 = t0

    def notify(self, algorithm) -> None:
        elapsed = time.time() - self.t0
        gen = algorithm.n_gen
        n_evals = algorithm.evaluator.n_eval
        opt_F = algorithm.opt.get("F") if algorithm.opt is not None else None
        n_pareto = 0 if opt_F is None else len(opt_F)
        print(
            f"[gen {gen:>2d}/{self.total_gens}] evals={n_evals:>3d} "
            f"pareto={n_pareto:>2d} elapsed={elapsed/60:.1f}m"
        )


def run_pymoo_study(
    dataset_name: str = "fashion_mnist",
    pop_size: int = 30,
    n_gen: int = 6,
    train_subset_size: int | None = 8000,
    seed: int = 42,
    num_workers: int = 2,
    use_amp: bool = False,
    inference_warmup: int = 50,
    inference_timed: int = 500,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, Any], Path]:
    dataset_key = dataset_name.lower().replace("-", "_")
    ds_info = get_dataset_info(dataset_key)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"pymoo_nsga2_{dataset_key}_{timestamp}"
    study_dir = Path(output_root) / dataset_key / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    trials_csv = study_dir / "trials.csv"

    problem = ImageClsProblem(
        dataset_name=dataset_key,
        train_subset_size=train_subset_size,
        base_seed=seed,
        num_workers=num_workers,
        trial_log_path=trials_csv,
        max_resolution=int(ds_info["default_resolution"]),
        use_amp=use_amp,
        inference_warmup=inference_warmup,
        inference_timed=inference_timed,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )

    t0 = time.time()
    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False,
        callback=GenLogger(n_gen, t0),
        save_history=False,
    )
    elapsed = time.time() - t0

    # Final-population non-dominated set.
    pareto_csv = study_dir / "pareto_front.csv"
    with pareto_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["accuracy", "inference_ms", "param_count", *SEARCH_SPACE_KEYS]
        )
        for x_i, F_i in zip(res.X, res.F):
            cfg = _normalize_cfg(x_i, int(ds_info["default_resolution"]))
            writer.writerow(
                [
                    f"{-float(F_i[0]):.6f}",
                    f"{float(F_i[1]):.6f}",
                    int(F_i[2]),
                    *[cfg[k] for k in SEARCH_SPACE_KEYS],
                ]
            )

    summary = {
        "study_name": study_name,
        "framework": "pymoo",
        "algorithm": "NSGA2",
        "dataset_name": dataset_key,
        "device": str(DEVICE),
        "n_trials_completed": problem.n_evaluated,
        "n_pareto_points": int(len(res.F)),
        "pop_size": pop_size,
        "n_gen": n_gen,
        "train_subset_size": train_subset_size,
        "seed": seed,
        "num_workers": num_workers,
        "use_amp": use_amp,
        "inference_warmup": inference_warmup,
        "inference_timed": inference_timed,
        "elapsed_seconds": elapsed,
        "objective_names": ["accuracy", "inference_ms", "param_count"],
        "objective_directions": ["maximize", "minimize", "minimize"],
        "artifacts": {
            "trials_csv": str(trials_csv),
            "pareto_csv": str(pareto_csv),
        },
    }
    summary_path = study_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[done] elapsed={elapsed/60:.1f}m | trials={problem.n_evaluated} "
        f"| pareto={len(res.F)}"
    )
    print(f"  trials : {trials_csv}")
    print(f"  pareto : {pareto_csv}")
    print(f"  summary: {summary_path}")

    return summary, study_dir


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="pymoo NSGA-II for MOML project")
    ap.add_argument(
        "--dataset",
        default="fashion_mnist",
        choices=["cifar10", "fashion_mnist"],
    )
    ap.add_argument("--pop-size", type=int, default=30)
    ap.add_argument("--n-gen", type=int, default=6)
    ap.add_argument("--train-subset-size", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    print("=" * 80)
    print(f"pymoo NSGA-II | device={DEVICE} | dataset={args.dataset}")
    print(
        f"pop={args.pop_size} gen={args.n_gen} "
        f"(max evals={args.pop_size * args.n_gen}) "
        f"subset={args.train_subset_size} seed={args.seed}"
    )
    print("=" * 80)
    run_pymoo_study(
        dataset_name=args.dataset,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        train_subset_size=args.train_subset_size,
        seed=args.seed,
        num_workers=args.num_workers,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
