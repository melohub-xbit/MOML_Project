"""
moo_optuna.py -- Optuna NSGA-II multi-objective optimization for MOML project.

Objectives (same as project spec):
    O1: Maximize classification accuracy
    O2: Minimize inference time (ms/sample on CPU)
    O3: Minimize model size (trainable parameter count)

This module is intentionally CUDA-only. If CUDA is not available, execution
fails immediately with a clear RuntimeError.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna

from data_loader import DEVICE, get_dataset_info
from train_eval import train_and_evaluate

OBJECTIVE_DIRECTIONS = ("maximize", "minimize", "minimize")
OBJECTIVE_NAMES = ("accuracy", "inference_ms", "param_count")

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


def _require_cuda() -> None:
    """Fail fast if CUDA is not active."""
    if DEVICE.type != "cuda":
        raise RuntimeError(
            "CUDA is required for this Optuna pipeline. "
            "Install CUDA-enabled PyTorch in the project venv and verify with: "
            "python -c \"import torch; print(torch.cuda.is_available())\""
        )


def _suggest_config(trial: optuna.trial.Trial) -> dict[str, Any]:
    """Sample one architecture/training configuration from the project search space."""
    return {
        "arch_type": trial.suggest_categorical(
            "arch_type", ["plain", "residual", "depthwise_separable"]
        ),
        "num_conv_layers": trial.suggest_int("num_conv_layers", 1, 4),
        "num_channels": trial.suggest_categorical("num_channels", [8, 16, 32, 64, 128]),
        "num_fc_units": trial.suggest_int("num_fc_units", 32, 256, step=32),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "num_epochs": trial.suggest_int("num_epochs", 5, 15),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
        "optimizer_type": trial.suggest_categorical("optimizer_type", ["SGD", "Adam"]),
        "input_resolution": trial.suggest_categorical("input_resolution", [16, 32]),
    }


def _build_objective(
    dataset_name: str,
    base_seed: int,
    train_subset_size: int,
    show_trial_progress: bool,
):
    """Create Optuna objective closure bound to one dataset and experiment settings."""

    def objective(trial: optuna.trial.Trial) -> tuple[float, float, float]:
        config = _suggest_config(trial)
        trial_seed = base_seed + trial.number

        results = train_and_evaluate(
            config=config,
            dataset_name=dataset_name,
            seed=trial_seed,
            train_subset_size=train_subset_size,
            show_progress=show_trial_progress,
        )

        trial.set_user_attr("seed", trial_seed)

        return (
            float(results["accuracy"]),
            float(results["inference_ms"]),
            float(results["param_count"]),
        )

    return objective


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None
    ]


def _write_trials_csv(trials: list[optuna.trial.FrozenTrial], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_number",
        "accuracy",
        "inference_ms",
        "param_count",
        "seed",
        *SEARCH_SPACE_KEYS,
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            row = {
                "trial_number": trial.number,
                "accuracy": trial.values[0],
                "inference_ms": trial.values[1],
                "param_count": trial.values[2],
                "seed": trial.user_attrs.get("seed"),
            }
            for key in SEARCH_SPACE_KEYS:
                row[key] = trial.params.get(key)
            writer.writerow(row)


def _write_pareto_csv(best_trials: list[optuna.trial.FrozenTrial], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_number",
        "accuracy",
        "inference_ms",
        "param_count",
        "seed",
        *SEARCH_SPACE_KEYS,
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in best_trials:
            row = {
                "trial_number": trial.number,
                "accuracy": trial.values[0],
                "inference_ms": trial.values[1],
                "param_count": trial.values[2],
                "seed": trial.user_attrs.get("seed"),
            }
            for key in SEARCH_SPACE_KEYS:
                row[key] = trial.params.get(key)
            writer.writerow(row)


def run_optuna_study(
    dataset_name: str,
    n_trials: int,
    train_subset_size: int = 20_000,
    seed: int = 42,
    population_size: int = 24,
    output_root: str = "results/optuna",
    show_progress_bar: bool = False,
    show_trial_progress: bool = False,
) -> dict[str, Any]:
    """Run one NSGA-II study and persist trial + Pareto artifacts.

    Parameters
    ----------
    dataset_name : str
        "cifar10" or "fashion_mnist".
    n_trials : int
        Number of Optuna trials.
    train_subset_size : int
        Number of training samples per trial (20,000 recommended on CUDA).
    seed : int
        Base seed. Trial seed is base_seed + trial_number.
    population_size : int
        NSGA-II population size.
    output_root : str
        Root directory for all saved artifacts.
    show_progress_bar : bool
        If True, show Optuna study-level progress bar.
    show_trial_progress : bool
        If True, show per-trial train/eval tqdm bars.

    Returns
    -------
    dict
        Summary information and output paths for downstream notebook usage.
    """
    _require_cuda()
    dataset_key = dataset_name.lower().replace("-", "_")
    get_dataset_info(dataset_key)  # Validate dataset name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"optuna_nsga2_{dataset_key}_{timestamp}"

    dataset_dir = Path(output_root) / dataset_key / study_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    storage_path = dataset_dir / "study.sqlite3"
    storage_url = f"sqlite:///{storage_path.as_posix()}"

    sampler = optuna.samplers.NSGAIISampler(seed=seed, population_size=population_size)
    study = optuna.create_study(
        study_name=study_name,
        directions=list(OBJECTIVE_DIRECTIONS),
        sampler=sampler,
        storage=storage_url,
        load_if_exists=False,
    )

    if hasattr(study, "set_metric_names"):
        study.set_metric_names(list(OBJECTIVE_NAMES))

    t0 = time.time()
    objective = _build_objective(
        dataset_name=dataset_key,
        base_seed=seed,
        train_subset_size=train_subset_size,
        show_trial_progress=show_trial_progress,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=show_progress_bar,
    )
    elapsed_seconds = time.time() - t0

    completed = _completed_trials(study)
    pareto = study.best_trials

    trials_csv = dataset_dir / "trials.csv"
    pareto_csv = dataset_dir / "pareto_front.csv"
    summary_json = dataset_dir / "summary.json"

    _write_trials_csv(completed, trials_csv)
    _write_pareto_csv(pareto, pareto_csv)

    accuracies = [trial.values[0] for trial in completed]
    infer_ms = [trial.values[1] for trial in completed]
    params = [trial.values[2] for trial in completed]

    summary = {
        "study_name": study_name,
        "dataset_name": dataset_key,
        "device": str(DEVICE),
        "n_trials_requested": n_trials,
        "n_trials_completed": len(completed),
        "n_pareto_points": len(pareto),
        "train_subset_size": train_subset_size,
        "population_size": population_size,
        "seed": seed,
        "elapsed_seconds": elapsed_seconds,
        "objective_names": list(OBJECTIVE_NAMES),
        "accuracy_max": max(accuracies),
        "accuracy_min": min(accuracies),
        "inference_ms_min": min(infer_ms),
        "inference_ms_max": max(infer_ms),
        "param_count_min": min(params),
        "param_count_max": max(params),
        "artifacts": {
            "storage": str(storage_path),
            "trials_csv": str(trials_csv),
            "pareto_csv": str(pareto_csv),
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    result = {
        "study": study,
        "summary": summary,
        "paths": {
            "study_dir": str(dataset_dir),
            "storage": str(storage_path),
            "trials_csv": str(trials_csv),
            "pareto_csv": str(pareto_csv),
            "summary_json": str(summary_json),
        },
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna NSGA-II for MOML project")
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["cifar10", "fashion_mnist", "both"],
        help="Dataset to optimize",
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials per dataset")
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=20_000,
        help="Training subset size per trial",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=24,
        help="NSGA-II population size",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/optuna",
        help="Output directory root",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    datasets = ["cifar10", "fashion_mnist"] if args.dataset == "both" else [args.dataset]

    for dataset_name in datasets:
        print("=" * 80)
        print(f"Running Optuna NSGA-II on: {dataset_name}")
        print(f"Device: {DEVICE}")
        print(f"Trials: {args.n_trials} | Subset: {args.train_subset_size} | Population: {args.population_size}")
        print("=" * 80)

        result = run_optuna_study(
            dataset_name=dataset_name,
            n_trials=args.n_trials,
            train_subset_size=args.train_subset_size,
            seed=args.seed,
            population_size=args.population_size,
            output_root=args.output_root,
            show_progress_bar=True,
            show_trial_progress=True,
        )

        print("\nCompleted study.")
        print(f"Study dir       : {result['paths']['study_dir']}")
        print(f"Trials CSV      : {result['paths']['trials_csv']}")
        print(f"Pareto CSV      : {result['paths']['pareto_csv']}")
        print(f"Summary JSON    : {result['paths']['summary_json']}")
        print(f"Pareto solutions: {result['summary']['n_pareto_points']}")
        print()


if __name__ == "__main__":
    main()
