

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from data_loader import DEVICE, PROJECT_ROOT, get_dataset_info
from train_eval import train_and_evaluate

DEFAULT_OUTPUT_ROOT = str(PROJECT_ROOT / "results" / "botorch")

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


# ---------------------------------------------------------------------------
# Search space - mirrors moo_pymoo.make_search_space() exactly.
# ---------------------------------------------------------------------------


class SearchSpace:
    

    def __init__(self) -> None:
        self.arch_type = ["plain"]
        self.num_conv_layers = [1, 2, 4]
        self.num_channels = [8, 16, 32]
        self.num_fc_units = [64,128,256]
        # learning_rate: continuous, log-scale in [1e-5, 1e-2]
        # batch_size: categorical {16, 64}
        self.batch_size = [64,128,256]
        # num_epochs: integer in [5, 10]
        # dropout_rate: continuous in [0.0, 0.5]
        self.optimizer_type = ["SGD", "Adam"]
        self.input_resolution = [16, 32]

        self.dim = 10
        # Index layout in the unit cube:
        #   0 arch_type        (cat)
        #   1 num_conv_layers  (cat)
        #   2 num_channels     (cat)
        #   3 num_fc_units     (cat)
        #   4 learning_rate    (real, log-scale)
        #   5 batch_size       (cat)
        #   6 num_epochs       (int)
        #   7 dropout_rate     (real)
        #   8 optimizer_type   (cat)
        #   9 input_resolution (cat)

    @staticmethod
    def _pick_from_unit(u: float, choices: list[Any]) -> Any:
        idx = int(np.clip(np.floor(u * len(choices)), 0, len(choices) - 1))
        return choices[idx]

    @staticmethod
    def _int_from_unit(u: float, low: int, high: int) -> int:
        # Inclusive on both sides.
        idx = int(np.clip(np.floor(u * (high - low + 1)), 0, high - low))
        return low + idx

    def decode(self, x_unit: np.ndarray, max_resolution: int) -> dict[str, Any]:
        x = np.asarray(x_unit, dtype=float).reshape(-1)

        # learning_rate: log-uniform in [1e-5, 1e-2]
        log_lr_lo, log_lr_hi = np.log10(1e-5), np.log10(1e-2)
        lr = float(10 ** (log_lr_lo + (log_lr_hi - log_lr_lo) * x[4]))

        cfg: dict[str, Any] = {
            "arch_type": self._pick_from_unit(x[0], self.arch_type),
            "num_conv_layers": int(self._pick_from_unit(x[1], self.num_conv_layers)),
            "num_channels": int(self._pick_from_unit(x[2], self.num_channels)),
            "num_fc_units": int(self._pick_from_unit(x[3], self.num_fc_units)),
            "learning_rate": lr,
            "batch_size": int(self._pick_from_unit(x[5], self.batch_size)),
            "num_epochs": self._int_from_unit(x[6], 5, 10),
            "dropout_rate": float(np.clip(x[7] * 0.5, 0.0, 0.5)),
            "optimizer_type": self._pick_from_unit(x[8], self.optimizer_type),
            "input_resolution": int(self._pick_from_unit(x[9], self.input_resolution)),
        }

        # Clamp resolution to dataset's native size to avoid upsampling.
        cfg["input_resolution"] = min(cfg["input_resolution"], max_resolution)
        return cfg


def _config_key(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True)


def _objective_transform(raw: dict[str, Any]) -> torch.Tensor:
    
    return torch.tensor(
        [
            float(raw["accuracy"]),
            -float(raw["inference_ms"]),
            -float(raw["param_count"]) / 1_000_000.0,
        ],
        dtype=torch.double,
    )


def _compute_ref_point(y: torch.Tensor) -> list[float]:
    
    margin = torch.tensor([0.02, 5.0, 0.5], dtype=torch.double)
    ref = y.min(dim=0).values - margin
    return ref.tolist()


# ---------------------------------------------------------------------------
# Trial-level CSV streaming (matches moo_pymoo schema).
# ---------------------------------------------------------------------------


def _init_trial_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "trial_number",
                "seed",
                "wall_time_s",
                "phase",
                "accuracy",
                "inference_ms",
                "param_count",
                *SEARCH_SPACE_KEYS,
            ]
        )


def _append_trial_row(
    path: Path,
    *,
    trial_number: int,
    seed: int,
    wall_time_s: float,
    phase: str,
    cfg: dict[str, Any],
    raw: dict[str, Any],
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                trial_number,
                seed,
                f"{wall_time_s:.3f}",
                phase,
                f"{float(raw['accuracy']):.6f}",
                f"{float(raw['inference_ms']):.6f}",
                int(raw["param_count"]),
                *[cfg[k] for k in SEARCH_SPACE_KEYS],
            ]
        )


def _pareto_mask_min(values_min: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows in minimization space."""
    n = values_min.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        p = values_min[i]
        dominates = np.all(values_min <= p, axis=1) & np.any(values_min < p, axis=1)
        dominates[i] = False
        if dominates.any():
            keep[i] = False
    return keep


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_botorch_study(
    dataset_name: str = "fashion_mnist",
    total_trials: int = 80,
    n_init: int = 16,
    train_subset_size: int | None = 8000,
    seed: int = 42,
    num_workers: int = 2,
    num_restarts: int = 3,
    raw_samples: int = 64,
    mc_samples: int = 16,
    use_amp: bool = False,
    inference_warmup: int = 50,
    inference_timed: int = 500,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, Any], Path]:
    if n_init < 1 or n_init >= total_trials:
        raise ValueError("n_init must be in [1, total_trials)")

    dataset_key = dataset_name.lower().replace("-", "_")
    ds_info = get_dataset_info(dataset_key)
    max_resolution = int(ds_info["default_resolution"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"botorch_qnehvi_{dataset_key}_{timestamp}"
    study_dir = Path(output_root) / dataset_key / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    trials_csv = study_dir / "trials.csv"
    _init_trial_log(trials_csv)

    space = SearchSpace()
    bounds = torch.stack(
        [
            torch.zeros(space.dim, dtype=torch.double),
            torch.ones(space.dim, dtype=torch.double),
        ]
    )

    sobol = torch.quasirandom.SobolEngine(
        dimension=space.dim, scramble=True, seed=seed
    )
    x_init = sobol.draw(n_init).to(dtype=torch.double)

    x_all = x_init.clone()
    y_list: list[torch.Tensor] = []
    raw_results: list[dict[str, Any]] = []
    cfg_list: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    print("=" * 80)
    print(
        f"BoTorch qNEHVI | device={DEVICE} | dataset={dataset_key} | "
        f"total_trials={total_trials} (init={n_init}, bo={total_trials - n_init})"
    )
    print("=" * 80)

    t0 = time.time()

    # ---- Sobol initialization phase --------------------------------------
    print("\n[phase] Sobol init")
    for i in range(n_init):
        cfg = space.decode(x_all[i].cpu().numpy(), max_resolution)
        # Resample if Sobol mapped to a duplicate discrete config.
        attempts = 0
        while _config_key(cfg) in seen_keys and attempts < 30:
            x_all[i] = sobol.draw(1).to(dtype=torch.double)[0]
            cfg = space.decode(x_all[i].cpu().numpy(), max_resolution)
            attempts += 1
        seen_keys.add(_config_key(cfg))

        trial_seed = seed + i
        ts = time.time()
        raw = train_and_evaluate(
            config=cfg,
            dataset_name=dataset_key,
            seed=trial_seed,
            train_subset_size=train_subset_size,
            show_progress=False,
            num_workers=num_workers,
            use_amp=use_amp,
            inference_warmup=inference_warmup,
            inference_timed=inference_timed,
        )
        elapsed = time.time() - ts

        y_list.append(_objective_transform(raw))
        raw_results.append(raw)
        cfg_list.append(cfg)

        _append_trial_row(
            trials_csv,
            trial_number=i,
            seed=trial_seed,
            wall_time_s=elapsed,
            phase="init",
            cfg=cfg,
            raw=raw,
        )
        print(
            f"  trial {i:03d} | acc={raw['accuracy']:.4f} | ms={raw['inference_ms']:7.3f} | "
            f"params={int(raw['param_count']):>9,} | {elapsed:5.1f}s | init | "
            f"{cfg['arch_type']}/{cfg['num_conv_layers']}L/{cfg['num_channels']}ch/"
            f"{cfg['num_fc_units']}fc/bs{cfg['batch_size']}"
        )

    y_all = torch.stack(y_list, dim=0)

    # ---- Bayesian optimization phase -------------------------------------
    n_bo = total_trials - n_init
    print(f"\n[phase] BO ({n_bo} iters)")
    for j in range(n_bo):
        trial_id = n_init + j

        try:
            model = SingleTaskGP(
                x_all,
                y_all,
                outcome_transform=Standardize(m=y_all.shape[-1]),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            ref_point = _compute_ref_point(y_all)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
            acq = qLogNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point,
                X_baseline=x_all,
                sampler=sampler,
                prune_baseline=True,
                cache_root=False,
            )
            candidate, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"maxiter": 50},
            )
            new_x = candidate.detach().view(1, -1)
        except Exception as exc:
            # Graceful fallback: if the GP fit or acqf optimization fails
            # (rare, but happens on degenerate posteriors), draw a Sobol
            # candidate so the loop still makes progress.
            print(f"  [warn] acqf failure: {exc}; falling back to Sobol candidate")
            new_x = sobol.draw(1).to(dtype=torch.double)

        cfg = space.decode(new_x[0].cpu().numpy(), max_resolution)
        attempts = 0
        while _config_key(cfg) in seen_keys and attempts < 30:
            new_x = sobol.draw(1).to(dtype=torch.double)
            cfg = space.decode(new_x[0].cpu().numpy(), max_resolution)
            attempts += 1
        seen_keys.add(_config_key(cfg))

        trial_seed = seed + trial_id
        ts = time.time()
        raw = train_and_evaluate(
            config=cfg,
            dataset_name=dataset_key,
            seed=trial_seed,
            train_subset_size=train_subset_size,
            show_progress=False,
            num_workers=num_workers,
            use_amp=use_amp,
            inference_warmup=inference_warmup,
            inference_timed=inference_timed,
        )
        elapsed = time.time() - ts

        new_y = _objective_transform(raw).view(1, -1)

        x_all = torch.cat([x_all, new_x], dim=0)
        y_all = torch.cat([y_all, new_y], dim=0)
        raw_results.append(raw)
        cfg_list.append(cfg)

        _append_trial_row(
            trials_csv,
            trial_number=trial_id,
            seed=trial_seed,
            wall_time_s=elapsed,
            phase="bo",
            cfg=cfg,
            raw=raw,
        )
        print(
            f"  trial {trial_id:03d} | acc={raw['accuracy']:.4f} | ms={raw['inference_ms']:7.3f} | "
            f"params={int(raw['param_count']):>9,} | {elapsed:5.1f}s | bo "
            f"({(time.time()-t0)/60:.1f}m total) | "
            f"{cfg['arch_type']}/{cfg['num_conv_layers']}L/{cfg['num_channels']}ch/"
            f"{cfg['num_fc_units']}fc/bs{cfg['batch_size']}"
        )

    elapsed_total = time.time() - t0

    # ---- Pareto extraction over all trials -------------------------------
    obj_min = np.array(
        [
            [
                1.0 - float(r["accuracy"]),
                float(r["inference_ms"]),
                float(r["param_count"]),
            ]
            for r in raw_results
        ],
        dtype=np.float64,
    )
    pareto_mask = _pareto_mask_min(obj_min)

    pareto_csv = study_dir / "pareto_front.csv"
    with pareto_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["accuracy", "inference_ms", "param_count", *SEARCH_SPACE_KEYS]
        )
        for keep, raw, cfg in zip(pareto_mask, raw_results, cfg_list):
            if not keep:
                continue
            writer.writerow(
                [
                    f"{float(raw['accuracy']):.6f}",
                    f"{float(raw['inference_ms']):.6f}",
                    int(raw["param_count"]),
                    *[cfg[k] for k in SEARCH_SPACE_KEYS],
                ]
            )

    n_pareto = int(pareto_mask.sum())

    summary = {
        "study_name": study_name,
        "framework": "botorch",
        "algorithm": "qNEHVI",
        "dataset_name": dataset_key,
        "device": str(DEVICE),
        "n_trials_completed": len(raw_results),
        "n_pareto_points": n_pareto,
        "n_init": n_init,
        "n_bo": total_trials - n_init,
        "train_subset_size": train_subset_size,
        "seed": seed,
        "num_workers": num_workers,
        "num_restarts": num_restarts,
        "raw_samples": raw_samples,
        "mc_samples": mc_samples,
        "use_amp": use_amp,
        "inference_warmup": inference_warmup,
        "inference_timed": inference_timed,
        "elapsed_seconds": elapsed_total,
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
        f"\n[done] elapsed={elapsed_total/60:.1f}m | "
        f"trials={len(raw_results)} | pareto={n_pareto}"
    )
    print(f"  trials : {trials_csv}")
    print(f"  pareto : {pareto_csv}")
    print(f"  summary: {summary_path}")

    return summary, study_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BoTorch qNEHVI for MOML project")
    ap.add_argument("--dataset", default="fashion_mnist", choices=["cifar10", "fashion_mnist"])
    ap.add_argument("--total-trials", type=int, default=80)
    ap.add_argument("--n-init", type=int, default=16)
    ap.add_argument("--train-subset-size", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--num-restarts", type=int, default=3)
    ap.add_argument("--raw-samples", type=int, default=64)
    ap.add_argument("--mc-samples", type=int, default=16)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_botorch_study(
        dataset_name=args.dataset,
        total_trials=args.total_trials,
        n_init=args.n_init,
        train_subset_size=args.train_subset_size,
        seed=args.seed,
        num_workers=args.num_workers,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
        mc_samples=args.mc_samples,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
