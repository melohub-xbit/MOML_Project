

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

OBJECTIVE_KEYS = ("accuracy", "inference_ms", "param_count")
DEFAULT_REFERENCE_POINT = (0.0, 1000.0, 10_000_000.0)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_objective_matrix(csv_path: str | Path) -> np.ndarray:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [k for k in OBJECTIVE_KEYS if k not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV missing required columns {missing}: {path}")

        for row in reader:
            rows.append([
                float(row["accuracy"]),
                float(row["inference_ms"]),
                float(row["param_count"]),
            ])

    if not rows:
        raise ValueError(f"CSV has no rows: {path}")

    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Objective-space transforms
# ---------------------------------------------------------------------------


def _to_minimization_space(points: np.ndarray) -> np.ndarray:
    
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected points with shape (N, 3)")

    return np.column_stack((1.0 - points[:, 0], points[:, 1], points[:, 2]))


def _reference_to_minimization_space(reference_point: tuple[float, float, float]) -> np.ndarray:
    acc_ref, inf_ref, param_ref = reference_point
    return np.asarray([1.0 - acc_ref, inf_ref, param_ref], dtype=np.float64)


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------


def _pareto_mask_min(points_min: np.ndarray) -> np.ndarray:
    n = points_min.shape[0]
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not mask[i]:
            continue
        p = points_min[i]

        # Any point that is <= in all dims and < in at least one dim dominates p.
        dominates_i = np.all(points_min <= p, axis=1) & np.any(points_min < p, axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            mask[i] = False

    return mask


def pareto_front(points_mixed: np.ndarray) -> np.ndarray:
    
    points_min = _to_minimization_space(points_mixed)
    mask = _pareto_mask_min(points_min)
    return points_mixed[mask]


# ---------------------------------------------------------------------------
# Hypervolume (exact 3D for minimization)
# ---------------------------------------------------------------------------


def _hypervolume_2d_min(points_2d: np.ndarray, reference_2d: np.ndarray) -> float:
    if points_2d.size == 0:
        return 0.0

    valid = points_2d[np.all(points_2d <= reference_2d, axis=1)]
    if valid.size == 0:
        return 0.0

    nd_mask = _pareto_mask_min(valid)
    nd = valid[nd_mask]

    # Sort by first objective ascending.
    order = np.argsort(nd[:, 0])
    nd = nd[order]

    area = 0.0
    current_z = reference_2d[1]

    for y, z in nd:
        if z < current_z:
            area += (reference_2d[0] - y) * (current_z - z)
            current_z = z

    return float(area)


def hypervolume_3d(points_mixed: np.ndarray, reference_point: tuple[float, float, float]) -> float:
    
    points_min = _to_minimization_space(points_mixed)
    ref_min = _reference_to_minimization_space(reference_point)

    valid = points_min[np.all(points_min <= ref_min, axis=1)]
    if valid.size == 0:
        return 0.0

    nd_mask = _pareto_mask_min(valid)
    nd = valid[nd_mask]

    # Sort by first objective (minimization axis) ascending.
    order = np.argsort(nd[:, 0])
    nd = nd[order]

    hv = 0.0
    for i in range(nd.shape[0]):
        x_i = nd[i, 0]
        x_next = nd[i + 1, 0] if i + 1 < nd.shape[0] else ref_min[0]
        dx = x_next - x_i
        if dx <= 0:
            continue

        # For this x-slice, active points are those with x <= x_i.
        active_2d = nd[: i + 1, 1:3]
        area = _hypervolume_2d_min(active_2d, ref_min[1:3])
        hv += dx * area

    return float(hv)


# ---------------------------------------------------------------------------
# Spacing and generational distance
# ---------------------------------------------------------------------------


def _minmax_normalize(points: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    denom = np.where((maxs - mins) < 1e-12, 1.0, (maxs - mins))
    return (points - mins) / denom


def spacing_metric(points_mixed: np.ndarray) -> float:
    
    if points_mixed.shape[0] < 2:
        return 0.0

    front_mixed = pareto_front(points_mixed)
    if front_mixed.shape[0] < 2:
        return 0.0

    front_min = _to_minimization_space(front_mixed)
    mins = front_min.min(axis=0)
    maxs = front_min.max(axis=0)
    norm = _minmax_normalize(front_min, mins, maxs)

    n = norm.shape[0]
    nearest = np.empty(n, dtype=np.float64)

    for i in range(n):
        dists = np.sum(np.abs(norm - norm[i]), axis=1)
        dists[i] = np.inf
        nearest[i] = np.min(dists)

    mean_d = float(np.mean(nearest))
    if n == 1:
        return 0.0

    spacing = np.sqrt(np.sum((nearest - mean_d) ** 2) / (n - 1))
    return float(spacing)


def generational_distance(approx_mixed: np.ndarray, reference_mixed: np.ndarray) -> float:
    if approx_mixed.size == 0:
        raise ValueError("approx_mixed is empty")
    if reference_mixed.size == 0:
        raise ValueError("reference_mixed is empty")

    approx_front = pareto_front(approx_mixed)
    ref_front = pareto_front(reference_mixed)

    approx_min = _to_minimization_space(approx_front)
    ref_min = _to_minimization_space(ref_front)

    combined = np.vstack([approx_min, ref_min])
    mins = combined.min(axis=0)
    maxs = combined.max(axis=0)

    approx_norm = _minmax_normalize(approx_min, mins, maxs)
    ref_norm = _minmax_normalize(ref_min, mins, maxs)

    sq_dists = []
    for p in approx_norm:
        d2 = np.sum((ref_norm - p) ** 2, axis=1)
        sq_dists.append(np.min(d2))

    gd = np.sqrt(np.mean(sq_dists))
    return float(gd)


# ---------------------------------------------------------------------------
# High-level API for Optuna artifacts
# ---------------------------------------------------------------------------


def evaluate_optuna_study_dir(
    study_dir: str | Path,
    reference_point: tuple[float, float, float] = DEFAULT_REFERENCE_POINT,
    save_json: bool = True,
) -> dict[str, Any]:
    
    root = Path(study_dir)
    trials_csv = root / "trials.csv"
    pareto_csv = root / "pareto_front.csv"

    trials = _load_objective_matrix(trials_csv)
    pareto = _load_objective_matrix(pareto_csv)

    hv = hypervolume_3d(pareto, reference_point)
    spacing = spacing_metric(pareto)

    metrics = {
        "study_dir": str(root),
        "reference_point": {
            "accuracy": float(reference_point[0]),
            "inference_ms": float(reference_point[1]),
            "param_count": float(reference_point[2]),
        },
        "n_trials": int(trials.shape[0]),
        "n_pareto_points": int(pareto.shape[0]),
        "hypervolume": float(hv),
        "spacing": float(spacing),
    }

    if save_json:
        out_path = root / "pareto_metrics.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        metrics["metrics_json"] = str(out_path)

    return metrics


def compare_two_fronts(
    approx_front_csv: str | Path,
    other_front_csv: str | Path,
) -> dict[str, float]:
    
    approx = _load_objective_matrix(approx_front_csv)
    other = _load_objective_matrix(other_front_csv)

    merged = np.vstack([approx, other])
    reference = pareto_front(merged)

    return {
        "gd_approx_to_reference": float(generational_distance(approx, reference)),
        "gd_other_to_reference": float(generational_distance(other, reference)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pareto metrics for an Optuna study")
    parser.add_argument("--study-dir", type=str, required=True, help="Path to Optuna study directory")
    parser.add_argument("--ref-acc", type=float, default=0.0, help="Reference accuracy")
    parser.add_argument("--ref-inference-ms", type=float, default=1000.0, help="Reference inference (ms)")
    parser.add_argument("--ref-params", type=float, default=10_000_000.0, help="Reference param count")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = evaluate_optuna_study_dir(
        study_dir=args.study_dir,
        reference_point=(args.ref_acc, args.ref_inference_ms, args.ref_params),
        save_json=True,
    )

    print("Pareto metrics:")
    print(f"  n_trials       : {metrics['n_trials']}")
    print(f"  n_pareto       : {metrics['n_pareto_points']}")
    print(f"  hypervolume    : {metrics['hypervolume']:.6f}")
    print(f"  spacing        : {metrics['spacing']:.6f}")
    print(f"  metrics_json   : {metrics['metrics_json']}")


if __name__ == "__main__":
    main()
