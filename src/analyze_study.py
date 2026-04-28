"""
analyze_study.py -- Framework-agnostic post-hoc analysis for an MOO study.

Works on any study dir written by moo_pymoo.py or moo_botorch.py: both write
the same trials.csv schema (accuracy, inference_ms, param_count + 10 search
space keys) and the same study_dir layout.

Inputs:
    --study-dir : study dir containing trials.csv. The non-dominated set is
                  recomputed from trials.csv so all trials contribute.
    --compare   : optional path to another framework's pareto_front.csv for
                  side-by-side plotting and generational-distance vs the
                  joint reference front (e.g. pymoo vs botorch comparison).

Outputs (written into --study-dir):
    pareto_metrics.json
    pareto_table.csv          (>=4 representative Pareto points)
    appendix_solution.json    (one fully-detailed Pareto-optimal config)
    plot_2d_panels.png
    plot_3d_scatter.png
    plot_3d_pareto.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from pareto_analysis import (
    DEFAULT_REFERENCE_POINT,
    OBJECTIVE_KEYS,
    _to_minimization_space,
    generational_distance,
    hypervolume_3d,
    pareto_front,
    spacing_metric,
)

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

# Plot palette
TRIAL_COLOR = "#bdc3c7"      # light gray for the cloud of all trials
PARETO_COLOR = "#c0392b"     # red for our Pareto front
COMPARE_COLOR = "#2980b9"    # blue for the comparison Pareto front
PICK_COLORS = {
    "fast":     "#e67e22",   # orange
    "small":    "#27ae60",   # green
    "accurate": "#8e44ad",   # purple
    "balanced": "#f39c12",   # gold
}


def _load_trials(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = dict(r)
            row["accuracy"] = float(row["accuracy"])
            row["inference_ms"] = float(row["inference_ms"])
            row["param_count"] = float(row["param_count"])
            rows.append(row)
    return rows


def _objective_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [[r[k] for k in OBJECTIVE_KEYS] for r in rows], dtype=np.float64
    )


def _pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute the non-dominated set from all observed trials."""
    pts = _objective_matrix(rows)
    front = pareto_front(pts)
    front_set = {tuple(p) for p in front}
    return [r for r in rows if (r["accuracy"], r["inference_ms"], r["param_count"]) in front_set]


def _representative_picks(pareto_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pick the canonical 4 Pareto points the report needs:
    fast (min ms), small (min params), accurate (max acc), balanced (closest to ideal)."""
    if not pareto_rows:
        return {}

    by_ms = min(pareto_rows, key=lambda r: r["inference_ms"])
    by_params = min(pareto_rows, key=lambda r: r["param_count"])
    by_acc = max(pareto_rows, key=lambda r: r["accuracy"])

    pts = _objective_matrix(pareto_rows)
    pts_min = _to_minimization_space(pts)  # [1-acc, ms, params]
    mins = pts_min.min(axis=0)
    maxs = pts_min.max(axis=0)
    denom = np.where((maxs - mins) < 1e-12, 1.0, (maxs - mins))
    norm = (pts_min - mins) / denom
    # Distance to ideal point (origin in normalized minimization space).
    d = np.linalg.norm(norm, axis=1)
    balanced = pareto_rows[int(np.argmin(d))]

    return {
        "fast": by_ms,
        "small": by_params,
        "accurate": by_acc,
        "balanced": balanced,
    }


def _write_pareto_table(picks: dict[str, dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "trial_number",
                "accuracy",
                "inference_ms",
                "param_count",
                *SEARCH_SPACE_KEYS,
            ]
        )
        for label, row in picks.items():
            w.writerow(
                [
                    label,
                    row.get("trial_number", ""),
                    f"{row['accuracy']:.4f}",
                    f"{row['inference_ms']:.4f}",
                    int(row["param_count"]),
                    *[row.get(k, "") for k in SEARCH_SPACE_KEYS],
                ]
            )


def _pareto_2d_indices(
    xs: list[float], ys: list[float], x_min: bool = True, y_min: bool = True
) -> list[int]:
    """Indices of 2D non-dominated points in the projection of a 3D Pareto set.

    A 3D-Pareto point can be 2D-dominated when projecting to two of the three
    objectives, so we recompute non-domination per panel before drawing the
    staircase line (otherwise the connecting line zig-zags through dominated
    points and looks wrong).
    """
    n = len(xs)
    keep: list[int] = []
    for i in range(n):
        xi, yi = xs[i], ys[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            xj, yj = xs[j], ys[j]
            x_better = (xj <= xi) if x_min else (xj >= xi)
            y_better = (yj <= yi) if y_min else (yj >= yi)
            x_strict = (xj < xi) if x_min else (xj > xi)
            y_strict = (yj < yi) if y_min else (yj > yi)
            if x_better and y_better and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def _draw_pareto_panel(
    ax,
    all_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    compare_rows: list[dict[str, Any]] | None,
    picks: dict[str, dict[str, Any]],
    *,
    x_fn,
    y_fn,
    x_label: str,
    y_label: str,
    panel_title: str,
    x_min: bool = True,
    y_min: bool = True,
    x_log: bool = False,
    y_log: bool = False,
    show_legend: bool = False,
) -> None:
    # 1. all trials cloud
    xs_all = [x_fn(r) for r in all_rows]
    ys_all = [y_fn(r) for r in all_rows]
    ax.scatter(
        xs_all, ys_all, c=TRIAL_COLOR, s=22, alpha=0.55,
        edgecolors="none", label=f"All trials ({len(all_rows)})", zorder=1,
    )

    # 2. our Pareto front + 2D-attainment staircase
    if pareto_rows:
        xs_p = [x_fn(r) for r in pareto_rows]
        ys_p = [y_fn(r) for r in pareto_rows]
        idx2d = _pareto_2d_indices(xs_p, ys_p, x_min=x_min, y_min=y_min)
        front = sorted([(xs_p[i], ys_p[i]) for i in idx2d], key=lambda p: p[0])
        if front:
            xf = [p[0] for p in front]
            yf = [p[1] for p in front]
            ax.step(
                xf, yf, where="post", color=PARETO_COLOR, linewidth=2.2,
                alpha=0.7, zorder=3,
            )
        non_idx = [i for i in range(len(xs_p)) if i not in idx2d]
        if non_idx:
            ax.scatter(
                [xs_p[i] for i in non_idx], [ys_p[i] for i in non_idx],
                facecolors="white", edgecolors=PARETO_COLOR, s=46,
                linewidths=1.0, zorder=2.5,
                label=f"3D Pareto, 2D-dominated ({len(non_idx)})",
            )
        ax.scatter(
            [xs_p[i] for i in idx2d], [ys_p[i] for i in idx2d],
            c=PARETO_COLOR, s=72, edgecolors="white", linewidths=0.9,
            zorder=4, label=f"2D non-dominated ({len(idx2d)})",
        )

    # 3. comparison front overlay
    if compare_rows:
        xs_c = [x_fn(r) for r in compare_rows]
        ys_c = [y_fn(r) for r in compare_rows]
        idx_c = _pareto_2d_indices(xs_c, ys_c, x_min=x_min, y_min=y_min)
        cf = sorted([(xs_c[i], ys_c[i]) for i in idx_c], key=lambda p: p[0])
        if cf:
            ax.step(
                [p[0] for p in cf], [p[1] for p in cf], where="post",
                color=COMPARE_COLOR, linewidth=2.0, alpha=0.55,
                linestyle="--", zorder=2,
            )
        ax.scatter(
            xs_c, ys_c, c=COMPARE_COLOR, s=44, marker="D",
            edgecolors="white", linewidths=0.7, alpha=0.85, zorder=3.5,
            label=f"Comparison ({len(compare_rows)})",
        )

    # 4. representative picks (stars + label boxes)
    if picks:
        for label, r in picks.items():
            px, py = x_fn(r), y_fn(r)
            ax.scatter(
                [px], [py], c=PICK_COLORS[label], s=210, marker="*",
                edgecolors="black", linewidths=1.2, zorder=5,
            )
            ax.annotate(
                label, xy=(px, py), xytext=(8, 8), textcoords="offset points",
                fontsize=9, fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.25", fc="white",
                    ec=PICK_COLORS[label], alpha=0.92,
                ),
                zorder=6,
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(panel_title, fontsize=11)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3, linestyle="--")
    if show_legend:
        ax.legend(loc="best", fontsize=8, framealpha=0.92)


def _plot_2d_panels(
    all_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    compare_pareto_rows: list[dict[str, Any]] | None,
    picks: dict[str, dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))

    _draw_pareto_panel(
        axes[0], all_rows, pareto_rows, compare_pareto_rows, picks,
        x_fn=lambda r: r["param_count"], y_fn=lambda r: r["accuracy"] * 100,
        x_label="Parameters (log)", y_label="Accuracy (%)",
        panel_title="Accuracy ↑   vs   Size ↓",
        x_min=True, y_min=False, x_log=True, show_legend=True,
    )
    _draw_pareto_panel(
        axes[1], all_rows, pareto_rows, compare_pareto_rows, picks,
        x_fn=lambda r: r["inference_ms"], y_fn=lambda r: r["accuracy"] * 100,
        x_label="Inference time (ms, CPU, batch=1, log)", y_label="Accuracy (%)",
        panel_title="Accuracy ↑   vs   Speed ↓",
        x_min=True, y_min=False, x_log=True,
    )
    _draw_pareto_panel(
        axes[2], all_rows, pareto_rows, compare_pareto_rows, picks,
        x_fn=lambda r: r["param_count"], y_fn=lambda r: r["inference_ms"],
        x_label="Parameters (log)", y_label="Inference time (ms, log)",
        panel_title="Speed ↓   vs   Size ↓",
        x_min=True, y_min=True, x_log=True, y_log=True,
    )

    fig.suptitle(title, fontsize=12.5, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_3d_pareto(
    all_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    compare_pareto_rows: list[dict[str, Any]] | None,
    picks: dict[str, dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    """Classic 3D Pareto-front scatter (MATLAB-style) — three labeled objective
    axes, all in minimization orientation so the front forms the canonical
    'bowl' near the origin.

    Inference time and parameter count span orders of magnitude on this problem,
    so we plot ``log10(value)`` for those two axes (mpl3d's native ``set_yscale
    ('log')`` is buggy in 3D — it leaves tick labels stacked on top of each
    other) and reformat the ticks back to physical units.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)
    from matplotlib.ticker import FuncFormatter

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    def _xyz(r: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(1.0 - r["accuracy"]),
            float(np.log10(max(r["inference_ms"], 1e-9))),
            float(np.log10(max(r["param_count"], 1.0))),
        )

    pareto_set = {id(r) for r in pareto_rows}
    non_pareto = [r for r in all_rows if id(r) not in pareto_set]

    if non_pareto:
        xs, ys, zs = zip(*(_xyz(r) for r in non_pareto))
        ax.scatter(
            xs, ys, zs, c=TRIAL_COLOR, s=14, alpha=0.45,
            edgecolors="none", depthshade=True,
            label=f"All trials ({len(all_rows)})",
        )

    if pareto_rows:
        xs, ys, zs = zip(*(_xyz(r) for r in pareto_rows))
        ax.scatter(
            xs, ys, zs, c=PARETO_COLOR, s=55, alpha=0.95,
            edgecolors="black", linewidths=0.6, depthshade=True,
            label=f"Pareto front ({len(pareto_rows)})",
        )

    if compare_pareto_rows:
        xs, ys, zs = zip(*(_xyz(r) for r in compare_pareto_rows))
        ax.scatter(
            xs, ys, zs, c=COMPARE_COLOR, marker="D", s=42, alpha=0.85,
            edgecolors="white", linewidths=0.5, depthshade=True,
            label=f"Comparison Pareto ({len(compare_pareto_rows)})",
        )

    if picks:
        for label, r in picks.items():
            x, y, z = _xyz(r)
            ax.scatter(
                [x], [y], [z], marker="*", s=320,
                facecolors=PICK_COLORS[label], edgecolors="black",
                linewidths=1.2, depthshade=False, label=label,
            )

    ax.set_xlabel("Error rate  (1 − accuracy)")
    ax.set_ylabel("Inference time (ms)")
    ax.set_zlabel("Parameters")
    ax.set_title(f"{title}\nPareto Front (all-minimize space)", fontsize=11)

    def _fmt_ms(v, _pos):
        x = 10.0 ** v
        if x < 1.0:
            return f"{x:.2f}"
        if x < 100.0:
            return f"{x:.1f}"
        return f"{int(x)}"

    def _fmt_params(v, _pos):
        x = 10.0 ** v
        if x >= 1_000_000:
            return f"{x/1_000_000:.1f}M"
        if x >= 1_000:
            return f"{x/1_000:.0f}K"
        return f"{int(x)}"

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_ms))
    ax.zaxis.set_major_formatter(FuncFormatter(_fmt_params))

    # Camera angle that shows the bowl shape (matches MATLAB default-ish view).
    ax.view_init(elev=22, azim=-58)

    # Soft panes / faint grid for the clean MATLAB-style look.
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor("#cccccc")
        pane.set_alpha(0.05)
    ax.grid(True, alpha=0.25)

    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_3d_scatter(
    all_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    compare_pareto_rows: list[dict[str, Any]] | None,
    picks: dict[str, dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    """3 objectives in 2D: x=ms, y=accuracy, marker size + viridis color = params.

    A standard 3-objective Pareto visualization (replaces the old hard-to-read
    3D rotation). Filename retained so existing notebook display cells still
    point at the right PNG.
    """
    fig, ax = plt.subplots(figsize=(11, 6.6))

    all_params = np.array([r["param_count"] for r in all_rows], dtype=float)
    p_min = max(float(all_params.min()), 1.0)
    p_max = max(float(all_params.max()), p_min + 1.0)
    log_p_min = np.log10(p_min)
    log_p_max = np.log10(p_max)
    cmap = plt.get_cmap("viridis_r")  # darker = fewer params = "better" on this axis

    def _norm_p(p: float) -> float:
        return float((np.log10(max(p, 1.0)) - log_p_min) / (log_p_max - log_p_min + 1e-12))

    def _size(p: float, base: float = 30, scale: float = 380) -> float:
        return base + scale * _norm_p(p)

    def _color(p: float):
        return cmap(_norm_p(p))

    pareto_set = {id(r) for r in pareto_rows}

    # 1. non-Pareto trials, faint
    for r in all_rows:
        if id(r) in pareto_set:
            continue
        ax.scatter(
            r["inference_ms"], r["accuracy"] * 100,
            s=_size(r["param_count"], 12, 180),
            c=[_color(r["param_count"])],
            alpha=0.28, edgecolors="none", zorder=1,
        )

    # 2. Pareto front, prominent
    for r in pareto_rows:
        ax.scatter(
            r["inference_ms"], r["accuracy"] * 100,
            s=_size(r["param_count"]),
            c=[_color(r["param_count"])],
            alpha=0.95, edgecolors="black", linewidths=1.3, zorder=3,
        )

    # 3. comparison front (diamonds, magenta edge)
    if compare_pareto_rows:
        for r in compare_pareto_rows:
            ax.scatter(
                r["inference_ms"], r["accuracy"] * 100,
                s=_size(r["param_count"]),
                c=[_color(r["param_count"])],
                marker="D", alpha=0.9, edgecolors="magenta",
                linewidths=1.5, zorder=2.5,
            )

    # 4. representative picks (stars + labels)
    if picks:
        for label, r in picks.items():
            px, py = r["inference_ms"], r["accuracy"] * 100
            ax.scatter(
                [px], [py], marker="*", s=440,
                facecolors=PICK_COLORS[label],
                edgecolors="black", linewidths=1.5, zorder=5,
            )
            ax.annotate(
                label, xy=(px, py), xytext=(11, 11),
                textcoords="offset points", fontsize=9, fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white",
                    ec=PICK_COLORS[label], alpha=0.95,
                ),
                zorder=6,
            )

    ax.set_xlabel("Inference time (ms, CPU, batch=1)   →   lower is better")
    ax.set_ylabel("Accuracy (%)   ↑   higher is better")
    ax.set_xscale("log")
    ax.grid(True, which="major", alpha=0.3, linestyle="--")
    ax.set_title(
        f"{title}\nMarker size & color encode parameter count (smaller / darker = fewer params)",
        fontsize=11,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=LogNorm(vmin=p_min, vmax=p_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Parameters")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor="none", markersize=7, alpha=0.4, label="All trials"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor="black", markersize=10, label="Pareto front"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=14, label="Representative picks"),
    ]
    if compare_pareto_rows:
        legend_handles.insert(
            -1,
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#888",
                   markeredgecolor="magenta", markersize=10, label="Comparison Pareto"),
        )
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def analyze(study_dir: Path, compare_pareto_path: Path | None = None) -> dict[str, Any]:
    trials_csv = study_dir / "trials.csv"
    if not trials_csv.exists():
        raise FileNotFoundError(f"trials.csv not found in {study_dir}")

    rows = _load_trials(trials_csv)
    pts = _objective_matrix(rows)

    pareto_rows_all = _pareto_rows(rows)
    pareto_pts = _objective_matrix(pareto_rows_all)

    hv = hypervolume_3d(pareto_pts, DEFAULT_REFERENCE_POINT)
    spacing = spacing_metric(pareto_pts)

    metrics: dict[str, Any] = {
        "study_dir": str(study_dir),
        "n_trials": len(rows),
        "n_pareto_points": len(pareto_rows_all),
        "reference_point": {
            "accuracy": DEFAULT_REFERENCE_POINT[0],
            "inference_ms": DEFAULT_REFERENCE_POINT[1],
            "param_count": DEFAULT_REFERENCE_POINT[2],
        },
        "hypervolume": float(hv),
        "spacing": float(spacing),
        "extremes": {
            "accuracy_max": float(pts[:, 0].max()),
            "inference_ms_min": float(pts[:, 1].min()),
            "param_count_min": int(pts[:, 2].min()),
        },
    }

    compare_rows: list[dict[str, Any]] | None = None
    if compare_pareto_path is not None and compare_pareto_path.exists():
        compare_rows = _load_trials(compare_pareto_path)
        compare_pts = _objective_matrix(compare_rows)
        # Joint reference: union -> non-dominated.
        joint = np.vstack([pareto_pts, compare_pts])
        joint_ref = pareto_front(joint)
        gd_ours = generational_distance(pareto_pts, joint_ref)
        gd_other = generational_distance(compare_pts, joint_ref)
        joint_set = {tuple(p) for p in joint_ref}
        ours_survive = sum(1 for p in pareto_pts if tuple(p) in joint_set)
        other_survive = sum(1 for p in compare_pts if tuple(p) in joint_set)
        metrics["comparison"] = {
            "compare_pareto_csv": str(compare_pareto_path),
            "compare_n_pareto": len(compare_rows),
            "joint_n_pareto": int(len(joint_ref)),
            "ours_surviving_in_joint": ours_survive,
            "other_surviving_in_joint": other_survive,
            "gd_ours_to_joint": float(gd_ours),
            "gd_other_to_joint": float(gd_other),
        }

    # Representative picks + table.
    picks = _representative_picks(pareto_rows_all)
    table_path = study_dir / "pareto_table.csv"
    _write_pareto_table(picks, table_path)
    metrics["pareto_table_csv"] = str(table_path)

    # Appendix solution = balanced pick (closest to ideal).
    if "balanced" in picks:
        appendix = {
            "label": "balanced (closest-to-ideal Pareto point)",
            "objectives": {
                "accuracy": picks["balanced"]["accuracy"],
                "inference_ms": picks["balanced"]["inference_ms"],
                "param_count": int(picks["balanced"]["param_count"]),
            },
            "decision_variables": {k: picks["balanced"].get(k) for k in SEARCH_SPACE_KEYS},
            "trial_number": picks["balanced"].get("trial_number"),
            "seed": picks["balanced"].get("seed"),
        }
        appendix_path = study_dir / "appendix_solution.json"
        with appendix_path.open("w", encoding="utf-8") as f:
            json.dump(appendix, f, indent=2)
        metrics["appendix_json"] = str(appendix_path)

    # Plots. Title pulls framework / algorithm from summary.json so the analyzer
    # works on either pymoo or BoTorch studies without branching.
    framework_label = "MOO study"
    summary_path_in = study_dir / "summary.json"
    if summary_path_in.exists():
        try:
            with summary_path_in.open("r", encoding="utf-8") as f:
                _summ = json.load(f)
            fw = str(_summ.get("framework", "")).strip()
            algo = str(_summ.get("algorithm", "")).strip()
            framework_label = " ".join(s for s in [fw, algo] if s) or framework_label
        except Exception:
            pass
    title = f"{framework_label}  |  n_trials={len(rows)}  |  n_pareto={len(pareto_rows_all)}"
    _plot_2d_panels(rows, pareto_rows_all, compare_rows, picks, study_dir / "plot_2d_panels.png", title)
    _plot_3d_scatter(rows, pareto_rows_all, compare_rows, picks, study_dir / "plot_3d_scatter.png", title)
    _plot_3d_pareto(rows, pareto_rows_all, compare_rows, picks, study_dir / "plot_3d_pareto.png", title)

    metrics_path = study_dir / "pareto_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    metrics["metrics_json"] = str(metrics_path)
    return metrics


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze a pymoo MOML study")
    ap.add_argument("--study-dir", required=True, type=Path)
    ap.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional path to another framework's pareto_front.csv (e.g. optuna).",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = analyze(args.study_dir, args.compare)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
