"""Plot QMIX/VDN comparison curves from per-run metrics.csv files.

Works on partial runs: missing runs, missing eval points, or single seeds are
skipped, and every figure is titled PRELIMINARY unless the metadata and all
expected runs show the full step budget and final eval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALGORITHMS = ("qmix", "vdn")
COLORS = {"qmix": "#1f77b4", "vdn": "#ff7f0e"}
T_CRITICAL = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _as_float(row: dict[str, object], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def _as_int(row: dict[str, object], key: str) -> int | None:
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return None


def _load_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    runs = root / "runs"
    if not runs.is_dir():
        return rows
    for path in sorted(runs.glob("*/metrics.csv")):
        with path.open(newline="", encoding="utf-8") as file:
            for raw in csv.DictReader(file):
                algorithm = str(raw.get("algorithm", ""))
                if algorithm not in ALGORITHMS:
                    continue
                seed = _as_int(raw, "seed")
                total_steps = _as_int(raw, "total_steps")
                evaluation = _as_int(raw, "eval")
                if seed is None or total_steps is None or evaluation is None:
                    continue
                row: dict[str, object] = dict(raw)
                row["seed"] = seed
                row["total_steps"] = total_steps
                row["eval"] = evaluation
                rows.append(row)
    return rows


def _load_metadata(root: Path) -> dict[str, object]:
    path = root / "run_metadata.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _mean_ci(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return math.nan, math.nan
    array = np.asarray(finite, dtype=float)
    mean = float(array.mean())
    if array.size < 2:
        return mean, 0.0
    standard_error = float(array.std(ddof=1)) / math.sqrt(array.size)
    critical = T_CRITICAL.get(array.size - 1, 1.96)
    return mean, critical * standard_error


def _is_complete(metadata: dict[str, object], rows: list[dict[str, object]]) -> bool:
    algorithms = [
        str(value)
        for value in metadata.get("algorithms", [])
        if str(value) in ALGORITHMS
    ]
    seeds: list[int] = []
    for value in metadata.get("train_seeds", []):
        seed = _as_int({"value": value}, "value")
        if seed is not None:
            seeds.append(seed)
    budget = _as_int(metadata, "total_steps")
    if not algorithms or not seeds or budget is None or budget <= 0:
        return False
    by_run: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        by_run.setdefault((str(row["algorithm"]), int(row["seed"])), []).append(row)
    for algorithm in algorithms:
        for seed in seeds:
            run_rows = by_run.get((algorithm, seed), [])
            if not run_rows:
                return False
            if max(int(row["total_steps"]) for row in run_rows) < budget:
                return False
            reaches_final_eval = any(
                int(row["eval"]) == 1 and int(row["total_steps"]) == budget
                for row in run_rows
            )
            if not reaches_final_eval:
                return False
    return True


def _eval_series(
    rows: list[dict[str, object]],
    metric: str,
) -> dict[str, list[tuple[int, float, float, int]]]:
    groups: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        if int(row["eval"]) != 1:
            continue
        value = _as_float(row, metric)
        if not np.isfinite(value):
            continue
        key = (str(row["algorithm"]), int(row["total_steps"]))
        groups.setdefault(key, []).append(value)
    series: dict[str, list[tuple[int, float, float, int]]] = {}
    for (algorithm, step), values in groups.items():
        mean, ci = _mean_ci(values)
        series.setdefault(algorithm, []).append((step, mean, ci, len(values)))
    for algorithm in series:
        series[algorithm].sort()
    return series


def _training_series(
    rows: list[dict[str, object]],
    metric: str,
    window: int,
    grid_size: int = 200,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    per_run: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for row in rows:
        if int(row["eval"]) != 0:
            continue
        value = _as_float(row, metric)
        if not np.isfinite(value):
            continue
        key = (str(row["algorithm"]), int(row["seed"]))
        per_run.setdefault(key, []).append((int(row["total_steps"]), value))

    smoothed: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, points in per_run.items():
        points.sort()
        xs = np.asarray([step for step, _ in points], dtype=float)
        ys = np.asarray([value for _, value in points], dtype=float)
        width = min(window, len(ys))
        if width <= 0:
            continue
        kernel = np.ones(width) / width
        smoothed[key] = (xs[width - 1 :], np.convolve(ys, kernel, mode="valid"))

    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for algorithm in ALGORITHMS:
        runs = [
            value for (name, _seed), value in smoothed.items() if name == algorithm
        ]
        max_steps = [
            float(xs[-1]) for xs, _ys in runs if len(xs) and np.isfinite(xs[-1])
        ]
        if not max_steps:
            continue
        grid = np.linspace(0.0, min(max_steps), grid_size)
        curves = [
            np.interp(grid, xs, ys)
            for xs, ys in runs
            if len(xs) > 0
        ]
        if not curves:
            continue
        array = np.vstack(curves)
        means = array.mean(axis=0)
        cis = np.asarray(
            [_mean_ci(array[:, index].tolist())[1] for index in range(array.shape[1])]
        )
        series[algorithm] = (grid, means, cis, len(curves))
    return series


def _save_figure(fig: plt.Figure, path: Path, title: str, dpi: int) -> Path:
    if title.strip():
        fig.suptitle(title.strip(), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_eval_curve(
    series: dict[str, list[tuple[int, float, float, int]]],
    path: Path,
    ylabel: str,
    title: str,
    prefix: str,
    dpi: int,
    ylim: tuple[float, float] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for algorithm in ALGORITHMS:
        if algorithm not in series:
            continue
        data = np.asarray(series[algorithm], dtype=float)
        label = f"{algorithm.upper()} (n={int(data[0, 3])})"
        ax.plot(
            data[:, 0],
            data[:, 1],
            marker="o",
            color=COLORS[algorithm],
            label=label,
        )
        ax.fill_between(
            data[:, 0],
            data[:, 1] - data[:, 2],
            data[:, 1] + data[:, 2],
            color=COLORS[algorithm],
            alpha=0.2,
        )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(prefix + title)
    ax.set_xlim(left=0.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    return _save_figure(fig, path, "", dpi)


def _plot_training_curve(
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    path: Path,
    title: str,
    prefix: str,
    dpi: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for algorithm in ALGORITHMS:
        if algorithm not in series:
            continue
        steps, means, cis, count = series[algorithm]
        ax.plot(
            steps,
            means,
            color=COLORS[algorithm],
            label=f"{algorithm.upper()} (n={count})",
        )
        ax.fill_between(
            steps,
            means - cis,
            means + cis,
            color=COLORS[algorithm],
            alpha=0.2,
        )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode team return (smoothed)")
    ax.set_title(prefix + title)
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.3)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    return _save_figure(fig, path, "", dpi)


def _plot_final_bars(
    return_series: dict[str, list[tuple[int, float, float, int]]],
    collision_series: dict[str, list[tuple[int, float, float, int]]],
    path: Path,
    prefix: str,
    dpi: int,
) -> Path | None:
    step_candidates = [
        int(values[-1][0])
        for values in return_series.values()
        if values
    ]
    if not step_candidates:
        return None
    final_step = min(step_candidates)
    panels = (
        (return_series, "Final eval team return", None),
        (collision_series, "Final eval collision rate", (-0.05, 1.05)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    plotted = False
    for ax, (series, title, ylim) in zip(axes, panels):
        labels: list[str] = []
        means: list[float] = []
        cis: list[float] = []
        colors: list[str] = []
        for algorithm in ALGORITHMS:
            entry = next(
                (
                    value
                    for value in series.get(algorithm, [])
                    if int(value[0]) == final_step
                ),
                None,
            )
            if entry is None:
                continue
            labels.append(algorithm.upper())
            means.append(entry[1])
            cis.append(entry[2])
            colors.append(COLORS[algorithm])
        if not labels:
            ax.set_visible(False)
            continue
        plotted = True
        positions = np.arange(len(labels))
        ax.bar(positions, means, yerr=cis, capsize=4, color=colors)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
    if not plotted:
        plt.close(fig)
        return None
    title = f"{prefix}Final evaluation at {final_step} steps"
    return _save_figure(fig, path, title, dpi)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw/comparison")
    parser.add_argument("--output", default="results/raw/plots")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    if args.window <= 0 or args.dpi <= 0:
        parser.error("window and dpi must be positive.")
    return args


def main() -> None:
    args = _parse_args()
    root = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(root)
    if not rows:
        print(f"no metrics rows found under {root}")
        return
    metadata = _load_metadata(root)
    complete = _is_complete(metadata, rows)
    prefix = "" if complete else "PRELIMINARY - "

    written: list[Path] = []
    return_eval = _eval_series(rows, "team_return")
    if return_eval:
        written.append(
            _plot_eval_curve(
                return_eval,
                output / "eval_return_curve.png",
                "Eval team return",
                "Greedy evaluation",
                prefix,
                args.dpi,
            )
        )
    collision_eval = _eval_series(rows, "collision")
    if collision_eval:
        written.append(
            _plot_eval_curve(
                collision_eval,
                output / "eval_collision_curve.png",
                "Eval collision rate",
                "Greedy evaluation",
                prefix,
                args.dpi,
                ylim=(-0.05, 1.05),
            )
        )
    training_return = _training_series(rows, "team_return", args.window)
    if training_return:
        written.append(
            _plot_training_curve(
                training_return,
                output / "training_return_curve.png",
                f"Training return (rolling window {args.window} episodes)",
                prefix,
                args.dpi,
            )
        )
    final_bars = _plot_final_bars(
        return_eval,
        collision_eval,
        output / "final_eval_bars.png",
        prefix,
        args.dpi,
    )
    if final_bars is not None:
        written.append(final_bars)

    algorithms = sorted({str(row["algorithm"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    runs = sorted({(str(row["algorithm"]), int(row["seed"])) for row in rows})
    status = "complete" if complete else "PRELIMINARY (partial data)"
    print(
        f"{status}: rows={len(rows)} runs={len(runs)} "
        f"algorithms={algorithms} seeds={seeds}"
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
