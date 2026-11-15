"""Build three-way comparison figures from training metrics and held-out episodes.

Reads per-run metrics.csv files, held-out episode rows, and training logs, then
writes preliminary training/evaluation curves, held-out per-seed summaries,
action histograms, and cost panels. Missing runs or points are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEARNERS = ("qmix", "vdn", "mappo")
REFERENCES = ("random", "keep_speed", "accelerate")
GROUPS = LEARNERS + REFERENCES
COLORS = {"qmix": "#1f77b4", "vdn": "#ff7f0e", "mappo": "#2ca02c"}
REFERENCE_COLOR = "#7f7f7f"
METRICS = (
    ("return", "Held-out return"),
    ("collision", "Held-out collision rate"),
    ("length", "Held-out episode length"),
    ("mean_speed", "Held-out mean speed"),
    ("min_ttc", "Held-out min TTC"),
    ("jerk", "Held-out jerk"),
)
N_BOOT = 2000
BOOT_SEED = 20261115
COST_RE = re.compile(
    r"^(?P<algorithm>qmix|vdn|mappo) seed=(?P<seed>\d+) "
    r"ep=\d+ total_steps=(?P<steps>\d+)/(?P<budget>\d+) "
    r".*elapsed=(?P<elapsed>[0-9.]+)s"
)


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
                if algorithm not in LEARNERS:
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


def _load_episodes(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            algorithm = str(raw.get("algorithm", ""))
            if algorithm not in GROUPS:
                continue
            seed = _as_int(raw, "seed")
            if seed is None:
                continue
            row: dict[str, object] = dict(raw)
            row["algorithm"] = algorithm
            row["seed"] = seed
            for metric, _label in METRICS:
                row[metric] = _as_float(raw, metric)
            for index in range(6):
                row[f"a{index}"] = _as_float(raw, f"a{index}")
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


def _short_seeds(values: object) -> str:
    seeds: list[int] = []
    if isinstance(values, list):
        for value in values:
            seed = _as_int({"value": value}, "value")
            if seed is not None:
                seeds.append(seed)
    seeds = sorted(set(seeds))
    if not seeds:
        return "?"
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}"
    return ",".join(str(seed) for seed in seeds)


def _title(metadata: dict[str, object], include_test: bool = False) -> str:
    layout = str(metadata.get("obs_layout", "unknown"))
    qmix_seeds = metadata.get("train_seeds", [])
    mappo_seeds = metadata.get("mappo_train_seeds") or qmix_seeds
    steps = metadata.get("total_steps", "?")
    seeds = f"train QMIX/VDN {_short_seeds(qmix_seeds)}, MAPPO {_short_seeds(mappo_seeds)}"
    if include_test:
        test_seeds = metadata.get("test_seeds", [])
        if isinstance(test_seeds, list) and test_seeds:
            seeds += f"; test {_short_seeds(test_seeds)}"
    return (
        f"PRELIMINARY | {layout} | {seeds} | {steps} train steps | "
        f"{date.today().isoformat()}"
    )


def _save(fig: plt.Figure, path: Path, title: str, dpi: int) -> Path:
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _bootstrap_ci(values: list[float], seed: int = BOOT_SEED) -> tuple[float, float, float]:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return math.nan, math.nan, math.nan
    array = np.asarray(finite, dtype=float)
    mean = float(array.mean())
    if array.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(N_BOOT, array.size))
    boot_means = array[indices].mean(axis=1)
    return (
        mean,
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def _bootstrap_band(
    curves: list[np.ndarray], seed: int = BOOT_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.vstack(curves)
    mean = array.mean(axis=0)
    if array.shape[0] < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.shape[0], size=(N_BOOT, array.shape[0]))
    boot_means = array[indices].mean(axis=1)
    return (
        mean,
        np.percentile(boot_means, 2.5, axis=0),
        np.percentile(boot_means, 97.5, axis=0),
    )


def _training_series(
    rows: list[dict[str, object]], metric: str, window: int, grid_size: int = 200
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
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

    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for algorithm in LEARNERS:
        runs = [
            value for (name, _seed), value in smoothed.items() if name == algorithm
        ]
        max_steps = [
            float(xs[-1]) for xs, _ys in runs if len(xs) and np.isfinite(xs[-1])
        ]
        if not max_steps:
            continue
        grid = np.linspace(0.0, min(max_steps), grid_size)
        curves = [np.interp(grid, xs, ys) for xs, ys in runs if len(xs) > 0]
        if not curves:
            continue
        mean, low, high = _bootstrap_band(curves)
        series[algorithm] = (grid, mean, low, high, len(curves))
    return series


def _eval_series(
    rows: list[dict[str, object]], metric: str
) -> dict[str, list[tuple[int, float, float, float, int]]]:
    per_seed: dict[tuple[str, int, int], list[float]] = {}
    for row in rows:
        if int(row["eval"]) != 1:
            continue
        value = _as_float(row, metric)
        if not np.isfinite(value):
            continue
        key = (str(row["algorithm"]), int(row["seed"]), int(row["total_steps"]))
        per_seed.setdefault(key, []).append(value)

    grouped: dict[tuple[str, int], list[float]] = {}
    for (algorithm, _seed, step), values in per_seed.items():
        grouped.setdefault((algorithm, step), []).append(float(np.mean(values)))

    series: dict[str, list[tuple[int, float, float, float, int]]] = {}
    for (algorithm, step), values in grouped.items():
        mean, low, high = _bootstrap_ci(values)
        series.setdefault(algorithm, []).append((step, mean, low, high, len(values)))
    for values in series.values():
        values.sort()
    return series


def _heldout_series(
    episodes: list[dict[str, object]]
) -> dict[str, list[dict[str, float]]]:
    series: dict[str, list[dict[str, float]]] = {}
    for group in GROUPS:
        rows = [row for row in episodes if row["algorithm"] == group]
        if not rows:
            continue
        by_seed: dict[int, list[dict[str, object]]] = {}
        for row in rows:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        entries: list[dict[str, float]] = []
        for seed, seed_rows in sorted(by_seed.items()):
            entry: dict[str, float] = {"seed": float(seed)}
            for metric, _label in METRICS:
                entry[metric] = float(
                    np.mean([float(row[metric]) for row in seed_rows])
                )
            entries.append(entry)
        series[group] = entries
    return series


def _heldout_stats(
    entries: list[dict[str, float]], metric: str
) -> tuple[list[float], float, float, float]:
    values = [entry[metric] for entry in entries]
    mean, low, high = _bootstrap_ci(values)
    return values, mean, low, high


def _plot_line_band(
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]],
    path: Path,
    ylabel: str,
    panel_title: str,
    title: str,
    dpi: int,
    ylim: tuple[float, float] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for algorithm in LEARNERS:
        if algorithm not in series:
            continue
        steps, mean, low, high, count = series[algorithm]
        ax.plot(
            steps,
            mean,
            color=COLORS[algorithm],
            label=f"{algorithm.upper()} (n={count})",
        )
        ax.fill_between(steps, low, high, color=COLORS[algorithm], alpha=0.2)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title)
    ax.set_xlim(left=0.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    return _save(fig, path, title, dpi)


def _draw_heldout_panel(
    ax: plt.Axes, series: dict[str, list[dict[str, float]]], metric: str
) -> None:
    groups = [group for group in GROUPS if group in series]
    for index, group in enumerate(groups):
        entries = series[group]
        values, mean, low, high = _heldout_stats(entries, metric)
        color = COLORS.get(group, REFERENCE_COLOR)
        count = len(values)
        offsets = np.linspace(-0.10, 0.10, count) if count > 1 else np.zeros(1)
        ax.scatter(
            index + offsets,
            values,
            color=color,
            s=28,
            alpha=0.9,
            zorder=3,
            edgecolors="none",
        )
        ax.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            color="black",
            markersize=5,
            capsize=4,
            linewidth=1.2,
            zorder=4,
        )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(
        [
            f"{group.upper().replace('_', ' ')}\nn={len(series[group])}"
            for group in groups
        ],
        fontsize=8,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def _plot_heldout_metric(
    series: dict[str, list[dict[str, float]]],
    metric: str,
    ylabel: str,
    path: Path,
    panel_title: str,
    title: str,
    dpi: int,
    ylim: tuple[float, float] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    _draw_heldout_panel(ax, series, metric)
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return _save(fig, path, title, dpi)


def _plot_heldout_summary(
    series: dict[str, list[dict[str, float]]], path: Path, title: str, dpi: int
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.2))
    for ax, (metric, label) in zip(axes.flat, METRICS):
        _draw_heldout_panel(ax, series, metric)
        ax.set_title(label, fontsize=10)
        if metric == "collision":
            ax.set_ylim(-0.05, 1.05)
    for ax in axes.flat[len(METRICS) :]:
        ax.set_visible(False)
    return _save(fig, path, title, dpi)


def _action_shares(
    episodes: list[dict[str, object]]
) -> dict[str, tuple[list[float], int]]:
    shares: dict[str, tuple[list[float], int]] = {}
    for group in GROUPS:
        rows = [row for row in episodes if row["algorithm"] == group]
        if not rows:
            continue
        counts = [sum(float(row[f"a{index}"]) for row in rows) for index in range(6)]
        total = sum(counts)
        if total <= 0.0:
            continue
        shares[group] = ([count / total for count in counts], len(rows))
    return shares


def _plot_actions(
    shares: dict[str, tuple[list[float], int]], path: Path, title: str, dpi: int
) -> Path:
    groups = [group for group in GROUPS if group in shares]
    columns = 4
    rows = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(13.0, 3.2 * rows), squeeze=False)
    for ax, group in zip(axes.flat, groups):
        values, count = shares[group]
        color = COLORS.get(group, REFERENCE_COLOR)
        positions = np.arange(6)
        ax.bar(positions, values, color=color)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"a{index}" for index in positions])
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{group.upper().replace('_', ' ')} ({count} eps)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in axes.flat[len(groups) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Action share")
    return _save(fig, path, title, dpi)


def _load_cost(root: Path) -> list[dict[str, float | int | str]]:
    costs: dict[tuple[str, int], dict[str, float | int | str]] = {}
    runs = root / "runs"
    if runs.is_dir():
        for path in sorted(runs.glob("*/timing.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(value, dict):
                continue
            algorithm = str(value.get("algorithm", ""))
            seed = _as_int(value, "seed")
            elapsed = _as_float(value, "elapsed_s")
            steps = _as_int(value, "total_steps")
            if algorithm not in LEARNERS or seed is None or steps is None:
                continue
            if not np.isfinite(elapsed) or elapsed <= 0.0:
                continue
            costs[(algorithm, seed)] = {
                "algorithm": algorithm,
                "seed": seed,
                "total_steps": steps,
                "elapsed_s": elapsed,
                "steps_per_s": steps / elapsed,
            }

    for log_path in sorted(root.parent.glob("*train.log")):
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        best: dict[tuple[str, int], tuple[int, float]] = {}
        for line in lines:
            match = COST_RE.match(line)
            if match is None:
                continue
            steps = int(match.group("steps"))
            budget = int(match.group("budget"))
            if steps < budget:
                continue
            key = (match.group("algorithm"), int(match.group("seed")))
            elapsed = float(match.group("elapsed"))
            if elapsed <= 0.0:
                continue
            if key not in best or steps > best[key][0]:
                best[key] = (steps, elapsed)
        for (algorithm, seed), (steps, elapsed) in best.items():
            if (algorithm, seed) in costs:
                continue
            costs[(algorithm, seed)] = {
                "algorithm": algorithm,
                "seed": seed,
                "total_steps": steps,
                "elapsed_s": elapsed,
                "steps_per_s": steps / elapsed,
            }
    return sorted(
        costs.values(), key=lambda item: (str(item["algorithm"]), int(item["seed"]))
    )


def _plot_cost(
    costs: list[dict[str, float | int | str]], path: Path, title: str, dpi: int
) -> Path | None:
    groups = [
        algorithm
        for algorithm in LEARNERS
        if any(item["algorithm"] == algorithm for item in costs)
    ]
    if not groups:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    panels = (
        ("elapsed_s", "Training wall-clock (s)", "{:.0f}"),
        ("steps_per_s", "Training throughput (steps/s)", "{:.1f}"),
    )
    for ax, (key, ylabel, number_format) in zip(axes, panels):
        for index, algorithm in enumerate(groups):
            runs = sorted(
                [item for item in costs if item["algorithm"] == algorithm],
                key=lambda item: int(item["seed"]),
            )
            values = [float(item[key]) for item in runs]
            offsets = (
                np.linspace(-0.12, 0.12, len(values))
                if len(values) > 1
                else np.zeros(1)
            )
            ax.bar(
                index,
                float(np.mean(values)),
                width=0.5,
                color=COLORS[algorithm],
                alpha=0.35,
            )
            ax.scatter(
                index + offsets,
                values,
                color=COLORS[algorithm],
                s=30,
                zorder=3,
                edgecolors="none",
            )
            for offset, value in zip(offsets, values):
                ax.annotate(
                    number_format.format(value),
                    (index + offset, value),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=7,
                )
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(
            [
                f"{algorithm.upper()}\nn={sum(1 for item in costs if item['algorithm'] == algorithm)}"
                for algorithm in groups
            ]
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_title("Wall-clock")
    axes[1].set_title("Throughput")
    return _save(fig, path, title, dpi)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw/comparison")
    parser.add_argument("--eval", default="results/raw/eval/heldout_episodes.csv")
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
    episodes = _load_episodes(Path(args.eval))
    metadata = _load_metadata(root)
    if not rows and not episodes:
        print(f"no metrics or held-out rows found under {root}")
        return

    written: list[Path] = []
    training_title = _title(metadata)
    test_title = _title(metadata, include_test=True)

    if rows:
        training_return = _training_series(rows, "team_return", args.window)
        if training_return:
            written.append(
                _plot_line_band(
                    training_return,
                    output / "training_return_curve.png",
                    "Episode team return (smoothed)",
                    f"Training return, rolling window {args.window} episodes",
                    training_title,
                    args.dpi,
                )
            )
        return_eval = _eval_series(rows, "team_return")
        if return_eval:
            written.append(
                _plot_line_band(
                    return_eval,
                    output / "eval_return_curve.png",
                    "Eval team return",
                    "Greedy evaluation return",
                    training_title,
                    args.dpi,
                )
            )
        collision_eval = _eval_series(rows, "collision")
        if collision_eval:
            written.append(
                _plot_line_band(
                    collision_eval,
                    output / "eval_collision_curve.png",
                    "Eval collision rate",
                    "Greedy evaluation collision rate",
                    training_title,
                    args.dpi,
                    ylim=(-0.05, 1.05),
                )
            )

    if episodes:
        heldout = _heldout_series(episodes)
        for metric, label in METRICS:
            if not any(metric in entry for entries in heldout.values() for entry in entries):
                continue
            ylim = (-0.05, 1.05) if metric == "collision" else None
            written.append(
                _plot_heldout_metric(
                    heldout,
                    metric,
                    label,
                    output / f"heldout_{metric}.png",
                    label,
                    test_title,
                    args.dpi,
                    ylim=ylim,
                )
            )
        written.append(
            _plot_heldout_summary(
                heldout,
                output / "heldout_summary.png",
                test_title,
                args.dpi,
            )
        )
        shares = _action_shares(episodes)
        if shares:
            written.append(
                _plot_actions(
                    shares,
                    output / "action_histogram.png",
                    test_title,
                    args.dpi,
                )
            )

    costs = _load_cost(root)
    cost_path = _plot_cost(costs, output / "cost.png", training_title, args.dpi)
    if cost_path is not None:
        written.append(cost_path)

    algorithms = sorted({str(row["algorithm"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    runs = sorted({(str(row["algorithm"]), int(row["seed"])) for row in rows})
    heldout_runs = sorted(
        {(str(row["algorithm"]), int(row["seed"])) for row in episodes}
    )
    print(
        f"rows={len(rows)} runs={len(runs)} algorithms={algorithms} seeds={seeds} "
        f"heldout_rows={len(episodes)} heldout_runs={len(heldout_runs)} "
        f"cost_runs={len(costs)}"
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
