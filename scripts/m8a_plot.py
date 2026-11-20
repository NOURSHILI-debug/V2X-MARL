"""Milestone 8A plotting: regenerate figures from saved JSON/CSV (no CARLA needed)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_summary_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(rows, key):
    vals = []
    for r in rows:
        v = r.get(key)
        if v in (None, "", "None"):
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            pass
    return vals


def _mean_std(vals):
    if not vals:
        return 0.0, 0.0
    arr = np.array(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _modes(rows):
    return sorted({r["selection_mode"] for r in rows})


def plot_success_termination(rows, out: Path):
    modes = _modes(rows)
    metrics = [
        ("success", "Success"),
        ("collision", "Collision"),
        ("off_road", "Off-road"),
        ("stuck", "Stuck"),
    ]
    # timeout derived from ending_reason
    data = {m: [] for m in modes}
    labels = [label for _, label in metrics] + ["Timeout"]
    for m in modes:
        mr = [r for r in rows if r["selection_mode"] == m]
        n = len(mr) or 1
        for key, _ in metrics:
            data[m].append(sum(1 for r in mr if str(r.get(key)).lower() in ("1", "true")) / n)
        data[m].append(sum(1 for r in mr if r.get("ending_reason") == "timeout") / n)

    x = np.arange(len(labels))
    width = 0.35 if len(modes) == 2 else 0.8 / max(1, len(modes))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(modes):
        offset = (i - (len(modes) - 1) / 2) * width
        ax.bar(x + offset, data[m], width, label=m)
    ax.set_ylabel("Rate")
    ax.set_title("Success and termination comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "success_termination.png", dpi=300)
    plt.close(fig)


def plot_bar_metric(rows, out: Path, key, title, ylabel, filename):
    modes = _modes(rows)
    means, stds = [], []
    for m in modes:
        mr = [r for r in rows if r["selection_mode"] == m]
        vals = _to_float(mr, key)
        mu, sd = _mean_std(vals)
        means.append(mu); stds.append(sd)
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(modes))
    ax.bar(x, means, yerr=stds, capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=300)
    plt.close(fig)


def plot_collision_rate(rows, out: Path):
    modes = _modes(rows)
    rates = []
    for m in modes:
        mr = [r for r in rows if r["selection_mode"] == m]
        n = len(mr) or 1
        rates.append(sum(1 for r in mr if str(r.get("collision")).lower() in ("1", "true")) / n)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(np.arange(len(modes)), rates)
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.05)
    ax.set_title("Collision rate comparison")
    ax.set_ylabel("Collision rate")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "collision_rate.png", dpi=300)
    plt.close(fig)


def plot_candidate_pool(rows, out: Path):
    modes = _modes(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, key, title, ylabel in [
        (axes[0], "mean_pool_size", "Mean eligible candidate pool size", "Pool size"),
        (axes[1], "mean_missing_slots", "Mean missing selection slots", "Missing slots"),
    ]:
        means, stds = [], []
        for m in modes:
            mr = [r for r in rows if r["selection_mode"] == m]
            vals = _to_float(mr, key)
            mu, sd = _mean_std(vals)
            means.append(mu); stds.append(sd)
        ax.bar(np.arange(len(modes)), means, yerr=stds, capsize=6)
        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels(modes)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "candidate_pool_statistics.png", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "milestone8a"))
    p.add_argument("--figures-dir", default=None)
    args = p.parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "episode_summary.csv"
    json_path = results_dir / "episode_summary.json"
    if csv_path.exists():
        rows = load_summary_csv(csv_path)
    elif json_path.exists():
        rows = load_json(json_path)
    else:
        raise SystemExit(f"No episode_summary.csv/json found in {results_dir}")

    if not rows:
        raise SystemExit("No episode rows to plot")

    plot_success_termination(rows, figures_dir)
    plot_bar_metric(rows, figures_dir, "team_reward_total", "Mean episode reward (total)", "Team reward", "mean_episode_reward.png")
    plot_collision_rate(rows, figures_dir)
    plot_bar_metric(rows, figures_dir, "steps", "Episode length comparison", "Steps", "episode_length.png")
    plot_candidate_pool(rows, figures_dir)
    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
