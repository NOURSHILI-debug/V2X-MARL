"""Held-out evaluation and comparison for final QMIX, VDN, and MAPPO runs.

The evaluator loads only final ``resume.pt`` checkpoints, runs deterministic
(greedy) policies on the fixed held-out seeds, and writes one episode-level CSV
plus bootstrap summaries.  It never trains or modifies an environment or
training script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from marl.mappo import MAPPO
from marl.qmix import QMIX
from marl.vdn import VDN
from train_common import (
    any_flag,
    bootstrap_ci,
    make_env,
    random_policy,
    stack_masks,
    stack_observations,
)

EPISODE_FIELDS = (
    "algorithm",
    "seed",
    "test_seed",
    "return",
    "collision",
    "length",
    "mean_speed",
    "min_ttc",
    "jerk",
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
)
N_ACTIONS = 6
PRIMARY_SEEDS = (0, 1, 2)
MAPPO_SEEDS = (0, 1, 2, 3, 4)
LEARNER_SEEDS = {
    "qmix": PRIMARY_SEEDS,
    "vdn": PRIMARY_SEEDS,
    "mappo": MAPPO_SEEDS,
}
DARWIN = os.name == "nt"
TRAIN_METRICS = ("return", "collision", "length", "mean_speed", "min_ttc", "jerk")
SUMMARY_METRICS = TRAIN_METRICS + ("a0_share",)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _as_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _test_seeds(metadata: dict[str, Any], cli_seeds: list[int]) -> list[int]:
    if cli_seeds:
        return [int(seed) for seed in cli_seeds]
    values = metadata.get("test_seeds", [])
    if isinstance(values, list) and values:
        return [int(seed) for seed in values]
    return list(range(9001, 9031))


def _env_config(
    metadata: dict[str, Any],
    obs_layout: str | None,
    background_vehicles: int | None,
    duration: float | None,
) -> tuple[str, int, float]:
    return (
        obs_layout or str(metadata.get("obs_layout", "contract56")),
        int(
            background_vehicles
            if background_vehicles is not None
            else metadata.get("background_vehicles", 12)
        ),
        float(
            duration
            if duration is not None
            else metadata.get("duration", 60.0)
        ),
    )


def _episode_action_total(row: dict[str, Any]) -> float:
    return sum(_as_float(row.get(f"a{index}", 0.0)) for index in range(N_ACTIONS))


def _action_share(row: dict[str, Any], index: int = 0) -> float:
    total = _episode_action_total(row)
    if total <= 0.0:
        return math.nan
    return _as_float(row.get(f"a{index}", 0.0)) / total


def _metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == "a0_share":
        return _action_share(row, 0)
    return _as_float(row.get(metric))


def _load_learner(task: dict[str, Any]) -> Any:
    path = Path(task["checkpoint"])
    algorithm = str(task["algorithm"])
    device = str(task["device"])
    if algorithm == "qmix":
        return QMIX.from_checkpoint(path, device=device)
    if algorithm == "vdn":
        return VDN.from_checkpoint(path, device=device)
    if algorithm == "mappo":
        return MAPPO.from_checkpoint(path, device=device)
    raise ValueError(f"Unknown learner: {algorithm}")


def _learner_act(learner: Any, algorithm: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if algorithm == "mappo":
        return lambda obs, masks: learner.act(obs, masks, deterministic=True)
    return lambda obs, masks: learner.act(obs, masks, epsilon=0.0)


def _run_episode(
    env: Any,
    act_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    test_seed: int,
) -> dict[str, Any]:
    agents = tuple(env.agent_ids)
    observations_dict, _ = env.reset(seed=test_seed)
    observations = stack_observations(observations_dict, agents)
    masks = stack_masks(env)
    team_return = 0.0
    speed_samples: list[float] = []
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    collision = False
    step_count = 0
    final_info: dict[str, Any] = {}

    while True:
        actions = np.asarray(act_fn(observations, masks), dtype=np.int64)
        if actions.shape != (len(agents),):
            raise ValueError(f"Policy returned actions with shape {actions.shape}.")
        next_observations, _rewards, terminated, truncated, info = env.step(
            {
                agent: int(actions[index])
                for index, agent in enumerate(agents)
            }
        )
        final_info = info
        team_return += float(info["team_reward"])
        speed_samples.append(
            float(
                np.mean(
                    [
                        float(info["per_agent"][agent]["speed_mps"])
                        for agent in agents
                    ]
                )
            )
        )
        action_counts += np.bincount(actions, minlength=N_ACTIONS)[:N_ACTIONS]
        collision = collision or bool(info["collision"])
        step_count += 1
        observations = stack_observations(next_observations, agents)
        masks = stack_masks(env)
        if any_flag(terminated) or any_flag(truncated):
            break

    row: dict[str, Any] = {
        "algorithm": "",
        "seed": 0,
        "test_seed": int(test_seed),
        "return": float(team_return),
        "collision": int(bool(collision)),
        "length": int(step_count),
        "mean_speed": float(np.mean(speed_samples)) if speed_samples else 0.0,
        "min_ttc": _as_float(final_info.get("min_ttc", math.nan)),
        "jerk": _as_float(final_info.get("jerk", math.nan)),
    }
    for index, count in enumerate(action_counts):
        row[f"a{index}"] = int(count)
    return row


def _policy_task(
    task: dict[str, Any],
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(int(task.get("torch_threads", 1)))
    if bool(task.get("deterministic_algorithms", True)):
        torch.use_deterministic_algorithms(True)

    env = make_env(
        str(task["obs_layout"]),
        int(task["background_vehicles"]),
        float(task["duration"]),
    )
    try:
        kind = str(task["kind"])
        algorithm = str(task["algorithm"])
        seed = int(task["seed"])
        learner = _load_learner(task) if kind == "learner" else None
        learner_act = _learner_act(learner, algorithm) if learner is not None else None
        rows: list[dict[str, Any]] = []
        step_total = 0
        started = time.perf_counter()
        for test_seed in task["test_seeds"]:
            if kind == "learner":
                act_fn = learner_act
            elif algorithm == "random":
                rng = np.random.default_rng(10_000 + int(test_seed))
                act_fn = lambda obs, masks, rng=rng: random_policy(obs, masks, rng)
            elif algorithm == "keep_speed":
                act_fn = lambda obs, masks: np.zeros(len(obs), dtype=np.int64)
            elif algorithm == "accelerate":
                act_fn = lambda obs, masks: np.ones(len(obs), dtype=np.int64)
            else:
                raise ValueError(f"Unknown reference policy: {algorithm}")
            row = _run_episode(env, act_fn, int(test_seed))
            row["algorithm"] = algorithm
            row["seed"] = seed
            rows.append(row)
            step_total += int(row["length"])
        return {
            "task": task,
            "rows": rows,
            "elapsed_s": time.perf_counter() - started,
            "episodes": len(rows),
            "steps": step_total,
        }
    finally:
        env.close()


def _write_episode_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(EPISODE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _append_episode_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(EPISODE_FIELDS))
        writer.writerows(rows)


def _make_tasks(
    args: argparse.Namespace,
    test_seeds: list[int],
    obs_layout: str,
    background_vehicles: int,
    duration: float,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for algorithm in args.algorithms:
        seeds = args.train_seeds or list(LEARNER_SEEDS[algorithm])
        for seed in seeds:
            checkpoint = (
                Path(args.runs)
                / f"{algorithm}_seed{int(seed)}"
                / "resume.pt"
            )
            tasks.append(
                {
                    "kind": "learner",
                    "algorithm": algorithm,
                    "seed": int(seed),
                    "checkpoint": str(checkpoint),
                    "test_seeds": test_seeds,
                    "obs_layout": obs_layout,
                    "background_vehicles": background_vehicles,
                    "duration": duration,
                    "device": args.device,
                    "torch_threads": args.torch_threads,
                    "deterministic_algorithms": not args.allow_nondeterministic,
                }
            )
    for reference in args.references:
        tasks.append(
            {
                "kind": "reference",
                "algorithm": reference,
                "seed": 0,
                "test_seeds": test_seeds,
                "obs_layout": obs_layout,
                "background_vehicles": background_vehicles,
                "duration": duration,
                "device": args.device,
                "torch_threads": args.torch_threads,
                "deterministic_algorithms": not args.allow_nondeterministic,
            }
        )
    return tasks


def _run_tasks(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> Path:
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    episode_path = output / "heldout_episodes.csv"
    _write_episode_csv(episode_path, [])
    print(f"eval: writing {episode_path}", flush=True)
    print(
        f"eval: tasks={len(tasks)} seeds={len(tasks[0]['test_seeds'])} "
        f"jobs={args.jobs}",
        flush=True,
    )
    started = time.perf_counter()
    completed = 0
    if args.jobs <= 1:
        for task in tasks:
            result = _policy_task(task)
            _append_episode_rows(episode_path, result["rows"])
            completed += 1
            print(
                f"done {task['algorithm']} seed={task['seed']} "
                f"episodes={result['episodes']} steps={result['steps']} "
                f"wall={result['elapsed_s']:.1f}s "
                f"total_wall={time.perf_counter() - started:.1f}s "
                f"({completed}/{len(tasks)})",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_policy_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                result = future.result()
                _append_episode_rows(episode_path, result["rows"])
                completed += 1
                print(
                    f"done {task['algorithm']} seed={task['seed']} "
                    f"episodes={result['episodes']} steps={result['steps']} "
                    f"wall={result['elapsed_s']:.1f}s "
                    f"total_wall={time.perf_counter() - started:.1f}s "
                    f"({completed}/{len(tasks)})",
                    flush=True,
                )
    print(
        f"eval: complete wall={time.perf_counter() - started:.1f}s "
        f"rows={sum(1 for _ in episode_path.open(encoding='utf-8')) - 1}",
        flush=True,
    )
    return episode_path


def _load_episode_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            row: dict[str, Any] = dict(raw)
            row["seed"] = int(row["seed"])
            row["test_seed"] = int(row["test_seed"])
            for key in EPISODE_FIELDS:
                if key not in ("algorithm", "seed", "test_seed"):
                    row[key] = _as_float(row.get(key))
            rows.append(row)
    return rows


def _rows_for(
    rows: list[dict[str, Any]],
    algorithm: str,
    seeds: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row for row in rows if str(row["algorithm"]) == algorithm
    ]
    if seeds is None:
        return selected
    allowed = {int(seed) for seed in seeds}
    return [row for row in selected if int(row["seed"]) in allowed]


def _seed_means(rows: list[dict[str, Any]], metric: str) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        value = _metric_value(row, metric)
        if math.isfinite(value):
            grouped.setdefault(int(row["seed"]), []).append(value)
    return [
        float(np.mean(values))
        for _seed, values in sorted(grouped.items())
        if values
    ]


def _values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [
        value
        for value in (_metric_value(row, metric) for row in rows)
        if math.isfinite(value)
    ]


def _mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(data)) if data else math.nan


def _bootstrap(values: list[float], n_boot: int) -> tuple[float, float, float]:
    return bootstrap_ci(values, n_boot=n_boot, seed=0)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_specs(available_algorithms: set[str]) -> list[tuple[str, str, list[int]]]:
    specs: list[tuple[str, str, list[int]]] = []
    for algorithm in ("qmix", "vdn", "mappo"):
        if algorithm in available_algorithms:
            specs.append((algorithm, algorithm, list(PRIMARY_SEEDS)))
    if "mappo" in available_algorithms and any(
        seed in MAPPO_SEEDS for seed in MAPPO_SEEDS[3:]
    ):
        specs.append(("mappo_extra", "mappo", list(MAPPO_SEEDS[3:])))
    for reference in ("random", "keep_speed", "accelerate"):
        if reference in available_algorithms:
            specs.append((reference, reference, [0]))
    return specs


def _write_summary(
    rows: list[dict[str, Any]],
    output: Path,
    n_boot: int,
) -> list[dict[str, Any]]:
    available = {str(row["algorithm"]) for row in rows}
    summary_rows: list[dict[str, Any]] = []
    for group, algorithm, seeds in _group_specs(available):
        selected = _rows_for(rows, algorithm, seeds)
        if not selected:
            continue
        unique_seeds = sorted({int(row["seed"]) for row in selected})
        for metric in SUMMARY_METRICS:
            seed_values = _seed_means(selected, metric)
            mean, low, high = _bootstrap(seed_values, n_boot)
            summary_rows.append(
                {
                    "group": group,
                    "algorithm": algorithm,
                    "seeds": ";".join(str(seed) for seed in unique_seeds),
                    "n_seeds": len(unique_seeds),
                    "n_episodes": len(selected),
                    "metric": metric,
                    "level": "seed",
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
            episode_values = _values(selected, metric)
            mean, low, high = _bootstrap(episode_values, n_boot)
            summary_rows.append(
                {
                    "group": group,
                    "algorithm": algorithm,
                    "seeds": ";".join(str(seed) for seed in unique_seeds),
                    "n_seeds": len(unique_seeds),
                    "n_episodes": len(selected),
                    "metric": metric,
                    "level": "episode",
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    _write_rows(output / "summary.csv", summary_rows)
    return summary_rows


def _write_paired_differences(
    rows: list[dict[str, Any]],
    output: Path,
    n_boot: int,
) -> list[dict[str, Any]]:
    paired_rows: list[dict[str, Any]] = []
    for left, right in (
        ("qmix", "vdn"),
        ("mappo", "vdn"),
        ("mappo", "qmix"),
    ):
        for metric in TRAIN_METRICS:
            differences: list[float] = []
            for seed in PRIMARY_SEEDS:
                left_rows = _rows_for(rows, left, [seed])
                right_rows = _rows_for(rows, right, [seed])
                if not left_rows or not right_rows:
                    continue
                differences.append(
                    _mean(_values(left_rows, metric))
                    - _mean(_values(right_rows, metric))
                )
            if not differences:
                continue
            mean, low, high = _bootstrap(differences, n_boot)
            contains_zero = bool(low <= 0.0 <= high)
            paired_rows.append(
                {
                    "metric": metric,
                    "comparison": f"{left}-{right}",
                    "n_seeds": len(differences),
                    "mean_diff": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "contains_zero": int(contains_zero),
                    "conclusion": (
                        "no demonstrated difference"
                        if contains_zero
                        else "difference"
                    ),
                }
            )
    _write_rows(output / "paired_differences.csv", paired_rows)
    return paired_rows


def _outcome(metric: str, low: float, high: float) -> str:
    if not (math.isfinite(low) and math.isfinite(high)):
        return "unknown"
    if low > 0.0:
        return "better" if metric == "return" else "worse"
    if high < 0.0:
        return "worse" if metric == "return" else "better"
    return "tie"


def _paired_test_seed_means(
    rows: list[dict[str, Any]],
    algorithm: str,
    seeds: list[int],
) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in _rows_for(rows, algorithm, seeds):
        value = _as_float(row.get("return"))
        if math.isfinite(value):
            grouped.setdefault(int(row["test_seed"]), []).append(value)
    return {
        test_seed: float(np.mean(values))
        for test_seed, values in grouped.items()
        if values
    }


def _write_reference_comparison(
    rows: list[dict[str, Any]],
    output: Path,
    n_boot: int,
) -> list[dict[str, Any]]:
    reference_rows: list[dict[str, Any]] = []
    references = [name for name in ("random", "keep_speed", "accelerate") if _rows_for(rows, name)]
    for group, algorithm, seeds in _group_specs(
        {str(row["algorithm"]) for row in rows}
    ):
        if group not in ("qmix", "vdn", "mappo"):
            continue
        for reference in references:
            for metric in ("return", "collision"):
                learner_by_test: dict[int, list[float]] = {}
                for row in _rows_for(rows, algorithm, seeds):
                    value = _as_float(row.get(metric))
                    if math.isfinite(value):
                        learner_by_test.setdefault(int(row["test_seed"]), []).append(value)
                reference_by_test: dict[int, float] = {}
                for row in _rows_for(rows, reference, [0]):
                    value = _as_float(row.get(metric))
                    if math.isfinite(value):
                        reference_by_test[int(row["test_seed"])] = value
                common = sorted(set(learner_by_test) & set(reference_by_test))
                differences = [
                    float(np.mean(learner_by_test[test_seed]))
                    - reference_by_test[test_seed]
                    for test_seed in common
                ]
                if not differences:
                    continue
                learner_mean = _mean(
                    [
                        float(np.mean(learner_by_test[test_seed]))
                        for test_seed in common
                    ]
                )
                reference_mean = _mean(
                    [reference_by_test[test_seed] for test_seed in common]
                )
                mean, low, high = _bootstrap(differences, n_boot)
                reference_rows.append(
                    {
                        "group": group,
                        "reference": reference,
                        "metric": metric,
                        "n_test_seeds": len(common),
                        "learner_mean": learner_mean,
                        "reference_mean": reference_mean,
                        "mean_diff": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "outcome": _outcome(metric, low, high),
                    }
                )
    _write_rows(output / "reference_comparison.csv", reference_rows)
    return reference_rows


def _write_collapse_check(
    rows: list[dict[str, Any]],
    output: Path,
    n_boot: int,
) -> list[dict[str, Any]]:
    keep_rows = _rows_for(rows, "keep_speed", [0])
    if not keep_rows:
        return []
    keep_by_test = {
        int(row["test_seed"]): _as_float(row.get("return"))
        for row in keep_rows
    }
    collapse_rows: list[dict[str, Any]] = []
    for algorithm in ("qmix", "vdn", "mappo"):
        seeds = sorted(
            {
                int(row["seed"])
                for row in rows
                if str(row["algorithm"]) == algorithm
            }
        )
        for seed in seeds:
            selected = _rows_for(rows, algorithm, [seed])
            action_share = _mean(
                [_action_share(row, 0) for row in selected]
            )
            return_by_test = {
                int(row["test_seed"]): _as_float(row.get("return"))
                for row in selected
            }
            common = sorted(set(return_by_test) & set(keep_by_test))
            differences = [
                return_by_test[test_seed] - keep_by_test[test_seed]
                for test_seed in common
            ]
            mean, low, high = _bootstrap(differences, n_boot)
            collapsed = bool(
                action_share >= 0.9
                and math.isfinite(low)
                and math.isfinite(high)
                and low <= 0.0 <= high
            )
            collapse_rows.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "a0_share": action_share,
                    "return_mean": _mean(return_by_test.values()),
                    "keep_speed_return": _mean(
                        keep_by_test[test_seed] for test_seed in common
                    ),
                    "return_diff": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "collapsed_to_keep_speed": int(collapsed),
                }
            )
    _write_rows(output / "collapse_check.csv", collapse_rows)
    return collapse_rows


def _print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("held-out summary (seed-level mean [95% CI]):")
    for group in ("qmix", "vdn", "mappo", "mappo_extra", "random", "keep_speed", "accelerate"):
        for metric in ("return", "collision", "length", "a0_share"):
            row = next(
                (
                    item
                    for item in summary_rows
                    if item["group"] == group
                    and item["metric"] == metric
                    and item["level"] == "seed"
                ),
                None,
            )
            if row is None:
                continue
            print(
                f"  {group:11s} {metric:9s} "
                f"{row['mean']:9.3f} [{row['ci_low']:9.3f}, {row['ci_high']:9.3f}] "
                f"n_seeds={row['n_seeds']} n_ep={row['n_episodes']}"
            )


def _print_paired(paired_rows: list[dict[str, Any]]) -> None:
    print("paired differences over seeds 0,1,2 (mean [95% CI]):")
    for row in paired_rows:
        if row["metric"] not in ("return", "collision"):
            continue
        print(
            f"  {row['comparison']:12s} {row['metric']:9s} "
            f"{row['mean_diff']:9.3f} [{row['ci_low']:9.3f}, {row['ci_high']:9.3f}] "
            f"{row['conclusion']}"
        )


def _print_reference(reference_rows: list[dict[str, Any]]) -> None:
    print("reference comparison (paired by held-out test seed):")
    for row in reference_rows:
        if row["metric"] not in ("return", "collision"):
            continue
        print(
            f"  {row['group']:5s} vs {row['reference']:10s} {row['metric']:9s} "
            f"delta={row['mean_diff']:9.3f} "
            f"[{row['ci_low']:9.3f}, {row['ci_high']:9.3f}] {row['outcome']}"
        )


def _print_collapse(collapse_rows: list[dict[str, Any]]) -> None:
    print("keep-speed collapse check:")
    for row in collapse_rows:
        print(
            f"  {row['algorithm']:5s} seed={row['seed']} "
            f"a0_share={row['a0_share']:.3f} "
            f"return={row['return_mean']:9.3f} "
            f"diff_vs_keep={row['return_diff']:9.3f} "
            f"[{row['ci_low']:9.3f}, {row['ci_high']:9.3f}] "
            f"collapsed={bool(row['collapsed_to_keep_speed'])}"
        )


def _analyze(episode_path: Path, output: Path, n_boot: int) -> None:
    rows = _load_episode_rows(episode_path)
    if not rows:
        raise ValueError(f"No episode rows in {episode_path}.")
    summary_rows = _write_summary(rows, output, n_boot)
    paired_rows = _write_paired_differences(rows, output, n_boot)
    reference_rows = _write_reference_comparison(rows, output, n_boot)
    collapse_rows = _write_collapse_check(rows, output, n_boot)
    for name in (
        "summary.csv",
        "paired_differences.csv",
        "reference_comparison.csv",
        "collapse_check.csv",
    ):
        print(f"wrote {output / name}", flush=True)
    _print_summary(summary_rows)
    _print_paired(paired_rows)
    _print_reference(reference_rows)
    _print_collapse(collapse_rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="results/raw/comparison/runs")
    parser.add_argument("--metadata", default="results/raw/comparison/run_metadata.json")
    parser.add_argument("--out", default="results/raw/eval")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("qmix", "vdn", "mappo"),
        default=["qmix", "vdn", "mappo"],
    )
    parser.add_argument(
        "--train-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override training seeds for all requested algorithms.",
    )
    parser.add_argument("--test-seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--references",
        nargs="+",
        choices=("random", "keep_speed", "accelerate"),
        default=["random", "keep_speed", "accelerate"],
    )
    parser.add_argument("--obs-layout", default=None)
    parser.add_argument("--background-vehicles", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Reuse OUT/heldout_episodes.csv and only write summaries.",
    )
    args = parser.parse_args()
    if args.jobs <= 0 or args.torch_threads <= 0 or args.n_boot <= 0:
        parser.error("jobs, torch-threads, and n-boot must be positive.")
    return args


def main() -> None:
    args = _parse_args()
    output = Path(args.out)
    metadata = _load_json(Path(args.metadata))
    test_seeds = _test_seeds(metadata, args.test_seeds)
    obs_layout, background_vehicles, duration = _env_config(
        metadata,
        args.obs_layout,
        args.background_vehicles,
        args.duration,
    )
    print(
        f"config: obs_layout={obs_layout} background_vehicles={background_vehicles} "
        f"duration={duration} test_seeds={test_seeds[0]}..{test_seeds[-1]} "
        f"n_test_seeds={len(test_seeds)} device={args.device}",
        flush=True,
    )
    if args.analyze_only:
        episode_path = output / "heldout_episodes.csv"
        if not episode_path.exists():
            raise FileNotFoundError(episode_path)
        print(f"analyze-only: reusing {episode_path}", flush=True)
    else:
        tasks = _make_tasks(
            args,
            test_seeds,
            obs_layout,
            background_vehicles,
            duration,
        )
        episode_path = _run_tasks(args, tasks)
    _analyze(episode_path, output, args.n_boot)


if __name__ == "__main__":
    main()
