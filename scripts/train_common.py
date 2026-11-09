"""Shared utilities for Highway-Env QMIX, VDN, and MAPPO trainers."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from envs.highway_env import HighwayV2XEnv


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_env(
    obs_layout: str,
    background_vehicles: int,
    duration: float,
) -> HighwayV2XEnv:
    return HighwayV2XEnv(
        render_mode=None,
        background_vehicles=background_vehicles,
        duration=duration,
        obs_layout=obs_layout,
    )


def stack_observations(
    observations: dict[str, Any],
    agents: tuple[str, ...] | list[str],
) -> np.ndarray:
    return np.stack(
        [
            np.asarray(observations[agent], dtype=np.float32).reshape(-1)
            for agent in agents
        ]
    )


def stack_masks(env: HighwayV2XEnv) -> np.ndarray:
    masks = env.action_masks()
    return np.stack(
        [np.asarray(masks[agent], dtype=bool) for agent in env.agent_ids]
    )


def any_flag(flags: dict[str, bool] | bool) -> bool:
    return any(flags.values()) if isinstance(flags, dict) else bool(flags)


def episode_step_metrics(
    infos: dict[str, Any],
    agents: tuple[str, ...] | list[str],
) -> tuple[float, float, bool]:
    per_agent = infos["per_agent"]
    mean_speed = float(
        np.mean([float(per_agent[agent]["speed_mps"]) for agent in agents])
    )
    return float(infos["team_reward"]), mean_speed, bool(infos["collision"])


def random_policy(
    observations: np.ndarray,
    masks: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.asarray(
        [int(rng.choice(np.flatnonzero(mask))) for mask in masks],
        dtype=np.int64,
    )


def evaluate_policy(
    env: HighwayV2XEnv,
    act: Callable[[np.ndarray, np.ndarray], np.ndarray],
    seeds: list[int],
) -> dict[str, float]:
    """Run greedy evaluation on fixed seeds and aggregate episode metrics."""
    agents = tuple(env.agent_ids)
    returns: list[float] = []
    speeds: list[float] = []
    collisions: list[float] = []
    steps: list[float] = []
    for seed in seeds:
        observations, _ = env.reset(seed=seed)
        obs = stack_observations(observations, agents)
        masks = stack_masks(env)
        episode_return = 0.0
        speed_samples: list[float] = []
        collision = False
        step_count = 0
        while True:
            actions = act(obs, masks)
            observations, _, terminated, truncated, infos = env.step(
                {agent: int(actions[index]) for index, agent in enumerate(agents)}
            )
            team_reward, mean_speed, step_collision = episode_step_metrics(
                infos, agents
            )
            episode_return += team_reward
            speed_samples.append(mean_speed)
            collision = collision or step_collision
            step_count += 1
            obs = stack_observations(observations, agents)
            masks = stack_masks(env)
            if any_flag(terminated) or any_flag(truncated):
                break
        returns.append(episode_return)
        speeds.append(float(np.mean(speed_samples)))
        collisions.append(float(collision))
        steps.append(float(step_count))
    return {
        "eval_return": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_collision_rate": float(np.mean(collisions)),
        "eval_mean_speed_mps": float(np.mean(speeds)),
        "eval_episode_steps": float(np.mean(steps)),
    }


def bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(data.mean())
    if data.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(n_boot, data.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        mean,
        float(np.quantile(samples, alpha)),
        float(np.quantile(samples, 1.0 - alpha)),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_metadata(
    output: Path,
    algorithm: str,
    *,
    obs_layout: str,
    obs_dim: int,
    state_dim: int,
    n_actions: int,
    train_seeds: list[int],
    eval_seeds: list[int],
    background_vehicles: int,
    episodes: int,
    device: str,
) -> dict[str, Any]:
    metadata = {
        "algorithm": algorithm,
        "obs_layout": obs_layout,
        "obs_dim": int(obs_dim),
        "state_dim": int(state_dim),
        "n_actions": int(n_actions),
        "train_seeds": [int(seed) for seed in train_seeds],
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "background_vehicles": int(background_vehicles),
        "episodes": int(episodes),
        "device": str(device),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"run_metadata_{algorithm}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
    print(
        f"{algorithm} run: obs_layout={obs_layout} obs_dim={obs_dim} "
        f"state_dim={state_dim} n_actions={n_actions} "
        f"train_seeds={metadata['train_seeds']} eval_seeds={eval_seeds} "
        f"background_vehicles={background_vehicles}"
    )
    return metadata


def aggregate_rows(
    rows: list[dict[str, Any]],
    value_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["algorithm"]), int(row["episode"]))
        grouped.setdefault(key, []).append(row)
    aggregate: list[dict[str, Any]] = []
    for (algorithm, episode), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "algorithm": algorithm,
            "episode": episode,
            "n_seeds": len(group),
        }
        for value_key in value_keys:
            values = [
                float(row[value_key])
                for row in group
                if row.get(value_key) not in ("", None, "NA")
            ]
            if not values:
                continue
            mean, low, high = bootstrap_ci(values, seed=episode)
            item[f"{value_key}_mean"] = mean
            item[f"{value_key}_ci_low"] = low
            item[f"{value_key}_ci_high"] = high
        aggregate.append(item)
    return aggregate
