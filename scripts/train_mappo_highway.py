"""Train MAPPO on the same seeded Highway-Env contract56 episodes as QMIX/VDN.

The environment configuration, seeds, evaluation cadence, step budget, and CSV
schema match ``scripts/train_qmix_highway.py`` so the results are directly
comparable. Writes one CSV row per training episode and per evaluation episode
under ``results/raw/`` (gitignored), plus a resumable learner/rollout
checkpoint. No CARLA, no toy env.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.highway_env import HighwayV2XEnv
from marl.mappo import MAPPO, RolloutBuffer
from train_common import (
    any_flag,
    make_env,
    random_policy,
    seed_everything,
    stack_masks,
    stack_observations,
)

CSV_FIELDS = [
    "algorithm",
    "seed",
    "episode",
    "total_steps",
    "team_return",
    "collision",
    "mean_speed",
    "steps",
    "epsilon",
    "loss",
    "eval",
    "env_seed",
]


def _episode_seed(run_seed: int, episode: int) -> int:
    return 100_000 + run_seed * 1_000_000 + episode


def _episode_metrics(
    team_return: float,
    collision: bool,
    speeds: list[float],
    steps: int,
) -> dict[str, Any]:
    return {
        "team_return": float(team_return),
        "collision": int(bool(collision)),
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "steps": int(steps),
    }


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        file.flush()


def _truncate_metrics_by_step(path: Path, keep_through_step: int) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    kept = [
        row
        for row in rows
        if str(row.get("total_steps", "")).isdigit()
        and int(row["total_steps"]) <= keep_through_step
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def _clear_metrics(path: Path) -> None:
    if path.exists():
        path.unlink()


def _eval_episode(
    env: HighwayV2XEnv,
    act_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    reset_seed: int,
) -> dict[str, Any]:
    observations_dict, _ = env.reset(seed=reset_seed)
    agents = env.agent_ids
    observations = stack_observations(observations_dict, agents)
    masks = stack_masks(env)
    team_return = 0.0
    speeds: list[float] = []
    collision = False
    steps = 0

    while True:
        actions = act_fn(observations, masks)
        next_obs_dict, _rewards, terminated, truncated, info = env.step(
            {agent: int(actions[index]) for index, agent in enumerate(agents)}
        )
        observations = stack_observations(next_obs_dict, agents)
        masks = stack_masks(env)
        team_return += float(info["team_reward"])
        speeds.extend(
            float(info["per_agent"][agent]["speed_mps"]) for agent in agents
        )
        collision = collision or bool(info["collision"])
        steps += 1
        if any_flag(terminated) or any_flag(truncated):
            break

    return _episode_metrics(team_return, collision, speeds, steps)


def _build_learner(
    args: argparse.Namespace,
    obs_dim: int,
    state_dim: int,
    n_actions: int,
    seed: int,
) -> tuple[MAPPO, RolloutBuffer]:
    n_agents = 4
    learner = MAPPO(
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        value_clip=args.value_clip,
        max_grad_norm=args.max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        reward_scale=args.reward_scale,
        state_norm=args.state_norm,
        state_mean=args.state_mean,
        state_std=args.state_std,
        device=args.device,
        seed=seed,
    )
    rollout = RolloutBuffer(
        args.rollout_steps,
        n_agents,
        obs_dim,
        state_dim,
        n_actions,
    )
    return learner, rollout


def _run_training(
    run_seed: int,
    args: argparse.Namespace,
    eval_seeds: list[int],
    output: Path,
) -> None:
    seed_everything(run_seed)
    run_dir = output / "runs" / f"mappo_seed{run_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    resume_path = run_dir / "resume.pt"
    rollout_path = run_dir / "rollout.npz"
    state_path = run_dir / "state.json"
    env = make_env(args.obs_layout, args.background_vehicles, args.duration)
    eval_env = make_env(args.obs_layout, args.background_vehicles, args.duration)
    try:
        agents = env.agent_ids
        env.reset(seed=run_seed * 100_000)
        obs_dim = int(env.obs_dim)
        state_dim = int(env.state().size)
        n_actions = int(env.n_actions)
        if args.obs_layout == "contract56":
            assert obs_dim == 56, f"contract56 must be 56-D, got {obs_dim}"

        learner, rollout = _build_learner(args, obs_dim, state_dim, n_actions, run_seed)
        total_steps = 0
        episode = 0
        updates = 0
        if args.resume and resume_path.exists() and state_path.exists():
            learner = MAPPO.from_checkpoint(resume_path, device=args.device)
            if rollout_path.exists():
                rollout.load(rollout_path)
            resume_state = json.loads(state_path.read_text(encoding="utf-8"))
            if "next_step" not in resume_state:
                raise ValueError("resume state lacks next_step; rerun without --resume")
            total_steps = int(resume_state["next_step"])
            episode = int(resume_state["next_episode"])
            updates = int(resume_state.get("updates", 0))
            _truncate_metrics_by_step(metrics_path, total_steps)
            print(
                f"resume mappo seed={run_seed}: "
                f"next_step={total_steps} next_episode={episode} "
                f"rollout={len(rollout)}",
                flush=True,
            )
        else:
            _clear_metrics(metrics_path)

        start = time.perf_counter()
        budget = int(args.total_steps)
        while total_steps < budget:
            episode_seed = _episode_seed(run_seed, episode)
            observations_dict, _ = env.reset(seed=episode_seed)
            observations = stack_observations(observations_dict, agents)
            state = env.state().astype(np.float32, copy=True)
            masks = stack_masks(env)
            team_return = 0.0
            speeds: list[float] = []
            collision = False
            episode_steps = 0
            losses: list[float] = []
            eval_metrics: dict[str, Any] | None = None

            while total_steps < budget:
                actions, log_probs, value = learner.act_with_info(
                    observations, masks, state, deterministic=False
                )
                next_obs_dict, _rewards, terminated, truncated, info = env.step(
                    {agent: int(actions[index]) for index, agent in enumerate(agents)}
                )
                next_observations = stack_observations(next_obs_dict, agents)
                next_state = env.state().astype(np.float32, copy=True)
                next_masks = stack_masks(env)
                team_reward = float(info["team_reward"])
                terminal = any_flag(terminated)
                truncated_flag = any_flag(truncated)
                rollout.add(
                    observations,
                    state,
                    actions,
                    log_probs,
                    value,
                    team_reward,
                    masks,
                    terminal,
                    truncated_flag,
                    next_state,
                )
                total_steps += 1
                episode_steps += 1
                team_return += team_reward
                speeds.extend(
                    float(info["per_agent"][agent]["speed_mps"]) for agent in agents
                )
                collision = collision or bool(info["collision"])
                observations, state, masks = (
                    next_observations,
                    next_state,
                    next_masks,
                )

                if len(rollout) >= args.rollout_steps:
                    stats = learner.update(rollout)
                    losses.append(stats["loss"])
                    updates += 1
                    rollout.reset()

                if total_steps % args.eval_interval == 0:
                    eval_rows = []
                    for eval_seed in eval_seeds:
                        result = _eval_episode(
                            eval_env,
                            lambda obs, eval_masks: learner.act(
                                obs, eval_masks, deterministic=True
                            ),
                            eval_seed,
                        )
                        eval_rows.append(
                            {
                                "algorithm": "mappo",
                                "seed": run_seed,
                                "episode": episode,
                                "total_steps": total_steps,
                                "epsilon": 0.0,
                                "loss": "",
                                "eval": 1,
                                "env_seed": eval_seed,
                                **result,
                            }
                        )
                    _append_rows(metrics_path, eval_rows)
                    eval_metrics = {
                        "team_return": float(
                            np.mean([row["team_return"] for row in eval_rows])
                        ),
                        "collision": float(
                            np.mean([row["collision"] for row in eval_rows])
                        ),
                    }
                if terminal or truncated_flag:
                    break

            if len(rollout) > 0 and total_steps >= budget:
                stats = learner.update(rollout)
                losses.append(stats["loss"])
                updates += 1
                rollout.reset()

            episode_metrics = _episode_metrics(
                team_return, collision, speeds, episode_steps
            )
            row = {
                "algorithm": "mappo",
                "seed": run_seed,
                "episode": episode,
                "total_steps": total_steps,
                "epsilon": 0.0,
                "loss": float(np.mean(losses)) if losses else "",
                "eval": 0,
                "env_seed": episode_seed,
                **episode_metrics,
            }
            _append_rows(metrics_path, [row])
            learner.save(resume_path)
            rollout.save(rollout_path)
            state_path.write_text(
                json.dumps(
                    {
                        "next_step": total_steps,
                        "next_episode": episode + 1,
                        "updates": updates,
                    }
                ),
                encoding="utf-8",
            )

            elapsed = time.perf_counter() - start
            loss_text = f"{row['loss']:.4f}" if row["loss"] != "" else "no-update"
            eval_text = (
                f" eval_return={eval_metrics['team_return']:.3f}"
                f" eval_collision={eval_metrics['collision']:.2f}"
                if eval_metrics is not None
                else ""
            )
            print(
                f"mappo seed={run_seed} ep={episode + 1} "
                f"total_steps={total_steps}/{budget} "
                f"return={episode_metrics['team_return']:.3f} "
                f"collision={episode_metrics['collision']} "
                f"speed={episode_metrics['mean_speed']:.3f} "
                f"steps={episode_steps} updates={updates} "
                f"loss={loss_text}{eval_text} elapsed={elapsed:.1f}s",
                flush=True,
            )
            episode += 1

        elapsed = time.perf_counter() - start
        (run_dir / "timing.json").write_text(
            json.dumps(
                {
                    "algorithm": "mappo",
                    "seed": run_seed,
                    "total_steps": total_steps,
                    "updates": updates,
                    "elapsed_s": elapsed,
                    "sec_per_step": elapsed / max(1, total_steps),
                    "steps_per_s": total_steps / max(elapsed, 1e-9),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    finally:
        env.close()
        eval_env.close()


def _run_random_baseline(
    args: argparse.Namespace,
    eval_seeds: list[int],
    output: Path,
) -> None:
    baseline_path = output / "runs" / "random" / "metrics.csv"
    if args.resume and baseline_path.exists() and baseline_path.stat().st_size > 0:
        print("random baseline: already present, skipping", flush=True)
        return
    _clear_metrics(baseline_path)
    env = make_env(args.obs_layout, args.background_vehicles, args.duration)
    try:
        for eval_seed in eval_seeds:
            rng = np.random.default_rng(10_000 + eval_seed)
            result = _eval_episode(
                env,
                lambda obs, masks: random_policy(obs, masks, rng),
                eval_seed,
            )
            _append_rows(
                baseline_path,
                [
                    {
                        "algorithm": "random",
                        "seed": 0,
                        "episode": 0,
                        "total_steps": 0,
                        "epsilon": 1.0,
                        "loss": "",
                        "eval": 1,
                        "env_seed": eval_seed,
                        **result,
                    }
                ],
            )
        print(
            f"random baseline: {len(eval_seeds)} eval episodes on seeds {eval_seeds}",
            flush=True,
        )
    finally:
        env.close()


def _update_metadata(
    output: Path,
    args: argparse.Namespace,
    obs_dim: int,
    state_dim: int,
    n_actions: int,
    eval_seeds: list[int],
    test_seeds: list[int],
    effective_background: int,
) -> dict[str, Any]:
    metadata_path = output / "run_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    algorithms = list(metadata.get("algorithms", []))
    if "mappo" not in algorithms:
        algorithms.append("mappo")
    shared_defaults = {
        "obs_layout": args.obs_layout,
        "obs_dim": int(obs_dim),
        "state_dim": int(state_dim),
        "n_actions": int(n_actions),
        "background_vehicles": int(effective_background),
        "duration": float(args.duration),
        "total_steps": int(args.total_steps),
        "eval_interval": int(args.eval_interval),
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "test_seeds": [int(seed) for seed in test_seeds],
        "device": str(args.device),
        "state_norm": bool(args.state_norm),
    }
    for key, value in shared_defaults.items():
        metadata.setdefault(key, value)
    metadata["algorithms"] = algorithms
    metadata["mappo_train_seeds"] = [int(seed) for seed in args.seeds]
    mappo_config = {
        "hidden_dim": int(args.hidden_dim),
        "learning_rate": float(args.learning_rate),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "clip_ratio": float(args.clip_ratio),
        "entropy_coef": float(args.entropy_coef),
        "value_coef": float(args.value_coef),
        "value_clip": args.value_clip,
        "max_grad_norm": float(args.max_grad_norm),
        "ppo_epochs": int(args.ppo_epochs),
        "minibatch_size": int(args.minibatch_size),
        "rollout_steps": int(args.rollout_steps),
        "reward_scale": float(args.reward_scale),
        "state_norm": bool(args.state_norm),
    }
    metadata["mappo_config"] = mappo_config
    output.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "run_metadata_mappo.json").write_text(
        json.dumps(
            {
                "algorithm": "mappo",
                "obs_layout": args.obs_layout,
                "obs_dim": int(obs_dim),
                "state_dim": int(state_dim),
                "n_actions": int(n_actions),
                "train_seeds": [int(seed) for seed in args.seeds],
                "eval_seeds": [int(seed) for seed in eval_seeds],
                "test_seeds": [int(seed) for seed in test_seeds],
                "background_vehicles": int(effective_background),
                "duration": float(args.duration),
                "total_steps": int(args.total_steps),
                "eval_interval": int(args.eval_interval),
                "device": str(args.device),
                "config": mappo_config,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--eval-seeds", nargs="+", type=int, default=(7001, 7002, 7003)
    )
    parser.add_argument("--test-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=15_000)
    parser.add_argument("--obs-layout", default="contract56")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--background-vehicles", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=3_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--value-clip", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--state-norm", action="store_true")
    parser.add_argument("--state-mean", nargs="+", type=float, default=None)
    parser.add_argument("--state-std", nargs="+", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--out", default="results/raw/comparison")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.total_steps <= 0 or args.duration <= 0 or args.rollout_steps <= 0:
        parser.error("total-steps, duration, and rollout-steps must be positive.")
    if args.total_steps >= 100_000:
        parser.error("total-steps must be below 100000.")
    if args.eval_interval <= 0 or args.torch_threads <= 0:
        parser.error("eval-interval and torch-threads must be positive.")
    if args.minibatch_size <= 0 or args.ppo_epochs <= 0:
        parser.error("minibatch-size and ppo-epochs must be positive.")
    if args.state_norm and (args.state_mean is None or args.state_std is None):
        parser.error("--state-norm requires explicit --state-mean and --state-std.")
    if args.test_seeds is None:
        args.test_seeds = list(range(9001, 9031))
    return args


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_threads)
    eval_seeds = sorted(set(args.eval_seeds))
    test_seeds = sorted(set(args.test_seeds))
    eval_set = set(eval_seeds)
    test_set = set(test_seeds)
    if eval_set & test_set:
        raise ValueError("Evaluation seeds overlap test seeds.")
    if any(seed < 0 or seed >= 100_000 for seed in eval_set | test_set):
        raise ValueError("Evaluation and test seeds must be in [0, 100000).")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    probe = make_env(args.obs_layout, args.background_vehicles, args.duration)
    try:
        probe.reset(seed=0)
        obs_dim = int(probe.obs_dim)
        state_dim = int(probe.state().size)
        n_actions = int(probe.n_actions)
    finally:
        probe.close()
    if args.obs_layout == "contract56":
        assert obs_dim == 56, f"contract56 must be 56-D, got {obs_dim}"
    effective_background = (
        max(args.background_vehicles, 12)
        if args.obs_layout == "contract56"
        else args.background_vehicles
    )

    metadata = _update_metadata(
        output,
        args,
        obs_dim,
        state_dim,
        n_actions,
        eval_seeds,
        test_seeds,
        effective_background,
    )
    print(
        f"mappo run config: obs_layout={args.obs_layout} obs_dim={obs_dim} "
        f"state_dim={state_dim} n_actions={n_actions} "
        f"train_seeds={args.seeds} eval_seeds={eval_seeds} "
        f"test_seeds={len(test_seeds)} total_steps={args.total_steps} "
        f"eval_interval={args.eval_interval} "
        f"background_vehicles={effective_background} duration={args.duration} "
        f"rollout_steps={args.rollout_steps} ppo_epochs={args.ppo_epochs} "
        f"minibatch_size={args.minibatch_size} clip_ratio={args.clip_ratio} "
        f"gae_lambda={args.gae_lambda} entropy_coef={args.entropy_coef} "
        f"value_coef={args.value_coef} reward_scale={args.reward_scale} "
        f"state_norm={args.state_norm} device={args.device} "
        f"algorithms={metadata.get('algorithms')}",
        flush=True,
    )

    _run_random_baseline(args, eval_seeds, output)
    for seed in args.seeds:
        _run_training(seed, args, eval_seeds, output)
    print(f"Done. CSVs and resume files under {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
