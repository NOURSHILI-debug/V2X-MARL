"""Train QMIX and VDN on identical seeded Highway-Env contract56 episodes.

Writes one CSV row per training episode and per evaluation episode under
``results/raw/`` (gitignored), plus a resumable learner/buffer checkpoint after
every episode. No CARLA, no toy env.
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
from marl.buffers import ReplayBuffer
from marl.qmix import QMIX, QMIXReplayBuffer
from marl.vdn import VDN
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
    "team_return",
    "collision",
    "mean_speed",
    "steps",
    "epsilon",
    "loss",
    "eval",
    "env_seed",
]
QMIX_BUFFER_KEYS = (
    "observations",
    "states",
    "actions",
    "rewards",
    "next_observations",
    "next_states",
    "terminated",
    "action_masks",
    "next_action_masks",
)
VDN_BUFFER_KEYS = ("obs", "act", "rew", "done", "mask", "next_obs", "next_mask")


def _transition(
    algorithm: str,
    observations: np.ndarray,
    state: np.ndarray,
    actions: np.ndarray,
    reward: float,
    next_observations: np.ndarray,
    next_state: np.ndarray,
    terminated: bool,
    next_masks: np.ndarray,
    masks: np.ndarray,
) -> tuple[Any, ...]:
    if algorithm == "qmix":
        return (
            observations,
            state,
            actions,
            reward,
            next_observations,
            next_state,
            terminated,
            next_masks,
            masks,
        )
    return (
        observations,
        actions,
        reward,
        next_observations,
        terminated,
        next_masks,
        masks,
    )


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


def _train_episode(
    env: HighwayV2XEnv,
    learner: QMIX | VDN,
    algorithm: str,
    reset_seed: int,
    epsilon: float,
) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    observations_dict, _ = env.reset(seed=reset_seed)
    agents = env.agent_ids
    observations = stack_observations(observations_dict, agents)
    state = env.state().astype(np.float32, copy=True)
    masks = stack_masks(env)
    transitions: list[tuple[Any, ...]] = []
    team_return = 0.0
    speeds: list[float] = []
    collision = False
    steps = 0

    while True:
        actions = learner.act(observations, masks, epsilon=epsilon)
        next_obs_dict, _rewards, terminated, truncated, info = env.step(
            {agent: int(actions[index]) for index, agent in enumerate(agents)}
        )
        next_observations = stack_observations(next_obs_dict, agents)
        next_state = env.state().astype(np.float32, copy=True)
        next_masks = stack_masks(env)
        team_reward = float(info["team_reward"])
        terminal = any_flag(terminated)

        transitions.append(
            _transition(
                algorithm,
                observations,
                state,
                actions,
                team_reward,
                next_observations,
                next_state,
                terminal,
                next_masks,
                masks,
            )
        )
        team_return += team_reward
        speeds.extend(
            float(info["per_agent"][agent]["speed_mps"]) for agent in agents
        )
        collision = collision or bool(info["collision"])
        steps += 1
        observations, state, masks = next_observations, next_state, next_masks
        if terminal or any_flag(truncated):
            break

    return transitions, _episode_metrics(team_return, collision, speeds, steps)


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
    algorithm: str,
    args: argparse.Namespace,
    obs_dim: int,
    state_dim: int,
    n_actions: int,
    seed: int,
) -> tuple[QMIX | VDN, QMIXReplayBuffer | ReplayBuffer]:
    n_agents = 4
    if algorithm == "qmix":
        learner = QMIX(
            n_agents=n_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=args.hidden_dim,
            mixing_dim=args.mixing_dim,
            hypernet_dim=args.hypernet_dim,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            tau=args.tau,
            reward_scale=args.reward_scale,
            double_q=args.double_q,
            state_norm=args.state_norm,
            state_mean=args.state_mean,
            state_std=args.state_std,
            device=args.device,
            seed=seed,
        )
        replay: QMIXReplayBuffer | ReplayBuffer = QMIXReplayBuffer(
            args.replay_capacity,
            n_agents,
            obs_dim,
            state_dim,
            n_actions,
        )
    else:
        learner = VDN(
            n_agents,
            obs_dim,
            n_actions,
            lr=args.learning_rate,
            gamma=args.gamma,
            hidden=args.hidden_dim,
            tau=args.tau,
            reward_scale=args.reward_scale,
            double_q=args.double_q,
            device=args.device,
            seed=seed,
        )
        replay = ReplayBuffer(
            args.replay_capacity,
            n_agents,
            obs_dim,
            n_actions,
        )
    return learner, replay


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


def _truncate_metrics(path: Path, keep_before_episode: int) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    kept = [
        row
        for row in rows
        if row.get("episode") not in (None, "")
        and int(row["episode"]) < keep_before_episode
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def _save_buffer(buffer: QMIXReplayBuffer | ReplayBuffer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(buffer, QMIXReplayBuffer):
        arrays = {key: getattr(buffer, key) for key in QMIX_BUFFER_KEYS}
        arrays["__size"] = np.asarray(buffer.size)
        arrays["__pointer"] = np.asarray(buffer.pointer)
    else:
        arrays = {key: getattr(buffer, key) for key in VDN_BUFFER_KEYS}
        arrays["__size"] = np.asarray(buffer.size)
        arrays["__pointer"] = np.asarray(buffer.ptr)
    np.savez(path, **arrays)


def _load_buffer(buffer: QMIXReplayBuffer | ReplayBuffer, path: Path) -> None:
    with np.load(path) as arrays:
        keys = QMIX_BUFFER_KEYS if isinstance(buffer, QMIXReplayBuffer) else VDN_BUFFER_KEYS
        for key in keys:
            getattr(buffer, key)[...] = arrays[key]
        buffer.size = int(arrays["__size"])
        if isinstance(buffer, QMIXReplayBuffer):
            buffer.pointer = int(arrays["__pointer"])
        else:
            buffer.ptr = int(arrays["__pointer"])


def _run_training(
    algorithm: str,
    run_seed: int,
    args: argparse.Namespace,
    eval_seeds: list[int],
    output: Path,
) -> None:
    seed_everything(run_seed)
    run_dir = output / "runs" / f"{algorithm}_seed{run_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    resume_path = run_dir / "resume.pt"
    buffer_path = run_dir / "buffer.npz"
    state_path = run_dir / "state.json"
    env = make_env(args.obs_layout, args.background_vehicles, args.duration)
    try:
        agents = env.agent_ids
        env.reset(seed=run_seed * 100_000)
        obs_dim = int(env.obs_dim)
        state_dim = int(env.state().size)
        n_actions = int(env.n_actions)
        if args.obs_layout == "contract56":
            assert obs_dim == 56, f"contract56 must be 56-D, got {obs_dim}"

        learner, replay = _build_learner(
            algorithm, args, obs_dim, state_dim, n_actions, run_seed
        )
        next_episode = 0
        if (
            args.resume
            and resume_path.exists()
            and state_path.exists()
        ):
            learner = (
                QMIX.from_checkpoint(resume_path, device=args.device)
                if algorithm == "qmix"
                else VDN.from_checkpoint(resume_path, device=args.device)
            )
            if buffer_path.exists():
                _load_buffer(replay, buffer_path)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            next_episode = int(state["next_episode"])
            learner.rng.bit_generator.state = state["learner_rng"]
            _truncate_metrics(metrics_path, next_episode)
            print(
                f"resume {algorithm} seed={run_seed}: next_episode={next_episode}",
                flush=True,
            )
        else:
            _truncate_metrics(metrics_path, 0)

        start = time.perf_counter()
        for episode in range(next_episode, args.episodes):
            epsilon = max(
                args.epsilon_end,
                1.0 - episode / max(1.0, args.epsilon_fraction * args.episodes),
            )
            episode_seed = run_seed * 100_000 + episode
            transitions, episode_metrics = _train_episode(
                env, learner, algorithm, episode_seed, epsilon
            )
            losses: list[float] = []
            for transition in transitions:
                replay.add(*transition)
                if len(replay) >= args.warmup:
                    batch = replay.sample(args.batch_size, args.device)
                    losses.append(learner.update(batch))

            row = {
                "algorithm": algorithm,
                "seed": run_seed,
                "episode": episode,
                "epsilon": epsilon,
                "loss": float(np.mean(losses)) if losses else "",
                "eval": 0,
                "env_seed": episode_seed,
                **episode_metrics,
            }
            _append_rows(metrics_path, [row])

            eval_metrics: dict[str, Any] | None = None
            is_eval = (
                (episode + 1) % args.eval_interval == 0
                or episode == args.episodes - 1
            )
            if is_eval:
                eval_rows = []
                for eval_seed in eval_seeds:
                    result = _eval_episode(
                        env,
                        lambda obs, masks: learner.act(obs, masks, epsilon=0.0),
                        eval_seed,
                    )
                    eval_rows.append(
                        {
                            "algorithm": algorithm,
                            "seed": run_seed,
                            "episode": episode,
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

            learner.save(resume_path)
            _save_buffer(replay, buffer_path)
            state_path.write_text(
                json.dumps(
                    {
                        "next_episode": episode + 1,
                        "learner_rng": learner.rng.bit_generator.state,
                    },
                    default=int,
                ),
                encoding="utf-8",
            )

            elapsed = time.perf_counter() - start
            loss_text = f"{row['loss']:.4f}" if row["loss"] != "" else "warmup"
            eval_text = (
                f" eval_return={eval_metrics['team_return']:.3f}"
                f" eval_collision={eval_metrics['collision']:.2f}"
                if eval_metrics is not None
                else ""
            )
            print(
                f"{algorithm} seed={run_seed} ep={episode + 1}/{args.episodes} "
                f"return={episode_metrics['team_return']:.3f} "
                f"collision={episode_metrics['collision']} "
                f"speed={episode_metrics['mean_speed']:.3f} "
                f"steps={episode_metrics['steps']} eps={epsilon:.3f} "
                f"loss={loss_text}{eval_text} elapsed={elapsed:.1f}s",
                flush=True,
            )
    finally:
        env.close()


def _run_random_baseline(
    args: argparse.Namespace,
    eval_seeds: list[int],
    output: Path,
) -> None:
    baseline_path = output / "runs" / "random" / "metrics.csv"
    if args.resume and baseline_path.exists() and baseline_path.stat().st_size > 0:
        print("random baseline: already present, skipping", flush=True)
        return
    _truncate_metrics(baseline_path, 0)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms", nargs="+", choices=("qmix", "vdn"), default=("qmix", "vdn")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument(
        "--eval-seeds",
        nargs="+",
        type=int,
        default=(7001, 7002, 7003, 7004, 7005),
    )
    parser.add_argument("--test-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--obs-layout", default="contract56")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--background-vehicles", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mixing-dim", type=int, default=32)
    parser.add_argument("--hypernet-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--double-q", action="store_true")
    parser.add_argument("--state-norm", action="store_true")
    parser.add_argument("--state-mean", nargs="+", type=float, default=None)
    parser.add_argument("--state-std", nargs="+", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-fraction", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--out", default="results/raw/comparison")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.episodes <= 0 or args.duration <= 0 or args.batch_size <= 0:
        parser.error("episodes, duration, and batch-size must be positive.")
    if args.eval_interval <= 0 or args.torch_threads <= 0:
        parser.error("eval-interval and torch-threads must be positive.")
    if args.warmup <= 0 or args.replay_capacity < args.warmup:
        parser.error("warmup must be positive and no greater than replay-capacity.")
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
    train_seeds = {
        seed * 100_000 + episode
        for seed in args.seeds
        for episode in range(args.episodes)
    }
    eval_set = set(eval_seeds)
    test_set = set(test_seeds)
    if train_seeds & eval_set:
        raise ValueError("Training episode seeds overlap evaluation seeds.")
    if train_seeds & test_set:
        raise ValueError("Training episode seeds overlap test seeds.")
    if eval_set & test_set:
        raise ValueError("Evaluation seeds overlap test seeds.")

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
    effective_background = max(args.background_vehicles, 12) if args.obs_layout == "contract56" else args.background_vehicles

    config = {
        "obs_layout": args.obs_layout,
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "n_actions": n_actions,
        "algorithms": list(args.algorithms),
        "train_seeds": list(args.seeds),
        "eval_seeds": eval_seeds,
        "test_seeds": test_seeds,
        "background_vehicles": effective_background,
        "duration": args.duration,
        "episodes": args.episodes,
        "eval_interval": args.eval_interval,
        "double_q": bool(args.double_q),
        "state_norm": bool(args.state_norm),
        "device": args.device,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"run config: obs_layout={args.obs_layout} obs_dim={obs_dim} "
        f"state_dim={state_dim} n_actions={n_actions} "
        f"train_seeds={args.seeds} eval_seeds={eval_seeds} "
        f"test_seeds={len(test_seeds)} episodes={args.episodes} "
        f"eval_interval={args.eval_interval} "
        f"background_vehicles={effective_background} duration={args.duration} "
        f"double_q={args.double_q} state_norm={args.state_norm} "
        f"device={args.device}",
        flush=True,
    )

    _run_random_baseline(args, eval_seeds, output)
    for algorithm in args.algorithms:
        for seed in args.seeds:
            _run_training(algorithm, seed, args, eval_seeds, output)
    print(f"Done. CSVs and resume files under {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
