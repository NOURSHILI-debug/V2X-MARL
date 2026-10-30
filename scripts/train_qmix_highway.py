"""Train QMIX and the VDN baseline on identical seeded Highway-Env episodes."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.highway_env import HighwayV2XEnv
from marl.buffers import ReplayBuffer
from marl.qmix import QMIX, QMIXReplayBuffer
from marl.vdn import VDN


UNAVAILABLE_METRICS = (
    "success_rate,min_ttc_s,mean_jerk,v2x_messages_received,"
    "v2x_messages_dropped,selected_teammate_ids"
)


def _stack(mapping: dict[str, Any], agents: tuple[str, ...]) -> np.ndarray:
    return np.stack(
        [np.asarray(mapping[agent], dtype=np.float32).reshape(-1) for agent in agents]
    )


def _masks(env: HighwayV2XEnv) -> np.ndarray:
    masks = env.action_masks()
    return np.stack(
        [np.asarray(masks[agent], dtype=bool) for agent in env.agent_ids]
    )


def _flag(mapping: Any) -> bool:
    return any(mapping.values()) if isinstance(mapping, dict) else bool(mapping)


def _team_reward(rewards: dict[str, float] | float) -> float:
    if isinstance(rewards, dict):
        return float(np.mean(list(rewards.values())))
    return float(rewards)


def _run_episode(
    env: HighwayV2XEnv,
    learner: QMIX | VDN,
    algorithm: str,
    seed: int,
    epsilon: float,
) -> tuple[list[tuple[Any, ...]], dict[str, float]]:
    obs_dict, _ = env.reset(seed=seed)
    agents = env.agent_ids
    observations = _stack(obs_dict, agents)
    state = env.state().astype(np.float32, copy=True)
    masks = _masks(env)
    transitions: list[tuple[Any, ...]] = []
    episode_return = 0.0
    speed_samples: list[float] = []
    collision_count = 0

    while True:
        actions = learner.act(observations, masks, epsilon=epsilon)
        next_obs_dict, rewards, terminated, truncated, infos = env.step(
            {agent: int(actions[index]) for index, agent in enumerate(agents)}
        )
        next_observations = _stack(next_obs_dict, agents)
        next_state = env.state().astype(np.float32, copy=True)
        next_masks = _masks(env)
        team_reward = _team_reward(rewards)
        terminal = _flag(terminated)
        ended = terminal or _flag(truncated)

        if algorithm == "qmix":
            transitions.append(
                (
                    observations,
                    state,
                    actions,
                    team_reward,
                    next_observations,
                    next_state,
                    terminal,
                    next_masks,
                )
            )
        else:
            transitions.append(
                (
                    observations,
                    actions,
                    team_reward,
                    next_observations,
                    terminal,
                    next_masks,
                )
            )

        episode_return += team_reward
        speed_samples.extend(float(info["speed_mps"]) for info in infos.values())
        collision_count += int(any(bool(info["crashed"]) for info in infos.values()))
        observations, state, masks = next_observations, next_state, next_masks
        if ended:
            return transitions, {
                "episode_return": episode_return,
                "mean_step_reward": episode_return / len(transitions),
                "collision_steps": float(collision_count),
                "collision_rate": float(collision_count > 0),
                "mean_speed_mps": float(np.mean(speed_samples)) if speed_samples else 0.0,
                "episode_steps": float(len(transitions)),
                "episode_duration_s": len(transitions) / 5.0,
            }


def _train_algorithm(
    algorithm: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = HighwayV2XEnv(
        render_mode=None,
        background_vehicles=args.background_vehicles,
        duration=args.duration,
    )
    try:
        agents = env.agent_ids
        obs_dict, _ = env.reset(seed=args.seed)
        obs_dim = int(np.asarray(obs_dict[agents[0]]).size)
        state_dim = int(env.state().size)
        n_actions = int(env.action_masks()[agents[0]].size)
        if len(agents) != 4 or n_actions != 6:
            raise ValueError("HighwayV2XEnv must expose four agents and six actions.")

        if algorithm == "qmix":
            learner = QMIX(
                n_agents=len(agents),
                obs_dim=obs_dim,
                state_dim=state_dim,
                n_actions=n_actions,
                hidden_dim=args.hidden_dim,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                tau=args.tau,
                reward_scale=args.reward_scale,
                device=args.device,
                seed=args.seed,
            )
            replay: QMIXReplayBuffer | ReplayBuffer = QMIXReplayBuffer(
                args.replay_capacity,
                len(agents),
                obs_dim,
                state_dim,
                n_actions,
            )
        else:
            learner = VDN(
                len(agents),
                obs_dim,
                n_actions,
                lr=args.learning_rate,
                gamma=args.gamma,
                hidden=args.hidden_dim,
                tau=args.tau,
                reward_scale=args.reward_scale,
                device=args.device,
            )
            replay = ReplayBuffer(args.replay_capacity, len(agents), obs_dim, n_actions)

        print(
            f"{algorithm}: agents={len(agents)} flattened_obs_dim={obs_dim} "
            f"state_dim={state_dim} actions={n_actions}"
        )
        rows: list[dict[str, Any]] = []
        parameter_rows: list[dict[str, Any]] = []
        for episode in range(args.episodes):
            epsilon = max(
                args.epsilon_end,
                1.0 - episode / max(1.0, args.epsilon_fraction * args.episodes),
            )
            episode_seed = args.seed + episode
            transitions, metrics = _run_episode(
                env, learner, algorithm, episode_seed, epsilon
            )
            losses = []
            for transition in transitions:
                replay.add(*transition)
                if len(replay) >= args.warmup:
                    batch = replay.sample(args.batch_size, args.device)
                    losses.append(learner.update(batch))
            row: dict[str, Any] = {
                "algorithm": algorithm,
                "episode": episode,
                "seed": episode_seed,
                "epsilon": epsilon,
                **metrics,
                "td_loss": float(np.mean(losses)) if losses else "",
                "success_rate": "NA",
                "min_ttc_s": "NA",
                "mean_jerk": "NA",
                "v2x_messages_received": "NA",
                "v2x_messages_dropped": "NA",
                "selected_teammate_ids": "NA",
                "unavailable_metrics": UNAVAILABLE_METRICS,
            }
            rows.append(row)
            if algorithm == "qmix":
                parameter_rows.append(
                    {"episode": episode, **learner.parameter_summary()}
                )
            if episode % args.log_every == 0 or episode == args.episodes - 1:
                loss_display = row["td_loss"] if row["td_loss"] != "" else "warming up"
                print(
                    f"{algorithm} ep={episode + 1}/{args.episodes} "
                    f"return={metrics['episode_return']:.3f} "
                    f"collisions={int(metrics['collision_steps'])} "
                    f"mean_speed={metrics['mean_speed_mps']:.3f} "
                    f"duration={metrics['episode_duration_s']:.1f}s "
                    f"td_loss={loss_display}"
                )

        output = Path(args.out)
        output.mkdir(parents=True, exist_ok=True)
        if algorithm == "qmix":
            learner.save(output / "qmix_highway.pt")
        else:
            torch.save(
                {
                    "algorithm": "vdn",
                    "n_agents": learner.n_agents,
                    "n_actions": learner.n_actions,
                    "q": learner.q.state_dict(),
                },
                output / "vdn_highway.pt",
            )
        return rows, parameter_rows
    finally:
        env.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(output: Path, rows: list[dict[str, Any]], parameter_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algorithms = sorted({row["algorithm"] for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for algorithm in algorithms:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        episodes = np.asarray([row["episode"] for row in selected])
        returns = np.asarray([float(row["episode_return"]) for row in selected])
        collisions = np.asarray([float(row["collision_rate"]) for row in selected])
        speeds = np.asarray([float(row["mean_speed_mps"]) for row in selected])
        losses = np.asarray(
            [float(row["td_loss"]) if row["td_loss"] != "" else np.nan for row in selected]
        )
        window = min(20, len(returns))
        rolling = np.convolve(returns, np.ones(window) / window, mode="valid")
        axes[0, 0].plot(episodes, returns, alpha=0.2)
        axes[0, 0].plot(episodes[window - 1 :], rolling, label=algorithm)
        axes[0, 1].plot(episodes, collisions, label=algorithm)
        axes[1, 0].plot(episodes, speeds, label=algorithm)
        axes[1, 1].plot(episodes, losses, alpha=0.8, label=algorithm)
    axes[0, 0].set_title("Episode return (raw + rolling mean)")
    axes[0, 1].set_title("Episodes with collision")
    axes[1, 0].set_title("Mean speed (m/s)")
    axes[1, 1].set_title("Mean TD loss")
    for axis in axes.flat:
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "highway_learning_curves.png", dpi=160)
    plt.close(figure)

    stability, axis = plt.subplots(figsize=(8, 4))
    for algorithm in algorithms:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        returns = np.asarray([float(row["episode_return"]) for row in selected])
        window = min(20, len(returns))
        if window > 1:
            rolling_std = [
                float(np.std(returns[max(0, index - window + 1) : index + 1]))
                for index in range(len(returns))
            ]
            axis.plot(range(len(returns)), rolling_std, label=algorithm)
    axis.set(title="Training stability (rolling return standard deviation)", xlabel="Episode")
    axis.grid(alpha=0.2)
    axis.legend()
    stability.tight_layout()
    stability.savefig(output / "training_stability.png", dpi=160)
    plt.close(stability)

    if parameter_rows:
        figure, axis = plt.subplots(figsize=(8, 4))
        for key in parameter_rows[0]:
            if key != "episode":
                axis.plot(
                    [row["episode"] for row in parameter_rows],
                    [row[key] for row in parameter_rows],
                    label=key,
                )
        axis.set(title="QMIX selected parameter norms", xlabel="Episode", ylabel="L2 norm")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "qmix_parameter_evolution.png", dpi=160)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=("qmix", "vdn"), default=("qmix", "vdn"))
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--background-vehicles", type=int, default=0)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-fraction", type=float, default=0.7)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--out", default="results/qmix_highway")
    args = parser.parse_args()
    if args.episodes <= 0 or args.duration <= 0 or args.batch_size <= 0:
        parser.error("episodes, duration, and batch-size must be positive.")
    if args.torch_threads <= 0:
        parser.error("torch-threads must be positive.")
    if args.warmup <= 0 or args.replay_capacity < args.warmup:
        parser.error("warmup must be positive and no greater than replay-capacity.")
    torch.set_num_threads(args.torch_threads)

    all_rows: list[dict[str, Any]] = []
    qmix_parameters: list[dict[str, Any]] = []
    for algorithm in args.algorithms:
        rows, parameter_rows = _train_algorithm(algorithm, args)
        all_rows.extend(rows)
        if algorithm == "qmix":
            qmix_parameters = parameter_rows
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "highway_training_metrics.csv", all_rows)
    _write_csv(output / "qmix_parameter_evolution.csv", qmix_parameters)
    _plot_results(output, all_rows, qmix_parameters)
    print(f"Metrics and plots saved under {output.resolve()}")
    print(f"Unavailable from this adapter: {UNAVAILABLE_METRICS}")


if __name__ == "__main__":
    main()
