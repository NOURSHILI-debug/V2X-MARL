"""Load a trained QMIX checkpoint and demonstrate its four Highway-Env agents."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.highway_env import HighwayV2XEnv
from marl.qmix import QMIX


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/qmix_highway/qmix_highway.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--background-vehicles", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive.")

    learner = QMIX.from_checkpoint(args.checkpoint, device=args.device)
    env = HighwayV2XEnv(
        render_mode=None if args.headless else "human",
        background_vehicles=args.background_vehicles,
    )
    try:
        if len(env.agent_ids) != learner.n_agents:
            raise ValueError(
                f"Checkpoint expects {learner.n_agents} agents, "
                f"but the environment has {len(env.agent_ids)}."
            )
        observations, _ = env.reset(seed=args.seed)
        observation_dim = observations[env.agent_ids[0]].size
        if observation_dim != learner.obs_dim:
            raise ValueError(
                f"Checkpoint expects flattened observations of size {learner.obs_dim}; "
                f"the environment provides {observation_dim}."
            )
        if env.state().size != learner.state_dim:
            raise ValueError(
                f"Checkpoint expects state size {learner.state_dim}; "
                f"the environment provides {env.state().size}."
            )
        print(
            f"Loaded QMIX: agents={learner.n_agents}, "
            f"obs_dim={learner.obs_dim}, state_dim={learner.state_dim}"
        )
        print(f"Agent order: {', '.join(env.agent_ids)}; seed={args.seed}")

        for step in range(args.steps):
            obs = [
                observations[agent].reshape(-1)
                for agent in env.agent_ids
            ]
            mask_map = env.action_masks()
            masks = [mask_map[agent] for agent in env.agent_ids]
            actions = learner.act(obs, masks, epsilon=0.0)
            observations, rewards, terminated, truncated, infos = env.step(
                {
                    agent: int(actions[index])
                    for index, agent in enumerate(env.agent_ids)
                }
            )
            speeds = {
                agent: round(info["speed_mps"], 2)
                for agent, info in infos.items()
            }
            collisions = [
                agent for agent, info in infos.items() if info["crashed"]
            ]
            print(
                f"step={step + 1:03d} "
                f"actions={actions.tolist()} "
                f"team_reward={sum(rewards.values()) / len(rewards):.3f} "
                f"speeds_mps={speeds} collisions={collisions}"
            )
            if any(terminated.values()) or any(truncated.values()):
                print("Episode ended.")
                break
            if not args.headless:
                time.sleep(0.03)
    finally:
        env.close()


if __name__ == "__main__":
    main()
