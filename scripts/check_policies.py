"""Compare simple CARLA control policies over the same fixed episode seeds.

Run with CARLA 0.9.16 already running:
    python scripts/check_policies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from envs.carla_env import CarlaMultiAgentEnv

EPISODE_SEEDS = tuple(range(20))
POLICY_NAMES = ("always_keep", "always_accelerate", "uniform_random_valid")


def _termination_reason(info, terminated, truncated):
    if info["collision"]:
        return "collision"
    if terminated and info["goal_reached"]:
        return "goal"
    if terminated and info["off_road"]:
        return "off_road"
    if terminated and info["stuck"]:
        return "stuck"
    if truncated:
        return "timeout"
    raise RuntimeError("Episode ended without a recognized termination reason.")


def _policy_action(policy_name, agent_id, masks, rng):
    if policy_name == "always_keep":
        return 0
    if policy_name == "always_accelerate":
        return 1
    return int(rng.choice(np.flatnonzero(masks[agent_id])))


def _run_policy(env, policy_name):
    rng = np.random.default_rng(2026)
    speed_samples = []
    episode_steps = []
    reason_counts = {
        "collision": 0,
        "goal": 0,
        "off_road": 0,
        "stuck": 0,
        "timeout": 0,
    }
    goal_reached_count = 0
    collision_count = 0

    for seed in EPISODE_SEEDS:
        env.reset(seed=seed)
        for _ in range(env.max_steps):
            masks = env.action_masks()
            actions = {
                agent_id: _policy_action(policy_name, agent_id, masks, rng)
                for agent_id in env.agents
            }
            _, _, terminated, truncated, info = env.step(actions)
            speed_samples.extend(
                float(vehicle.get_velocity().length())
                for vehicle in env.vehicles.values()
            )
            if any(terminated.values()) or any(truncated.values()):
                steps = int(info["steps"])
                reason = _termination_reason(
                    info,
                    any(terminated.values()),
                    any(truncated.values()),
                )
                episode_steps.append(steps)
                reason_counts[reason] += 1
                goal_reached_count += int(bool(info["goal_reached"]))
                collision_count += int(bool(info["collision"]))
                print(
                    f"policy={policy_name} episode={seed + 1:02d}/"
                    f"{len(EPISODE_SEEDS)} steps={steps} reason={reason}",
                    flush=True,
                )
                break
        else:
            raise RuntimeError(
                f"{policy_name}, seed {seed}: episode exceeded max_steps"
            )

    return {
        "mean_speed": float(np.mean(speed_samples)) if speed_samples else 0.0,
        "mean_steps": float(np.mean(episode_steps)) if episode_steps else 0.0,
        "reason_counts": reason_counts,
        "goal_reached": goal_reached_count,
        "collisions": collision_count,
    }


def main():
    env = CarlaMultiAgentEnv(town="Town03")
    try:
        results = {
            policy_name: _run_policy(env, policy_name)
            for policy_name in POLICY_NAMES
        }
    finally:
        env.close()

    for policy_name, result in results.items():
        print(f"\n{policy_name} ({len(EPISODE_SEEDS)} episodes)")
        print(f"  mean ego speed:   {result['mean_speed']:.2f} m/s")
        print(f"  mean steps:       {result['mean_steps']:.1f}")
        print(f"  termination counts: {result['reason_counts']}")
        print(f"  goal-reached episodes: {result['goal_reached']}")
        print(f"  collision episodes:    {result['collisions']}")


if __name__ == "__main__":
    main()
