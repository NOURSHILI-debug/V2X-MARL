"""Exercise the CARLA observation contract over 100 short random episodes.

Run with CARLA 0.9.16 already running:
    python scripts/check_obs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from envs.carla_env import CarlaMultiAgentEnv
from envs.obs_builder import EGO, MASK, OBS_DIM, ROUTE

EPISODES = 100
SHORT_EPISODE_STEPS = 100
MASK_NAMES = ("keep", "accelerate", "decelerate", "brake", "lane_left", "lane_right")
BLOCKS = {
    "ego": slice(0, EGO),
    "route": slice(EGO, EGO + ROUTE),
    "candidates": slice(EGO + ROUTE, EGO + ROUTE + 36),
    "mask": slice(OBS_DIM - MASK, OBS_DIM),
}


def _reason(info, terminated, truncated):
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
    return None


def _validate_observations(observations, agents, block_stats):
    for agent_id in agents:
        observation = np.asarray(observations[agent_id])
        assert observation.shape == (OBS_DIM,), (
            f"{agent_id}: expected ({OBS_DIM},), got {observation.shape}"
        )
        assert observation.dtype == np.float32, (
            f"{agent_id}: expected float32, got {observation.dtype}"
        )
        assert np.isfinite(observation).all(), f"{agent_id}: observation is not finite"
        for name, block in BLOCKS.items():
            values = observation[block]
            block_stats[name][0] = min(block_stats[name][0], float(values.min()))
            block_stats[name][1] = max(block_stats[name][1], float(values.max()))


def main():
    env = CarlaMultiAgentEnv(town="Town03", n_agents=4)
    env.max_steps = SHORT_EPISODE_STEPS
    reason_counts = {
        "collision": 0,
        "goal": 0,
        "off_road": 0,
        "stuck": 0,
        "timeout": 0,
    }
    block_stats = {name: [float("inf"), float("-inf")] for name in BLOCKS}
    false_counts = np.zeros(MASK, dtype=np.int64)
    mask_samples = 0
    rng = np.random.default_rng(2026)

    try:
        for episode in range(EPISODES):
            observations, _ = env.reset(seed=episode)
            _validate_observations(observations, env.agents, block_stats)
            speeds = []
            episode_false_counts = np.zeros(MASK, dtype=np.int64)
            episode_mask_samples = 0
            reason = None

            for _ in range(SHORT_EPISODE_STEPS):
                masks = env.action_masks()
                for agent_id in env.agents:
                    mask = np.asarray(masks[agent_id], dtype=bool)
                    assert mask.shape == (MASK,)
                    assert mask.any() and mask[3]
                    episode_false_counts += ~mask
                    episode_mask_samples += 1
                    speeds.append(
                        float(env.vehicles[agent_id].get_velocity().length())
                    )

                actions = {
                    agent_id: int(rng.choice(np.flatnonzero(masks[agent_id])))
                    for agent_id in env.agents
                }
                observations, _, terminated, truncated, info = env.step(actions)
                _validate_observations(observations, env.agents, block_stats)
                if any(terminated.values()) or any(truncated.values()):
                    reason = _reason(
                        info,
                        any(terminated.values()),
                        any(truncated.values()),
                    )
                    break

            if reason is None:
                raise AssertionError(
                    f"episode {episode}: did not terminate within "
                    f"{SHORT_EPISODE_STEPS} decisions"
                )
            reason_counts[reason] += 1
            false_counts += episode_false_counts
            mask_samples += episode_mask_samples
            fractions = episode_false_counts / episode_mask_samples
            print(
                f"episode={episode + 1:03d} steps={info['steps']:3d} "
                f"reason={reason:8s} mean_ego_speed={np.mean(speeds):.2f} m/s "
                f"mask_false={dict(zip(MASK_NAMES, fractions.round(3), strict=True))}",
                flush=True,
            )

        print("\nObservation block ranges over all episodes:")
        for name, (minimum, maximum) in block_stats.items():
            print(f"{name:12s} min={minimum:.4f} max={maximum:.4f}")
        print("\nTermination reason counts:")
        print(f"{'reason':12s} {'episodes':>8s}")
        for reason, count in reason_counts.items():
            print(f"{reason:12s} {count:8d}")
        print("\nFraction of agent-steps each action was masked:")
        for name, count in zip(MASK_NAMES, false_counts, strict=True):
            print(f"{name:12s} {count / mask_samples:.4f}")
        stuck_fraction = reason_counts["stuck"] / EPISODES
        if stuck_fraction > 0.8:
            print(
                f"*** WARNING: {stuck_fraction:.0%} of episodes ended as stuck; "
                "investigate routes, controls, and termination thresholds."
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
