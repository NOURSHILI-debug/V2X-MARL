"""Exercise the CARLA observation contract over 100 short random episodes.

Run with CARLA 0.9.16 already running:
    python scripts/check_obs.py
"""

from __future__ import annotations

import sys
import argparse
import datetime as dt
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from envs.obs_builder import EGO, MASK, OBS_DIM, ROUTE

DEFAULT_EPISODES = 100
DEFAULT_MAX_STEPS = 300
MASK_NAMES = ("keep", "accelerate", "decelerate", "brake", "lane_left", "lane_right")
BLOCKS = {
    "ego": slice(0, EGO),
    "route": slice(EGO, EGO + ROUTE),
    "candidates": slice(EGO + ROUTE, EGO + ROUTE + 36),
    "mask": slice(OBS_DIM - MASK, OBS_DIM),
}


def _reason(info, terminated, truncated):
    if info.get("collision", False):
        return "collision"
    if terminated and info.get("goal_reached", False):
        return "goal"
    if terminated and info.get("off_road", False):
        return "off_road"
    if terminated and info.get("stuck", False):
        return "stuck"
    if truncated:
        return "timeout"
    return "unknown"


def _per_agent_stop_reasons(info):
    per_agent = info.get("per_agent", {})
    return {
        agent_id: details.get("stop_reason", "none")
        for agent_id, details in per_agent.items()
    }


def _validate_observations(observations, agents, block_stats, saturation):
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
            clipped = np.isclose(np.abs(values), 1.0, atol=1e-7)
            saturation[name] += int(clipped.sum())
            saturation[f"{name}_total"] += values.size
            for index, value in enumerate(clipped):
                key = f"{name}_{index}"
                saturation[key] += int(value)
                saturation[f"{key}_total"] += 1


def _new_saturation_counters():
    counters = {}
    for name, block in BLOCKS.items():
        counters[name] = 0
        counters[f"{name}_total"] = 0
        for index in range(block.stop - block.start):
            key = f"{name}_{index}"
            counters[key] = 0
            counters[f"{key}_total"] = 0
    return counters


def _git_sha():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "UNAVAILABLE"
    return sha


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("--episodes and --max-steps must be positive")
    sha = _git_sha()
    env = None
    reason_counts = {
        "collision": 0,
        "goal": 0,
        "off_road": 0,
        "stuck": 0,
        "timeout": 0,
        "max_steps_cap": 0,
        "unknown": 0,
    }
    block_stats = {name: [float("inf"), float("-inf")] for name in BLOCKS}
    saturation = _new_saturation_counters()
    false_counts = np.zeros(MASK, dtype=np.int64)
    mask_samples = 0
    rng = np.random.default_rng(2026)
    completed = False

    try:
        from envs.carla_env import CarlaMultiAgentEnv

        env = CarlaMultiAgentEnv(town="Town03", n_agents=4)
        try:
            version = env.client.get_client_version() or str(env.client)
        except Exception:
            version = str(env.client)
        print(
            f"git_sha={sha} date={dt.datetime.now().astimezone().isoformat()} "
            f"seeds=0..{args.episodes-1} CARLA={version} "
            f"episodes={args.episodes} max_steps={args.max_steps}",
            flush=True,
        )
        for episode in range(args.episodes):
            observations, _ = env.reset(seed=episode)
            _validate_observations(observations, env.agents, block_stats, saturation)
            speeds = []
            episode_false_counts = np.zeros(MASK, dtype=np.int64)
            episode_mask_samples = 0
            reason = None
            step_seconds = []

            for _ in range(args.max_steps):
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
                started = time.perf_counter()
                observations, _, terminated, truncated, info = env.step(actions)
                step_seconds.append(time.perf_counter() - started)
                _validate_observations(observations, env.agents, block_stats, saturation)
                if all(terminated.values()) or any(truncated.values()):
                    reason = _reason(
                        info,
                        all(terminated.values()),
                        any(truncated.values()),
                    )
                    if reason == "unknown":
                        print(
                            f"UNKNOWN_END seed={episode} info={info!r} "
                            f"terminated={terminated!r} truncated={truncated!r} "
                            f"per_agent_stop_reason={_per_agent_stop_reasons(info)!r}",
                            flush=True,
                        )
                    break
                if int(info["steps"]) > int(getattr(env, "max_steps", 300)):
                    raise RuntimeError(
                        f"episode {episode}: exceeded env timeout without ending"
                    )
            if reason is None:
                reason = "max_steps_cap"
            reason_counts[reason] += 1
            false_counts += episode_false_counts
            mask_samples += episode_mask_samples
            fractions = episode_false_counts / max(episode_mask_samples, 1)
            print(
                f"episode={episode + 1:03d} steps={info['steps']:3d} "
                f"seconds_per_step={np.mean(step_seconds):.4f} "
                f"reason={reason:8s} mean_ego_speed={np.mean(speeds):.2f} m/s "
                f"mask_false={dict(zip(MASK_NAMES, fractions.round(3), strict=True))}",
                flush=True,
            )

        print("\nObservation block ranges over all episodes:")
        for name, (minimum, maximum) in block_stats.items():
            print(f"{name:12s} min={minimum:.4f} max={maximum:.4f}")
        flagged = []
        for name in saturation:
            if name.endswith("_total"):
                continue
            fraction = saturation[name] / max(saturation[f"{name}_total"], 1)
            print(f"clipped_fraction[{name}]={fraction:.6f}")
            if fraction > 0.01:
                flagged.append(name)
        print(f"features_above_1pct_saturation={flagged}")
        print("\nTermination reason counts:")
        print(f"{'reason':12s} {'episodes':>8s}")
        for reason, count in reason_counts.items():
            print(f"{reason:12s} {count:8d}")
        print("\nFraction of agent-steps each action was masked:")
        for name, count in zip(MASK_NAMES, false_counts, strict=True):
            print(f"{name:12s} {count / mask_samples:.4f}")
        stuck_fraction = reason_counts["stuck"] / args.episodes
        if stuck_fraction > 0.8:
            print(
                f"*** WARNING: {stuck_fraction:.0%} of episodes ended as stuck; "
                "investigate routes, controls, and termination thresholds."
            )
        completed = sum(reason_counts.values()) == args.episodes
    except BaseException:
        print("STATUS: INCOMPLETE", flush=True)
        raise
    finally:
        if env is not None:
            env.close()
    print("STATUS: COMPLETE" if completed else "STATUS: INCOMPLETE", flush=True)


if __name__ == "__main__":
    main()
