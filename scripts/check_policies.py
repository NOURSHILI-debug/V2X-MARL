"""Compare simple CARLA control policies over the same fixed episode seeds.

Run with CARLA 0.9.16 already running:
    python scripts/check_policies.py
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

POLICY_NAMES = ("scripted_drive_on", "always_accelerate", "random_valid")


def _per_agent_stop_reasons(info):
    per_agent = info.get("per_agent", {})
    return {
        agent_id: details.get("stop_reason", "none")
        for agent_id, details in per_agent.items()
    }


def _termination_reason(info, terminated, truncated):
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


def _policy_action(policy_name, agent_id, masks, rng):
    if policy_name == "scripted_drive_on":
        return 0
    if policy_name == "always_accelerate":
        return 1
    return int(rng.choice(np.flatnonzero(masks[agent_id])))


def _run_policy(env, policy_name, episode_seeds, max_steps):
    rng = np.random.default_rng(2026)
    speed_samples = []
    episode_steps = []
    reason_counts = {
        "collision": 0,
        "goal": 0,
        "off_road": 0,
        "stuck": 0,
        "timeout": 0,
        "max_steps_cap": 0,
        "unknown": 0,
    }
    goal_episode_count = 0
    arrived_agents = 0
    agent_count = 0
    arrival_steps = []
    unfinished_stop_reasons = {name: 0 for name in ("red_light", "blocked", "stuck", "none")}
    collision_count = 0
    seconds_per_step = []

    for seed in episode_seeds:
        env.reset(seed=seed)
        agent_count += len(env.agents)
        episode_step_seconds = []
        episode_reason = None
        info = None
        for _ in range(max_steps):
            masks = env.action_masks()
            actions = {
                agent_id: _policy_action(policy_name, agent_id, masks, rng)
                for agent_id in env.agents
            }
            started = time.perf_counter()
            _, _, terminated, truncated, info = env.step(actions)
            elapsed = time.perf_counter() - started
            seconds_per_step.append(elapsed)
            episode_step_seconds.append(elapsed)
            speed_samples.extend(
                float(vehicle.get_velocity().length())
                for vehicle in env.vehicles.values()
            )
            if all(terminated.values()) or any(truncated.values()):
                steps = int(info["steps"])
                reason = _termination_reason(
                    info,
                    all(terminated.values()),
                    any(truncated.values()),
                )
                episode_steps.append(steps)
                reason_counts[reason] += 1
                episode_reason = reason
                episode_arrivals = [
                    details.get("arrival_step")
                    for details in info.get("per_agent", {}).values()
                    if details.get("arrival_step") is not None
                ]
                arrived_agents += len(episode_arrivals)
                arrival_steps.extend(episode_arrivals)
                for details in info.get("per_agent", {}).values():
                    if not details.get("goal_reached", False):
                        reason = details.get("stop_reason", "none")
                        unfinished_stop_reasons[reason if reason in unfinished_stop_reasons else "none"] += 1
                goal_episode_count += int(bool(info["goal_reached"]))
                collision_count += int(bool(info["collision"]))
                if reason == "unknown":
                    print(
                        f"UNKNOWN_END policy={policy_name} seed={seed} "
                        f"info={info!r} terminated={terminated!r} "
                        f"truncated={truncated!r} "
                        f"per_agent_stop_reason={_per_agent_stop_reasons(info)!r}",
                        flush=True,
                    )
                print(
                    f"policy={policy_name} episode={seed + 1:02d}/"
                    f"{len(episode_seeds)} steps={steps} "
                    f"seconds_per_step={np.mean(episode_step_seconds):.4f} reason={reason}",
                    flush=True,
                )
                break
            if int(info["steps"]) > int(getattr(env, "max_steps", 300)):
                raise RuntimeError(
                    f"{policy_name}, seed {seed}: exceeded env timeout without ending"
                )
        if episode_reason is None:
            steps = int(info["steps"])
            episode_steps.append(steps)
            reason_counts["max_steps_cap"] += 1
            episode_arrivals = [
                details.get("arrival_step")
                for details in info.get("per_agent", {}).values()
                if details.get("arrival_step") is not None
            ]
            arrived_agents += len(episode_arrivals)
            arrival_steps.extend(episode_arrivals)
            for details in info.get("per_agent", {}).values():
                if not details.get("goal_reached", False):
                    reason = details.get("stop_reason", "none")
                    unfinished_stop_reasons[reason if reason in unfinished_stop_reasons else "none"] += 1
            goal_episode_count += int(bool(info["goal_reached"]))
            collision_count += int(bool(info["collision"]))
            if steps > int(getattr(env, "max_steps", 300)):
                raise RuntimeError(
                    f"{policy_name}, seed {seed}: exceeded env timeout without ending"
                )
            print(
                f"policy={policy_name} episode={seed + 1:02d}/{len(episode_seeds)} "
                f"steps={steps} seconds_per_step={np.mean(episode_step_seconds):.4f} "
                "reason=max_steps_cap",
                flush=True,
            )
        if info is None:
            raise RuntimeError(
                f"{policy_name}, seed {seed}: no environment step was executed"
            )

    return {
        "mean_speed": float(np.mean(speed_samples)) if speed_samples else 0.0,
        "mean_steps": float(np.mean(episode_steps)) if episode_steps else 0.0,
        "reason_counts": reason_counts,
        "goal_episodes": goal_episode_count,
        "mean_arrival_step": float(np.mean(arrival_steps)) if arrival_steps else 0.0,
        "arrived_agent_fraction": arrived_agents / agent_count if agent_count else 0.0,
        "collisions": collision_count,
        "seconds_per_step": float(np.mean(seconds_per_step)) if seconds_per_step else 0.0,
        "unfinished_stop_reasons": unfinished_stop_reasons,
        "arrival_steps_min": int(min(arrival_steps)) if arrival_steps else None,
        "arrival_steps_median": float(np.median(arrival_steps)) if arrival_steps else None,
        "arrival_steps_max": int(max(arrival_steps)) if arrival_steps else None,
        "arrival_steps_above_250": sum(step > 250 for step in arrival_steps),
        "arrival_step_count": len(arrival_steps),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("--episodes and --max-steps must be positive")
    seeds = tuple(range(args.episodes))
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "UNAVAILABLE"
    env = None
    completed = False
    try:
        from envs.carla_env import CarlaMultiAgentEnv

        env = CarlaMultiAgentEnv(town="Town03")
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
        results = {
            policy_name: _run_policy(env, policy_name, seeds, args.max_steps)
            for policy_name in POLICY_NAMES
        }
        completed = True
    except BaseException:
        print("STATUS: INCOMPLETE", flush=True)
        raise
    finally:
        if env is not None:
            env.close()

    for policy_name, result in results.items():
        print(f"\n{policy_name} ({args.episodes} episodes)")
        print(f"  mean ego speed:   {result['mean_speed']:.2f} m/s")
        print(f"  mean steps:       {result['mean_steps']:.1f}")
        print(f"  seconds per step: {result['seconds_per_step']:.4f}")
        print(f"  termination counts: {result['reason_counts']}")
        print(f"  goal-reached episodes: {result['goal_episodes']}")
        print(f"  mean arrival step:     {result['mean_arrival_step']:.1f}")
        print(f"  arrived agent fraction: {result['arrived_agent_fraction']:.3f}")
        print(f"  per-agent arrival rate: {result['arrived_agent_fraction']:.3f}")
        print(f"  unfinished stop reasons: {result['unfinished_stop_reasons']}")
        print(
            "  arrival steps (min/median/max/count>250/count): "
            f"{result['arrival_steps_min']}/{result['arrival_steps_median']}/"
            f"{result['arrival_steps_max']}/{result['arrival_steps_above_250']}/"
            f"{result['arrival_step_count']}"
        )
        print(f"  collision episodes:    {result['collisions']}")
    print("STATUS: COMPLETE" if completed else "STATUS: INCOMPLETE", flush=True)


if __name__ == "__main__":
    main()
