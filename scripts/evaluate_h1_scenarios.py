"""Scripted-policy measurements for one H1 highway profile at a time.

Example:
    python scripts/evaluate_h1_scenarios.py --profile h1_moderate
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import json
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from envs.highway_env import HighwayV2XEnv
from envs.obs_builder import CAND, CANDIDATE_RANGE, EGO, M_CANDIDATES, OBS_DIM, ROUTE
from envs.scenarios.highway_scenarios import load_profiles


POLICIES = ("keep_speed", "random", "accelerate")
PROFILE_IDS = tuple(sorted(load_profiles()))
DEFAULT_SEEDS = tuple(range(100, 110))


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _speed_and_density(env):
    native = env.env.unwrapped
    controlled = set(native.controlled_vehicles)
    vehicles = list(native.road.vehicles)
    background = [vehicle for vehicle in vehicles if vehicle not in controlled]
    gaps = []
    for lane_index in {vehicle.lane_index for vehicle in vehicles}:
        xs = sorted(
            float(vehicle.position[0])
            for vehicle in vehicles
            if vehicle.lane_index == lane_index
        )
        gaps.extend(float(end - start) for start, end in zip(xs, xs[1:]))
    speeds = [float(vehicle.speed) for vehicle in background]
    return {
        "background_vehicles": len(background),
        "mean_same_lane_spacing_m": float(np.mean(gaps)) if gaps else 0.0,
        "min_same_lane_spacing_m": float(min(gaps)) if gaps else 0.0,
        "speed_spread_mps": float(max(speeds) - min(speeds)) if speeds else 0.0,
        "speed_min_mps": float(min(speeds)) if speeds else 0.0,
        "speed_max_mps": float(max(speeds)) if speeds else 0.0,
    }


def _candidate_stats(env, observations):
    occupancy = []
    valid_candidates = []
    padding_ok = True
    shapes_ok = True
    for agent_id in env.agents:
        observation = observations[agent_id]
        rows = observation[EGO + ROUTE : EGO + ROUTE + CAND].reshape(M_CANDIDATES, 6)
        candidates = env.get_candidates(agent_id)
        valid = int(np.count_nonzero(rows[:, 5]))
        occupancy.append(valid)
        valid_candidates.append(len(candidates))
        if valid != min(M_CANDIDATES, len(candidates)):
            padding_ok = False
        if np.any(rows[valid:] != 0.0):
            padding_ok = False
        if observation.shape != (OBS_DIM,) or rows.shape != (M_CANDIDATES, 6):
            shapes_ok = False
    return {
        "occupancy_mean": float(np.mean(occupancy)),
        "valid_candidates_mean": float(np.mean(valid_candidates)),
        "padding_ok": padding_ok,
        "shapes_ok": shapes_ok,
    }


def _relevance_differs(env, agent_id):
    """Evaluate-only check: nearest candidate vs lowest-TTC candidate."""
    candidates = env.get_candidates(agent_id)
    if not candidates:
        return None
    ego = env._vehicle_for_agent(agent_id)
    reverse_ids = {
        env._vehicle_ids[id(vehicle)]: vehicle
        for vehicle in env.env.unwrapped.road.vehicles
    }
    relevant_lanes = {tuple(ego.lane_index), tuple(ego.target_lane_index)}
    nearest = candidates[0]
    best = nearest
    best_ttc = float("inf")
    for candidate in candidates:
        other = reverse_ids.get(candidate["actor_id"])
        if other is None or tuple(other.lane_index) not in relevant_lanes:
            continue
        gap = (
            float(candidate["x"])
            - float(ego.position[0])
            - 0.5 * (float(ego.LENGTH) + float(other.LENGTH))
        )
        closing = float(ego.velocity[0]) - float(candidate["vx"])
        if gap <= 0.0 or closing <= 0.0:
            continue
        ttc = min(gap / closing, 10.0)
        if ttc < best_ttc:
            best_ttc = ttc
            best = candidate
    return best["actor_id"] != nearest["actor_id"]


def _select_action(policy, agent_id, masks, rng):
    if policy == "keep_speed":
        action = 0
    elif policy == "accelerate":
        action = 1
    else:
        action = int(rng.choice(np.flatnonzero(masks[agent_id])))
    invalid = 0 if masks[agent_id][action] else 1
    return action, invalid


def _run_episode(profile, policy, seed, max_steps):
    env = HighwayV2XEnv(
        render_mode=None,
        obs_layout="contract56",
        scenario=profile,
    )
    rng = np.random.default_rng(seed + 100000)
    record = {
        "seed": seed,
        "policy": policy,
        "exception": None,
        "initialization_failed": False,
    }
    try:
        observations, info = env.reset(seed=seed)
        record["density"] = _speed_and_density(env)
        record["candidate_range_m"] = CANDIDATE_RANGE
        speeds = []
        returns = 0.0
        mean_reward_sum = 0.0
        crash_counts = []
        min_ttcs = []
        near_collisions = 0
        team_collisions = 0
        occupancies = []
        valid_candidates = []
        padding_ok = True
        shapes_ok = True
        masks_ok = True
        invalid_actions = 0
        relevance_diffs = 0
        relevance_steps = 0
        terminations = 0
        truncations = 0

        for _ in range(max_steps):
            masks = env.action_masks()
            actions = {}
            for agent_id in env.agents:
                mask = np.asarray(masks[agent_id])
                if mask.shape != (6,) or mask.dtype != bool or not mask.any():
                    masks_ok = False
                action, invalid = _select_action(policy, agent_id, masks, rng)
                actions[agent_id] = action
                invalid_actions += invalid

            observations, rewards, terminated, truncated, info = env.step(actions)
            step_speeds = [
                float(info["per_agent"][agent_id]["speed_mps"])
                for agent_id in env.agents
            ]
            speeds.extend(step_speeds)
            returns += float(info["team_reward"])
            mean_reward_sum += float(np.mean(list(rewards.values())))
            crash_counts.append(
                sum(
                    int(info["per_agent"][agent_id]["collision"])
                    for agent_id in env.agents
                )
            )
            min_ttcs.append(float(info["min_ttc"]))
            near_collisions += int(float(info["min_ttc"]) < 3.0)
            team_collisions += int(bool(info["collision"]))
            stats = _candidate_stats(env, observations)
            occupancies.append(stats["occupancy_mean"])
            valid_candidates.append(stats["valid_candidates_mean"])
            padding_ok = padding_ok and stats["padding_ok"]
            shapes_ok = shapes_ok and stats["shapes_ok"]
            for agent_id in env.agents:
                differs = _relevance_differs(env, agent_id)
                if differs is not None:
                    relevance_steps += 1
                    relevance_diffs += int(differs)

            if all(terminated.values()) or any(truncated.values()):
                terminations += int(all(terminated.values()))
                truncations += int(any(truncated.values()))
                break

        record.update(
            {
                "steps": int(info["steps"]),
                "return_team": returns,
                "return_mean_agent_sum": mean_reward_sum,
                "collision_occurred": bool(info["collision"]),
                "collision_steps": team_collisions,
                "max_crashed_agents": int(max(crash_counts)),
                "min_ttc": float(min(min_ttcs)) if min_ttcs else None,
                "near_collision_steps": near_collisions,
                "mean_cav_speed_mps": float(np.mean(speeds)) if speeds else None,
                "final_cav_speeds_mps": step_speeds,
                "candidate_occupancy_mean": float(np.mean(occupancies)),
                "valid_candidates_mean": float(np.mean(valid_candidates)),
                "observation_shape_ok": shapes_ok,
                "candidate_block_shape": [M_CANDIDATES, 6],
                "padding_ok": padding_ok,
                "invalid_actions": invalid_actions,
                "action_masks_ok": masks_ok,
                "terminations": terminations,
                "truncations": truncations,
                "goal_completion": "N/A (highway-env has no goal)",
                "relevance_differ_agent_steps": relevance_diffs,
                "relevance_total_agent_steps": relevance_steps,
                "relevance_differ_fraction": (
                    relevance_diffs / relevance_steps if relevance_steps else None
                ),
            }
        )
    except Exception as error:
        record["exception"] = f"{type(error).__name__}: {error}"
        record["initialization_failed"] = not record.get("steps")
    finally:
        env.close()
    return record


def _summarize(records):
    numeric_keys = (
        "steps",
        "return_team",
        "return_mean_agent_sum",
        "collision_steps",
        "max_crashed_agents",
        "min_ttc",
        "near_collision_steps",
        "mean_cav_speed_mps",
        "candidate_occupancy_mean",
        "valid_candidates_mean",
        "relevance_differ_fraction",
    )
    summary = {
        "episodes": len(records),
        "exceptions": sum(bool(record["exception"]) for record in records),
        "initialization_failures": sum(
            bool(record["initialization_failed"]) for record in records
        ),
        "collision_episodes": sum(
            bool(record.get("collision_occurred")) for record in records
        ),
        "terminated_episodes": sum(bool(record.get("terminations")) for record in records),
        "truncated_episodes": sum(bool(record.get("truncations")) for record in records),
        "padding_ok_episodes": sum(bool(record.get("padding_ok")) for record in records),
        "masks_ok_episodes": sum(bool(record.get("action_masks_ok")) for record in records),
        "invalid_actions": sum(int(record.get("invalid_actions", 0)) for record in records),
    }
    for key in numeric_keys:
        values = [
            float(record[key])
            for record in records
            if record.get(key) is not None
        ]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else None
        summary[f"{key}_std"] = float(np.std(values)) if len(values) > 1 else None
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=PROFILE_IDS)
    parser.add_argument("--policy", default="all", choices=(*POLICIES, "all"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    policies = POLICIES if args.policy == "all" else (args.policy,)
    if any(seed < 0 for seed in args.seeds) or args.max_steps <= 0:
        parser.error("seeds must be non-negative and --max-steps must be positive")

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_sha = "UNAVAILABLE"

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("scenario_results") / f"h1_{args.profile}_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "metadata": {
            "profile": args.profile,
            "policies": list(policies),
            "seeds": args.seeds,
            "max_steps": args.max_steps,
            "git_sha": git_sha,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "hardware_note": "CPU-only i3 laptop; single process, single thread",
        },
        "per_policy": {},
        "summary": {},
    }
    for policy in policies:
        records = [
            _run_episode(args.profile, policy, seed, args.max_steps)
            for seed in args.seeds
        ]
        result["per_policy"][policy] = records
        result["summary"][policy] = _summarize(records)
        summary = result["summary"][policy]
        steps_mean = summary["steps_mean"]
        return_mean = summary["return_team_mean"]
        print(
            f"{args.profile} {policy}: episodes={summary['episodes']} "
            f"steps={steps_mean if steps_mean is None else round(steps_mean, 1)} "
            f"collisions={summary['collision_episodes']} "
            f"return={return_mean if return_mean is None else round(return_mean, 2)} "
            f"relevance_diff={summary['relevance_differ_fraction_mean']} "
            f"exceptions={summary['exceptions']}",
            flush=True,
        )

    output.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(f"output={output}", flush=True)
    print("STATUS: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
