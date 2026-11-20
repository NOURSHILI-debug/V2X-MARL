"""Run G1-G6 gates for one configured Highway scenario.

Example:
    python scripts/evaluate_scenario_gate.py --scenario h1_forcing
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from envs.highway_env import HighwayV2XEnv
from envs.obs_builder import CAND, CAND_FEATURES, EGO, M_CANDIDATES, OBS_DIM, ROUTE
from envs.scenarios.highway_scenarios import load_profiles
from scripts.scripted_policies import choose_actions

DEFAULT_SEEDS = tuple(range(100, 110))
G5_HEADROOM_THRESHOLD = 0.05
HAZARD_TTC_S = 5.0
MAX_TTC_S = 10.0
VOI_CONDITIONS = ("none", "nearest2", "random2", "oracle2", "all6")
_CANDIDATE_START = EGO + ROUTE


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


def _risk_info(env, agent_id):
    candidates = env.get_candidates(agent_id)
    if not candidates:
        return {"fallback": True, "hazard": False, "top2_miss": False, "min_ttc": None}
    ego = env._vehicle_for_agent(agent_id)
    reverse = {
        env._vehicle_ids[id(vehicle)]: vehicle
        for vehicle in env.env.unwrapped.road.vehicles
    }
    relevant_lanes = {tuple(ego.lane_index), tuple(ego.target_lane_index)}
    scored = []
    for rank, candidate in enumerate(candidates):
        other = reverse.get(candidate["actor_id"])
        ttc = None
        if other is not None and tuple(other.lane_index) in relevant_lanes:
            gap = (
                float(candidate["x"])
                - float(ego.position[0])
                - 0.5 * (float(ego.LENGTH) + float(other.LENGTH))
            )
            closing = float(ego.velocity[0]) - float(candidate["vx"])
            if gap > 0.0 and closing > 0.0:
                ttc = min(gap / closing, MAX_TTC_S)
        scored.append({"rank": rank, "actor_id": candidate["actor_id"], "ttc": ttc})
    closers = [item for item in scored if item["ttc"] is not None]
    if not closers:
        return {"fallback": True, "hazard": False, "top2_miss": False, "min_ttc": None}
    best = min(closers, key=lambda item: (item["ttc"], item["rank"]))
    hazard = best["ttc"] < HAZARD_TTC_S
    return {
        "fallback": False,
        "hazard": hazard,
        "top2_miss": bool(hazard and best["rank"] >= 2),
        "min_ttc": float(best["ttc"]),
    }


def _candidate_stats(observations):
    padding_ok = True
    shapes_ok = True
    occupancy = []
    for observation in observations.values():
        rows = observation[_CANDIDATE_START : _CANDIDATE_START + CAND].reshape(
            M_CANDIDATES, CAND_FEATURES
        )
        valid = int(np.count_nonzero(rows[:, 5]))
        occupancy.append(valid)
        if np.any(rows[valid:] != 0.0):
            padding_ok = False
        if observation.shape != (OBS_DIM,) or rows.shape != (M_CANDIDATES, CAND_FEATURES):
            shapes_ok = False
    return float(np.mean(occupancy)), padding_ok, shapes_ok


def _oracle_ranks(env, agent_id, valid_indices):
    candidates = env.get_candidates(agent_id)[: len(valid_indices)]
    risks = []
    for rank, candidate in enumerate(candidates):
        ego = env._vehicle_for_agent(agent_id)
        reverse = {
            env._vehicle_ids[id(vehicle)]: vehicle
            for vehicle in env.env.unwrapped.road.vehicles
        }
        other = reverse.get(candidate["actor_id"])
        ttc = MAX_TTC_S
        if other is not None:
            gap = (
                float(candidate["x"])
                - float(ego.position[0])
                - 0.5 * (float(ego.LENGTH) + float(other.LENGTH))
            )
            closing = float(ego.velocity[0]) - float(candidate["vx"])
            if gap > 0.0 and closing > 0.0:
                ttc = min(gap / closing, MAX_TTC_S)
        risks.append((ttc, rank, candidate["actor_id"]))
    ordered = sorted(risks, key=lambda item: (item[0], item[1]))[:2]
    keep = [valid_indices[item[1]] for item in ordered]
    for index in valid_indices:
        if len(keep) >= 2:
            break
        if index not in keep:
            keep.append(index)
    return keep


def _restrict_observations(observations, condition, env, rng):
    restricted = {}
    for agent_id, observation in observations.items():
        result = np.array(observation, copy=True)
        rows = result[_CANDIDATE_START : _CANDIDATE_START + CAND].reshape(
            M_CANDIDATES, CAND_FEATURES
        )
        valid = [index for index in range(M_CANDIDATES) if rows[index, 5] > 0.0]
        if condition == "none":
            keep = []
        elif condition == "nearest2":
            keep = valid[:2]
        elif condition == "random2":
            selected = rng.choice(valid, size=min(2, len(valid)), replace=False) if valid else []
            keep = [int(index) for index in selected]
        elif condition == "oracle2":
            keep = _oracle_ranks(env, agent_id, valid)
        elif condition == "all6":
            keep = valid
        else:
            raise ValueError(f"unknown VOI condition '{condition}'.")
        for index in range(M_CANDIDATES):
            if index not in keep:
                rows[index] = 0.0
        restricted[agent_id] = result
    return restricted


def _run_episode(scenario_id, policy, condition, seed, max_steps):
    env = HighwayV2XEnv(render_mode=None, obs_layout="contract56", scenario=scenario_id)
    rng = np.random.default_rng(seed + 100000)
    record = {
        "seed": seed,
        "policy": policy,
        "condition": condition,
        "exception": None,
        "initialization_failed": False,
    }
    try:
        observations, _ = env.reset(seed=seed)
        speeds = []
        occupancies = []
        padding_ok = True
        shapes_ok = True
        masks_ok = True
        invalid_actions = 0
        fallback_steps = 0
        hazard_steps = 0
        top2_miss_steps = 0
        total_agent_steps = 0
        collision_steps = 0
        near_collision_steps = 0
        min_ttcs = []
        returns = 0.0
        terminations = 0
        truncations = 0
        for _ in range(max_steps):
            masks = env.action_masks()
            for mask in masks.values():
                if np.asarray(mask).shape != (6,) or not np.asarray(mask).any():
                    masks_ok = False
            if condition == "all6":
                policy_observations = observations
            else:
                policy_observations = _restrict_observations(
                    observations, condition, env, rng
                )
            actions = choose_actions(policy, policy_observations, masks, rng)
            for agent_id, action in actions.items():
                if not masks[agent_id][action]:
                    invalid_actions += 1
            if policy == "competent":
                occupancy, padding, shapes = _candidate_stats(observations)
                occupancies.append(occupancy)
                padding_ok = padding_ok and padding
                shapes_ok = shapes_ok and shapes
                for agent_id in env.agents:
                    risk = _risk_info(env, agent_id)
                    total_agent_steps += 1
                    fallback_steps += int(risk["fallback"])
                    hazard_steps += int(risk["hazard"])
                    top2_miss_steps += int(risk["top2_miss"])
            observations, _rewards, terminated, truncated, info = env.step(actions)
            step_speeds = [
                float(info["per_agent"][agent_id]["speed_mps"]) for agent_id in env.agents
            ]
            speeds.extend(step_speeds)
            returns += float(info["team_reward"])
            collision_steps += int(bool(info["collision"]))
            near_collision_steps += int(float(info["min_ttc"]) < 3.0)
            min_ttcs.append(float(info["min_ttc"]))
            if all(terminated.values()) or any(truncated.values()):
                terminations += int(all(terminated.values()))
                truncations += int(any(truncated.values()))
                break
        record.update(
            {
                "steps": int(info["steps"]),
                "collision_occurred": bool(info["collision"]),
                "collision_steps": collision_steps,
                "return_team": returns,
                "mean_speed_mps": float(np.mean(speeds)) if speeds else None,
                "final_speeds_mps": step_speeds,
                "min_ttc": float(min(min_ttcs)) if min_ttcs else None,
                "near_collision_steps": near_collision_steps,
                "candidate_occupancy_mean": float(np.mean(occupancies))
                if occupancies
                else None,
                "padding_ok": padding_ok,
                "observation_shape_ok": shapes_ok,
                "action_masks_ok": masks_ok,
                "invalid_actions": invalid_actions,
                "g5_fallback_steps": fallback_steps,
                "g5_hazard_steps": hazard_steps,
                "g5_top2_miss_steps": top2_miss_steps,
                "g5_total_agent_steps": total_agent_steps,
                "g5_fallback_fraction": (
                    fallback_steps / total_agent_steps if total_agent_steps else None
                ),
                "g5_headroom": (
                    top2_miss_steps / hazard_steps if hazard_steps else None
                ),
                "terminations": terminations,
                "truncations": truncations,
                "goal_completion": "N/A (highway-env has no goal)",
            }
        )
    except Exception as error:
        record["exception"] = f"{type(error).__name__}: {error}"
        record["initialization_failed"] = not bool(record.get("steps"))
    finally:
        env.close()
    return record


def _rate(records, key):
    return float(np.mean([bool(record[key]) for record in records]))


def _summarize(scenario_id, seeds, max_steps):
    episodes = {}
    for policy, condition in (
        ("keep_speed", "all6"),
        ("random", "all6"),
        ("accelerate", "all6"),
        ("competent", "all6"),
        ("competent", "none"),
        ("competent", "nearest2"),
        ("competent", "random2"),
        ("competent", "oracle2"),
    ):
        key = f"{policy}:{condition}"
        episodes[key] = [
            _run_episode(scenario_id, policy, condition, seed, max_steps)
            for seed in seeds
        ]

    keep = episodes["keep_speed:all6"]
    competent = episodes["competent:all6"]
    random = episodes["random:all6"]
    hazard_steps = sum(record["g5_hazard_steps"] for record in competent)
    miss_steps = sum(record["g5_top2_miss_steps"] for record in competent)
    fallback_steps = sum(record["g5_fallback_steps"] for record in competent)
    total_agent_steps = sum(record["g5_total_agent_steps"] for record in competent)

    per_condition = {
        condition: {
            "collision_rate": _rate(episodes[f"competent:{condition}"], "collision_occurred"),
            "mean_speed_mps": float(
                np.mean(
                    [
                        record["mean_speed_mps"]
                        for record in episodes[f"competent:{condition}"]
                        if record["mean_speed_mps"] is not None
                    ]
                )
            ),
            "per_seed": [
                {
                    "seed": record["seed"],
                    "collision": int(record["collision_occurred"]),
                    "steps": record["steps"],
                    "mean_speed_mps": record["mean_speed_mps"],
                }
                for record in episodes[f"competent:{condition}"]
            ],
        }
        for condition in VOI_CONDITIONS
    }
    oracle = per_condition["oracle2"]
    nearest = per_condition["nearest2"]
    none = per_condition["none"]
    all6 = per_condition["all6"]

    g1_rate = _rate(keep, "collision_occurred")
    g2_success = float(
        np.mean(
            [
                not record["collision_occurred"] and record["steps"] == max_steps
                for record in competent
            ]
        )
    )
    g3_median = float(statistics.median(record["steps"] for record in random))
    g5_headroom = miss_steps / hazard_steps if hazard_steps else 0.0
    g5_fallback = fallback_steps / total_agent_steps if total_agent_steps else 0.0
    g6_pass = bool(
        oracle["collision_rate"] < nearest["collision_rate"]
        and none["collision_rate"] > all6["collision_rate"]
    )

    gate_summary = {
        "G1": {
            "value": g1_rate,
            "threshold": ">= 0.50 collision rate",
            "pass": bool(g1_rate >= 0.50),
            "per_seed": [int(record["collision_occurred"]) for record in keep],
        },
        "G2": {
            "value": g2_success,
            "threshold": ">= 0.80 collision-free 300-step rate",
            "pass": bool(g2_success >= 0.80),
            "per_seed": [
                int(not record["collision_occurred"] and record["steps"] == max_steps)
                for record in competent
            ],
        },
        "G3": {
            "value": g3_median,
            "threshold": ">= 5 steps median survival",
            "pass": bool(g3_median >= 5.0),
            "per_seed": [record["steps"] for record in random],
        },
        "G4": {
            "value": g2_success,
            "threshold": "no unavoidable reset collisions (shown by G2)",
            "pass": bool(g2_success >= 0.80),
            "per_seed": [int(record["initialization_failed"]) for record in competent],
        },
        "G5": {
            "value": g5_headroom,
            "fallback_fraction": g5_fallback,
            "hazard_agent_steps": hazard_steps,
            "threshold": f"> {G5_HEADROOM_THRESHOLD} top-2 miss on hazard agent-steps",
            "pass": bool(g5_headroom > G5_HEADROOM_THRESHOLD),
            "per_seed": [
                record["g5_headroom"] if record["g5_headroom"] is not None else None
                for record in competent
            ],
        },
        "G6": {
            "value": {
                "oracle_collision": oracle["collision_rate"],
                "nearest_collision": nearest["collision_rate"],
                "none_collision": none["collision_rate"],
                "all6_collision": all6["collision_rate"],
            },
            "threshold": "oracle < nearest and none > all6 collision rate",
            "pass": g6_pass,
            "per_seed": {
                condition: [
                    item["collision"] for item in per_condition[condition]["per_seed"]
                ]
                for condition in VOI_CONDITIONS
            },
        },
    }
    return episodes, per_condition, gate_summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=sorted(load_profiles()))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if len(args.seeds) < 10:
        parser.error("gates require at least 10 episodes/seeds")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "UNAVAILABLE"
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("results/raw/scenarios") / f"{args.scenario}_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    episodes, per_condition, gates = _summarize(
        args.scenario, args.seeds, args.max_steps
    )
    result = {
        "metadata": {
            "scenario": args.scenario,
            "seeds": args.seeds,
            "max_steps": args.max_steps,
            "git_sha": git_sha,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "hardware_note": "CPU-only i3 laptop; single process, single thread",
            "g5_headroom_threshold": G5_HEADROOM_THRESHOLD,
        },
        "episodes": episodes,
        "voi": per_condition,
        "gates": gates,
    }
    output.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")

    print(f"GATE TABLE {args.scenario} seeds={args.seeds[0]}-{args.seeds[-1]}")
    for gate in ("G1", "G2", "G3", "G4", "G5", "G6"):
        item = gates[gate]
        print(
            f"{gate} value={item['value']} threshold={item['threshold']} "
            f"verdict={'PASS' if item['pass'] else 'FAIL'} per_seed={item['per_seed']}"
        )
    for condition in VOI_CONDITIONS:
        data = per_condition[condition]
        print(
            f"VOI {condition}: collision_rate={data['collision_rate']:.2f} "
            f"mean_speed={data['mean_speed_mps']:.2f}"
        )
    print(f"RAW={output}")
    print("STATUS: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
