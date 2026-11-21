"""Milestone 8A: paired CARLA evaluation runner for observation-conditioned heuristic.

Runs nearest_k vs random_k with the same deterministic
ObservationHeuristicController, identical environment/V2X/spawn configuration,
and paired seeds. Saves episode-level and aggregate JSON under results/milestone8a/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from envs.carla_env import CarlaMultiAgentEnv
from envs.v2x_channel import V2XConfig
from policies.observation_heuristic import ObservationHeuristicController

RESULTS_DIR = REPO_ROOT / "results" / "milestone8a"


def _finite_or_none(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def make_env(selection_mode: str, env_seed: int, selection_seed: int | None, v2x_seed: int,
             host: str, port: int, tm_port: int) -> CarlaMultiAgentEnv:
    return CarlaMultiAgentEnv(
        town="Town03",
        n_agents=4,
        host=host,
        port=port,
        tm_port=tm_port,
        seed=env_seed,
        selection_mode=selection_mode,
        selection_k=2,
        selection_seed=selection_seed,
        candidate_source="v2x",
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
        v2x_seed=v2x_seed,
        v2x_broadcasters="cavs_and_background",
        spawn_mode="cluster",
        spawn_cluster_max_distance=45.0,
    )


def run_episode(env: CarlaMultiAgentEnv, controller: ObservationHeuristicController,
                reset_seed: int, max_steps: int | None = None) -> dict[str, Any]:
    obs, reset_info = env.reset(seed=reset_seed)
    agents = list(env.agents)
    step_limit = int(max_steps) if max_steps is not None else int(env.max_steps)

    step_records: list[dict[str, Any]] = []
    team_reward_total = 0.0
    total_sent = 0
    total_delivered = 0
    total_dropped_range = 0
    total_dropped_loss = 0
    selected_ids_counter: Counter[str] = Counter()
    pool_sizes: list[int] = []
    missing_slots_values: list[int] = []
    received_counts: list[int] = []
    obs_violations = 0
    nan_inf_obs = 0
    arrival_steps: list[int] = []
    ended = False
    final_info: dict[str, Any] = {}

    for _ in range(step_limit):
        actions = {aid: int(controller.act(obs[aid])) for aid in agents}
        obs, rewards, terminated, truncated, info = env.step(actions)
        final_info = info
        v2x = info.get("v2x") or {}
        step_sent = int(v2x.get("sent", 0) or 0)
        step_delivered = int(v2x.get("delivered", 0) or 0)
        step_dropped_range = int(v2x.get("dropped_by_range", 0) or 0)
        step_dropped_loss = int(v2x.get("dropped_by_loss", 0) or 0)
        total_sent += step_sent
        total_delivered += step_delivered
        total_dropped_range += step_dropped_range
        total_dropped_loss += step_dropped_loss
        team_reward_total += float(info.get("team_reward", 0.0))

        per_agent_step = {}
        for aid in agents:
            arr = np.asarray(obs[aid])
            shape_ok = arr.shape == (56,)
            dtype_ok = str(arr.dtype) == "float32"
            finite_ok = bool(np.isfinite(arr).all())
            if not (shape_ok and dtype_ok and finite_ok):
                obs_violations += 1
            if not finite_ok:
                nan_inf_obs += 1
            pa = info["per_agent"][aid]
            sel = pa.get("selection") or {}
            recv = pa.get("v2x_received") or {}
            pool = sel.get("pool_size")
            miss = sel.get("missing_slots")
            sids = sel.get("selected_ids") or []
            if pool is not None:
                pool_sizes.append(int(pool))
            if miss is not None:
                missing_slots_values.append(int(miss))
            received_counts.append(int(recv.get("count", 0) or 0))
            for sid in sids:
                if sid is not None:
                    selected_ids_counter[str(sid)] += 1
            per_agent_step[aid] = {
                "reward": float(rewards.get(aid, 0.0)),
                "goal_reached": bool(pa.get("goal_reached", False)),
                "collision": bool(pa.get("collision", False)),
                "off_road": bool(pa.get("off_road", False)),
                "stuck": bool(pa.get("stuck", False)),
                "stop_reason": pa.get("stop_reason"),
                "arrival_step": pa.get("arrival_step"),
                "received_count": int(recv.get("count", 0) or 0),
                "received_sender_ids": recv.get("sender_ids", []),
                "received_ages": recv.get("ages", []),
                "pool_size": pool,
                "selected_ids": sids,
                "missing_slots": miss,
                "obs_shape": list(arr.shape),
                "obs_dtype": str(arr.dtype),
                "obs_finite": finite_ok,
            }

        step_records.append({
            "step": int(info.get("steps", len(step_records) + 1)),
            "v2x": {
                "step_index": v2x.get("step_index"),
                "sent": step_sent,
                "delivered": step_delivered,
                "dropped_by_range": step_dropped_range,
                "dropped_by_loss": step_dropped_loss,
                "in_flight": v2x.get("in_flight"),
                "episode": v2x.get("episode"),
            },
            "team_reward": float(info.get("team_reward", 0.0)),
            "per_agent": per_agent_step,
        })

        if all(terminated.values()) or any(truncated.values()):
            ended = True
            break

    # Collect arrivals
    for aid in agents:
        arr_step = (final_info.get("per_agent") or {}).get(aid, {}).get("arrival_step")
        if arr_step is not None:
            arrival_steps.append(int(arr_step))

    episode_v2x = (final_info.get("v2x") or {}).get("episode") or {}
    result = {
        "reset_seed": int(reset_seed),
        "ended": ended,
        "steps": int(final_info.get("steps", len(step_records))),
        "success": bool(final_info.get("success", False)) if final_info else False,
        "collision": bool(final_info.get("collision", False)) if final_info else False,
        "goal_reached": bool(final_info.get("goal_reached", False)) if final_info else False,
        "off_road": bool(final_info.get("off_road", False)) if final_info else False,
        "stuck": bool(final_info.get("stuck", False)) if final_info else False,
        "ending_reason": final_info.get("ending_reason"),
        "team_reward_total": float(team_reward_total),
        "team_reward_mean_step": float(team_reward_total / max(1, len(step_records))),
        "arrival_steps": arrival_steps,
        "arrival_step_mean": _mean([float(x) for x in arrival_steps]),
        "min_ttc": _finite_or_none(final_info.get("min_ttc")),
        "jerk": _finite_or_none(final_info.get("jerk")),
        "v2x": {
            "total_sent": total_sent,
            "total_delivered": total_delivered,
            "total_dropped_by_range": total_dropped_range,
            "total_dropped_by_loss": total_dropped_loss,
            "delivery_ratio": (total_delivered / total_sent) if total_sent else None,
            "mean_received_per_agent_step": _mean([float(x) for x in received_counts]),
            "episode_cumulative": episode_v2x,
        },
        "selection": {
            "mean_pool_size": _mean([float(x) for x in pool_sizes]),
            "mean_missing_slots": _mean([float(x) for x in missing_slots_values]),
            "selected_id_counts": dict(selected_ids_counter),
        },
        "observation_validity": {
            "steps_with_violations": obs_violations,
            "steps_with_non_finite": nan_inf_obs,
            "all_shape_56": obs_violations == 0,
        },
        "spawn": reset_info,
        "step_records": step_records,
    }
    return result


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {}
    def rate(key): return sum(1 for e in episodes if e.get(key)) / len(episodes)
    return {
        "n_episodes": len(episodes),
        "success_rate": rate("success"),
        "collision_rate": rate("collision"),
        "goal_reached_rate": rate("goal_reached"),
        "off_road_rate": rate("off_road"),
        "stuck_rate": rate("stuck"),
        "timeout_rate": sum(1 for e in episodes if e.get("ending_reason") == "timeout") / len(episodes),
        "mean_episode_length": _mean([float(e["steps"]) for e in episodes]),
        "mean_team_reward_total": _mean([float(e["team_reward_total"]) for e in episodes]),
        "mean_team_reward_per_step": _mean([float(e["team_reward_mean_step"]) for e in episodes]),
        "mean_arrival_step": _mean([float(e["arrival_step_mean"]) for e in episodes if e.get("arrival_step_mean") is not None]),
        "mean_min_ttc": _mean([float(e["min_ttc"]) for e in episodes if e.get("min_ttc") is not None]),
        "mean_jerk": _mean([float(e["jerk"]) for e in episodes if e.get("jerk") is not None]),
        "mean_v2x_delivery_ratio": _mean([float(e["v2x"]["delivery_ratio"]) for e in episodes if e["v2x"].get("delivery_ratio") is not None]),
        "mean_pool_size": _mean([float(e["selection"]["mean_pool_size"]) for e in episodes if e["selection"].get("mean_pool_size") is not None]),
        "mean_missing_slots": _mean([float(e["selection"]["mean_missing_slots"]) for e in episodes if e["selection"].get("mean_missing_slots") is not None]),
        "all_observations_valid": all(e["observation_validity"]["steps_with_violations"] == 0 for e in episodes),
    }


def paired_run(selection_modes: list[str], seeds: list[int], episodes_per_seed: int,
               max_steps: int | None, host: str, port: int, tm_port: int,
               base_selection_seed: int, base_v2x_seed: int) -> dict[str, Any]:
    controller = ObservationHeuristicController()  # deterministic, stateless logic
    results = {"config": {
        "selection_modes": selection_modes,
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "max_steps": max_steps,
        "town": "Town03",
        "n_agents": 4,
        "spawn_mode": "cluster",
        "spawn_cluster_max_distance": 45.0,
        "candidate_source": "v2x",
        "communication_range_m": 50.0,
        "candidate_observation_range_m": 50.0,
        "packet_loss": 0.0,
        "delay_steps": 0,
        "selection_k": 2,
        "controller": "ObservationHeuristicController",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }, "episodes": [], "aggregates": {}, "paired": []}

    for mode in selection_modes:
        for seed in seeds:
            for ep in range(episodes_per_seed):
                reset_seed = int(seed) + ep  # deterministic episode derivation, documented
                selection_seed = base_selection_seed + int(seed)
                v2x_seed = base_v2x_seed + int(seed)
                print(f"Running mode={mode} seed={seed} episode={ep} reset_seed={reset_seed}", flush=True)
                env = make_env(mode, seed, selection_seed, v2x_seed, host, port, tm_port)
                try:
                    ep_result = run_episode(env, controller, reset_seed=reset_seed, max_steps=max_steps)
                finally:
                    env.close()
                ep_result.update({
                    "selection_mode": mode,
                    "base_seed": int(seed),
                    "episode_index": ep,
                    "selection_seed": selection_seed,
                    "v2x_seed": v2x_seed,
                })
                # drop bulky step records from in-memory aggregate set? keep in file
                results["episodes"].append(ep_result)
                print(f"  -> steps={ep_result['steps']} success={ep_result['success']} "
                      f"ending={ep_result['ending_reason']} reward={ep_result['team_reward_total']:.3f}", flush=True)

    for mode in selection_modes:
        mode_eps = [e for e in results["episodes"] if e["selection_mode"] == mode]
        # store aggregates without step_records duplication handled at save time
        results["aggregates"][mode] = aggregate(mode_eps)

    # Paired comparison by (base_seed, episode_index)
    if "nearest_k" in selection_modes and "random_k" in selection_modes:
        idx = {(e["selection_mode"], e["base_seed"], e["episode_index"]): e for e in results["episodes"]}
        for seed in seeds:
            for ep in range(episodes_per_seed):
                a = idx.get(("nearest_k", seed, ep))
                b = idx.get(("random_k", seed, ep))
                if not a or not b:
                    continue
                results["paired"].append({
                    "base_seed": seed,
                    "episode_index": ep,
                    "nearest_k": {k: a[k] for k in ("steps","success","collision","off_road","stuck","ending_reason","team_reward_total","team_reward_mean_step","min_ttc","jerk")},
                    "random_k": {k: b[k] for k in ("steps","success","collision","off_road","stuck","ending_reason","team_reward_total","team_reward_mean_step","min_ttc","jerk")},
                    "delta_nearest_minus_random": {
                        "steps": a["steps"] - b["steps"],
                        "team_reward_total": a["team_reward_total"] - b["team_reward_total"],
                        "success": int(a["success"]) - int(b["success"]),
                        "collision": int(a["collision"]) - int(b["collision"]),
                    },
                })

    return results


def save_results(results: dict[str, Any], out_dir: Path, include_step_records: bool = True) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Full episode results (optionally with step records)
    episodes_out = []
    for e in results["episodes"]:
        e2 = dict(e)
        if not include_step_records:
            e2.pop("step_records", None)
        episodes_out.append(e2)
    ep_path = out_dir / "episode_results.json"
    with ep_path.open("w", encoding="utf-8") as f:
        json.dump(episodes_out, f, indent=2, default=_json_default)
    paths["episode_results"] = str(ep_path)

    # Compact episode summary JSON/CSV-friendly
    summary_rows = []
    for e in results["episodes"]:
        summary_rows.append({
            "selection_mode": e["selection_mode"],
            "base_seed": e["base_seed"],
            "episode_index": e["episode_index"],
            "reset_seed": e["reset_seed"],
            "steps": e["steps"],
            "success": e["success"],
            "collision": e["collision"],
            "goal_reached": e["goal_reached"],
            "off_road": e["off_road"],
            "stuck": e["stuck"],
            "ending_reason": e["ending_reason"],
            "team_reward_total": e["team_reward_total"],
            "team_reward_mean_step": e["team_reward_mean_step"],
            "arrival_step_mean": e.get("arrival_step_mean"),
            "min_ttc": e.get("min_ttc"),
            "jerk": e.get("jerk"),
            "v2x_sent": e["v2x"]["total_sent"],
            "v2x_delivered": e["v2x"]["total_delivered"],
            "v2x_dropped_range": e["v2x"]["total_dropped_by_range"],
            "v2x_dropped_loss": e["v2x"]["total_dropped_by_loss"],
            "v2x_delivery_ratio": e["v2x"]["delivery_ratio"],
            "mean_received_per_agent_step": e["v2x"]["mean_received_per_agent_step"],
            "mean_pool_size": e["selection"]["mean_pool_size"],
            "mean_missing_slots": e["selection"]["mean_missing_slots"],
            "obs_violations": e["observation_validity"]["steps_with_violations"],
        })
    summary_path = out_dir / "episode_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, default=_json_default)
    paths["episode_summary"] = str(summary_path)

    # CSV
    import csv
    csv_path = out_dir / "episode_summary.csv"
    if summary_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        paths["episode_summary_csv"] = str(csv_path)

    agg_path = out_dir / "aggregate_results.json"
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump({"config": results["config"], "aggregates": results["aggregates"]}, f, indent=2, default=_json_default)
    paths["aggregate_results"] = str(agg_path)

    paired_path = out_dir / "paired_comparison.json"
    with paired_path.open("w", encoding="utf-8") as f:
        json.dump(results["paired"], f, indent=2, default=_json_default)
    paths["paired_comparison"] = str(paired_path)

    config_path = out_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(results["config"], f, indent=2, default=_json_default)
    paths["config"] = str(config_path)

    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", default=["nearest_k", "random_k"])
    p.add_argument("--seeds", nargs="+", type=int, default=[7])
    p.add_argument("--episodes-per-seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument("--selection-seed-base", type=int, default=1000)
    p.add_argument("--v2x-seed-base", type=int, default=2000)
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument("--no-step-records", action="store_true")
    args = p.parse_args()

    results = paired_run(
        selection_modes=args.modes,
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        max_steps=args.max_steps,
        host=args.host, port=args.port, tm_port=args.tm_port,
        base_selection_seed=args.selection_seed_base,
        base_v2x_seed=args.v2x_seed_base,
    )
    paths = save_results(results, Path(args.out_dir), include_step_records=not args.no_step_records)
    print("Saved:", json.dumps(paths, indent=2))
    print("Aggregates:", json.dumps(results["aggregates"], indent=2, default=_json_default))


if __name__ == "__main__":
    main()
