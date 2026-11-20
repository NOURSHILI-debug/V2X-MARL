"""Small live CARLA validation for controllers.observation_heuristic (Milestone 8A).

Runs a few short episodes with the exact requested configuration for
selection_mode nearest_k and random_k using the identical controller.
Saves results under results/milestone8a/validation/ without overwriting
existing milestone8a smoke artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from controllers.observation_heuristic import ObservationHeuristicController
from envs.carla_env import CarlaMultiAgentEnv
from envs.obs_builder import OBS_DIM, MASK, V_MAX, EGO, ROUTE, M_CANDIDATES, CAND_FEATURES

CAND_START = EGO + ROUTE
C_REL_X = 0
C_REL_Y = 1
C_VALID = 5
from envs.v2x_channel import V2XConfig

OUT_DIR = REPO_ROOT / "results" / "milestone8a" / "validation"


def jdefault(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o); return v if math.isfinite(v) else None
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def _nearest_ahead_m(obs):
    best = None
    for s in range(M_CANDIDATES):
        b = CAND_START + s * CAND_FEATURES
        if float(obs[b + C_VALID]) <= 0.5:
            continue
        rx = float(obs[b + C_REL_X]) * 50.0
        ry = float(obs[b + C_REL_Y]) * 50.0
        if rx >= -1.0 and abs(ry) <= 4.0 and (best is None or rx < best):
            best = rx
    return best


def make_env(mode, host, port, tm_port, seed):
    return CarlaMultiAgentEnv(
        town="Town03",
        n_agents=4,
        host=host, port=port, tm_port=tm_port,
        seed=seed,
        selection_mode=mode,
        selection_k=2,
        selection_seed=1000 + seed if mode == "random_k" else seed,
        candidate_source="v2x",
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
        v2x_seed=2000 + seed,
        v2x_broadcasters="cavs_and_background",
        spawn_mode="cluster",
        spawn_cluster_max_distance=45.0,
    )


def run_episode(mode, controller, host, port, tm_port, seed, max_steps):
    env = make_env(mode, host, port, tm_port, seed)
    step_rows = []
    try:
        obs, reset_info = env.reset(seed=seed)
        agents = list(env.agents)
        assert len(agents) == 4
        final_info = {}
        ended = False
        for _ in range(max_steps):
            masks = env.action_masks()
            actions = {}
            for aid in agents:
                o = np.asarray(obs[aid], dtype=np.float32)
                assert o.shape == (OBS_DIM,) and o.dtype == np.float32 and np.all(np.isfinite(o))
                m = np.asarray(masks[aid], dtype=bool)
                assert m.shape == (MASK,)
                a = int(controller.act(o, action_mask=m))
                assert 0 <= a < MASK and bool(m[a])
                actions[aid] = a
            obs_next, rewards, terminated, truncated, info = env.step(actions)
            final_info = info
            step = int(info.get("steps", 0))
            v2x = info.get("v2x") or {}
            for aid in agents:
                pa = info["per_agent"][aid]
                sel = pa.get("selection") or {}
                o = np.asarray(obs_next[aid], dtype=np.float32)
                # per-step observation metadata (summary, not full 56-dump)
                cand_meta = []
                for s in range(M_CANDIDATES):
                    b = CAND_START + s * CAND_FEATURES
                    if float(o[b + C_VALID]) > 0.5:
                        cand_meta.append({
                            "slot": s,
                            "rel_x_m": float(o[b + C_REL_X]) * 50.0,
                            "rel_y_m": float(o[b + C_REL_Y]) * 50.0,
                        })
                step_rows.append({
                    "step": step,
                    "agent_id": aid,
                    "selection_mode": mode,
                    "action": actions[aid],
                    "action_mask": np.asarray(masks[aid], dtype=bool).astype(int).tolist(),
                    "reward": float(rewards.get(aid, 0.0)),
                    "ego_speed_mps": float(o[0]) * V_MAX,
                    "obs_shape": list(o.shape),
                    "obs_dtype": str(o.dtype),
                    "obs_finite": bool(np.all(np.isfinite(o))),
                    "selected_candidate_ids": sel.get("selected_ids"),
                    "candidate_pool_size": sel.get("pool_size"),
                    "selected_count": sel.get("selected_count"),
                    "missing_slots": sel.get("missing_slots"),
                    "received_count": (pa.get("v2x_received") or {}).get("count"),
                    "nearest_ahead_m": _nearest_ahead_m(o),
                    "valid_candidate_obs_count": len(cand_meta),
                    "candidate_obs_meta": cand_meta,
                    "collision": bool(pa.get("collision", False)),
                    "off_road": bool(pa.get("off_road", False)),
                    "stuck": bool(pa.get("stuck", False)),
                    "goal_reached": bool(pa.get("goal_reached", False)),
                    "stop_reason": pa.get("stop_reason"),
                    "arrival_step": pa.get("arrival_step"),
                })
            obs = obs_next
            if all(terminated.values()) or any(truncated.values()):
                ended = True
                break

        # episode summary
        per_agent = (final_info.get("per_agent") or {})
        arrivals = [pa.get("arrival_step") for pa in per_agent.values() if pa.get("arrival_step") is not None]
        # route completion/progress proxy from last obs per agent not directly stored; report arrival/goal/collisions
        episode = {
            "selection_mode": mode,
            "seed": seed,
            "max_steps": max_steps,
            "steps": int(final_info.get("steps", len(set(r["step"] for r in step_rows)))),
            "ended": ended,
            "ending_reason": final_info.get("ending_reason"),
            "success": bool(final_info.get("success", False)) if final_info else False,
            "collision": bool(final_info.get("collision", False)) if final_info else False,
            "off_road": bool(final_info.get("off_road", False)) if final_info else False,
            "stuck": bool(final_info.get("stuck", False)) if final_info else False,
            "timeout": (final_info.get("ending_reason") == "timeout") if final_info else False,
            "team_reward_total": float(sum(r["reward"] for r in step_rows) / 4.0) if step_rows else 0.0,
            "team_reward_info": final_info.get("team_reward"),
            "arrival_steps": arrivals,
            "route_completion": {aid: {"goal_reached": per_agent.get(aid, {}).get("goal_reached"), "arrival_step": per_agent.get(aid, {}).get("arrival_step")} for aid in agents},
            "travel_time_steps": int(final_info.get("steps", 0)),
            "min_ttc": final_info.get("min_ttc"),
            "jerk": final_info.get("jerk"),
            "v2x_episode": (final_info.get("v2x") or {}).get("episode"),
            "all_obs_valid": all(r["obs_finite"] and r["obs_shape"] == [56] and r["obs_dtype"] == "float32" for r in step_rows),
            "reset_info": reset_info,
        }
        return episode, step_rows
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "town": "Town03", "n_agents": 4,
        "spawn_mode": "cluster", "spawn_cluster_max_distance": 45.0,
        "candidate_source": "v2x",
        "v2x_config": {"range_m": 50.0, "delay_steps": 0, "loss_prob": 0.0},
        "selection_k": 2,
        "selection_modes": ["nearest_k", "random_k"],
        "controller": "controllers.observation_heuristic.ObservationHeuristicController",
        "controller_params_identical": True,
        "seed": args.seed, "max_steps": args.max_steps,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=jdefault), encoding="utf-8")

    controller = ObservationHeuristicController()  # same instance/params for both modes
    episodes = []
    all_rows = []
    for mode in ["nearest_k", "random_k"]:
        print(f"Running validation: {mode}", flush=True)
        ep, rows = run_episode(mode, controller, args.host, args.port, args.tm_port, args.seed, args.max_steps)
        episodes.append(ep)
        all_rows.extend(rows)
        print(f"  {mode}: steps={ep['steps']} ended={ep['ended']} reason={ep['ending_reason']} success={ep['success']} collision={ep['collision']} obs_valid={ep['all_obs_valid']}", flush=True)

    (out_dir / "episodes.json").write_text(json.dumps(episodes, indent=2, default=jdefault), encoding="utf-8")
    csv_path = out_dir / "per_step_log.csv"
    if all_rows:
        # flatten candidate_obs_meta for csv
        flat = []
        for r in all_rows:
            rr = dict(r)
            rr["candidate_obs_meta"] = json.dumps(rr["candidate_obs_meta"], default=jdefault)
            rr["action_mask"] = json.dumps(rr["action_mask"], default=jdefault)
            rr["selected_candidate_ids"] = json.dumps(rr["selected_candidate_ids"], default=jdefault)
            flat.append(rr)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader(); w.writerows(flat)

    summary = {
        "config": config,
        "episodes": episodes,
        "notes": "Small live validation only; not a safety proof and not a performance comparison.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=jdefault), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "episodes": [{k: e[k] for k in ('selection_mode','steps','ending_reason','success','collision','off_road','stuck','all_obs_valid')} for e in episodes]}, indent=2))


if __name__ == "__main__":
    main()
