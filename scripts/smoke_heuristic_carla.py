"""Milestone 8: CARLA smoke test for the observation-conditioned heuristic controller.

Runs ``marl.heuristic_controller.HeuristicController`` in CARLA 0.9.16 with the
existing V2X-enabled multi-agent environment and records per-decision-step logs,
CSV, JSON summaries, and PNG visualizations under ``results/milestone8/``.

Configuration (per milestone spec):
- Town03, 4 CAVs, cluster spawning
- candidate_source="v2x"
- V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0)
- selection_mode="nearest_k", selection_k=2 (primary smoke)
- 56-D observations
- Verifies the same controller handles random-k observations without parameter changes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.carla_env import CarlaMultiAgentEnv
from envs.v2x_channel import V2XConfig
from envs.obs_builder import OBS_DIM, V_MAX, MASK
from marl.heuristic_controller import HeuristicController, IDX_EGO_SPEED, CAND_START, C_REL_X, C_REL_Y, C_VALID, CAND_FEATURES, M_CANDIDATES

RESULTS_DIR = REPO_ROOT / "results" / "milestone8"


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def nearest_ahead_distance_m(obs: np.ndarray) -> float | None:
    """Closest valid candidate ahead in ego corridor from normalized obs (meters)."""
    best = None
    for slot in range(M_CANDIDATES):
        base = CAND_START + slot * CAND_FEATURES
        valid = float(obs[base + C_VALID]) > 0.5
        if not valid:
            continue
        rel_x = float(obs[base + C_REL_X]) * 50.0
        rel_y = float(obs[base + C_REL_Y]) * 50.0
        if rel_x >= -1.0 and abs(rel_y) <= 4.0:
            if best is None or rel_x < best:
                best = rel_x
    return best


def make_env(selection_mode: str, host: str, port: int, tm_port: int, seed: int = 7) -> CarlaMultiAgentEnv:
    return CarlaMultiAgentEnv(
        town="Town03",
        n_agents=4,
        host=host,
        port=port,
        tm_port=tm_port,
        seed=seed,
        selection_mode=selection_mode,
        selection_k=2,
        selection_seed=1000 + seed if selection_mode == "random_k" else seed,
        candidate_source="v2x",
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
        v2x_seed=2000 + seed,
        v2x_broadcasters="cavs_and_background",
        spawn_mode="cluster",
        spawn_cluster_max_distance=45.0,
    )


def run_smoke_episode(
    env: CarlaMultiAgentEnv,
    controller: HeuristicController,
    selection_mode: str,
    reset_seed: int,
    max_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs, reset_info = env.reset(seed=reset_seed)
    agents = list(env.agents)
    assert len(agents) == 4, f"expected 4 agents, got {len(agents)}"

    step_rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    ended = False

    for _ in range(max_steps):
        masks = env.action_masks()
        actions: dict[str, int] = {}
        for aid in agents:
            o = np.asarray(obs[aid], dtype=np.float32)
            assert o.shape == (OBS_DIM,), f"{aid} obs shape {o.shape}"
            assert o.dtype == np.float32, f"{aid} obs dtype {o.dtype}"
            assert np.all(np.isfinite(o)), f"{aid} obs has non-finite values"
            m = np.asarray(masks[aid], dtype=bool)
            assert m.shape == (MASK,)
            assert bool(m[3]), f"{aid} emergency brake must be available"
            a = int(controller.act(o, action_mask=m))
            assert 0 <= a < MASK, f"{aid} invalid action {a}"
            assert bool(m[a]), f"{aid} action {a} violates mask {m.tolist()}"
            actions[aid] = a

        obs_next, rewards, terminated, truncated, info = env.step(actions)
        final_info = info
        v2x = info.get("v2x") or {}
        sim_step = int(info.get("steps", len(step_rows) + 1))

        for aid in agents:
            pa = info["per_agent"][aid]
            sel = pa.get("selection") or {}
            recv = pa.get("v2x_received") or {}
            selected_ids = sel.get("selected_ids") or []
            # candidate count: selected (non-null) teammates in observation/selection
            selected_count = int(sel.get("selected_count", sum(1 for x in selected_ids if x is not None)))
            o = np.asarray(obs_next[aid], dtype=np.float32)
            ego_speed_mps = float(o[IDX_EGO_SPEED]) * V_MAX
            nearest_ahead = nearest_ahead_distance_m(o)
            step_rows.append({
                "simulation_step": sim_step,
                "agent_id": aid,
                "selection_mode": selection_mode,
                "ego_speed_mps": ego_speed_mps,
                "selected_candidate_ids": json.dumps(selected_ids, default=_json_default),
                "candidate_count_selected": selected_count,
                "candidate_pool_size": sel.get("pool_size"),
                "candidate_missing_slots": sel.get("missing_slots"),
                "action_selected": int(actions[aid]),
                "action_mask": json.dumps(np.asarray(masks[aid], dtype=bool).astype(int).tolist()),
                "safety_nearest_ahead_m": nearest_ahead,
                "safety_min_ttc_episode": _safe_float(info.get("min_ttc")),
                "collision": bool(pa.get("collision", False)),
                "off_road": bool(pa.get("off_road", False)),
                "stuck": bool(pa.get("stuck", False)),
                "v2x_received_count": int(recv.get("count", 0) or 0),
                "v2x_received_sender_ids": json.dumps(recv.get("sender_ids", []), default=_json_default),
                "v2x_received_ages": json.dumps(recv.get("ages", []), default=_json_default),
                "v2x_step_sent": int(v2x.get("sent", 0) or 0),
                "v2x_step_delivered": int(v2x.get("delivered", 0) or 0),
                "v2x_step_dropped_by_range": int(v2x.get("dropped_by_range", 0) or 0),
                "v2x_step_dropped_by_loss": int(v2x.get("dropped_by_loss", 0) or 0),
                "v2x_step_in_flight": v2x.get("in_flight"),
                "controller_reason": getattr(controller, "last_reason", None),
                "obs_finite": bool(np.all(np.isfinite(o))),
                "obs_shape": str(tuple(o.shape)),
                "obs_dtype": str(o.dtype),
            })

        obs = obs_next
        if all(terminated.values()) or any(truncated.values()):
            ended = True
            break

    summary = {
        "selection_mode": selection_mode,
        "reset_seed": int(reset_seed),
        "agents_expected": 4,
        "agents_observed": len(agents),
        "steps_executed": int(final_info.get("steps", len(set(r["simulation_step"] for r in step_rows)))),
        "ended": ended,
        "ending_reason": final_info.get("ending_reason"),
        "success": bool(final_info.get("success", False)) if final_info else False,
        "collision": bool(final_info.get("collision", False)) if final_info else False,
        "off_road": bool(final_info.get("off_road", False)) if final_info else False,
        "stuck": bool(final_info.get("stuck", False)) if final_info else False,
        "timeout": bool(final_info.get("ending_reason") == "timeout") if final_info else False,
        "team_reward_total": float(final_info.get("team_reward", 0.0)) if final_info else 0.0,
        "min_ttc": _safe_float(final_info.get("min_ttc")),
        "jerk": _safe_float(final_info.get("jerk")),
        "v2x_episode": (final_info.get("v2x") or {}).get("episode") if final_info else None,
        "reset_info": reset_info,
        "all_obs_shape_56": all(r["obs_shape"] == "(56,)" for r in step_rows),
        "all_obs_float32": all(r["obs_dtype"] == "float32" for r in step_rows),
        "all_obs_finite": all(r["obs_finite"] for r in step_rows),
        "all_actions_valid": True,  # asserted during run
        "all_masks_respected": True,  # asserted during run
        "v2x_metadata_populated": bool(final_info.get("v2x")) if final_info else False,
    }
    return summary, step_rows


def save_outputs(out_dir: Path, summaries: list[dict[str, Any]], all_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, default=_json_default), encoding="utf-8")
    paths["config"] = str(config_path)

    json_path = out_dir / "smoke_summary.json"
    json_path.write_text(json.dumps({"config": config, "runs": summaries}, indent=2, default=_json_default), encoding="utf-8")
    paths["summary_json"] = str(json_path)

    csv_path = out_dir / "per_step_log.csv"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        paths["per_step_csv"] = str(csv_path)

    # Visualizations
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    modes = [s["selection_mode"] for s in summaries]
    steps_by_mode = {}
    for mode in modes:
        mode_rows = [r for r in all_rows if r["selection_mode"] == mode]
        steps = sorted({int(r["simulation_step"]) for r in mode_rows})
        steps_by_mode[mode] = (steps, mode_rows)

    # 1) Action distribution
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4), squeeze=False)
    action_names = ["KEEP", "ACCEL", "DECEL", "BRAKE", "L_LEFT", "L_RIGHT"]
    for ax, mode in zip(axes[0], modes):
        _, rows = steps_by_mode[mode]
        counts = [sum(1 for r in rows if int(r["action_selected"]) == i) for i in range(6)]
        ax.bar(action_names, counts)
        ax.set_title(f"Action distribution ({mode})")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = fig_dir / "action_distribution.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["action_distribution_png"] = str(p)

    # 2) Ego speed over steps (per agent)
    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        steps, rows = steps_by_mode[mode]
        for aid in sorted({r["agent_id"] for r in rows}):
            aid_rows = [r for r in rows if r["agent_id"] == aid]
            xs = [int(r["simulation_step"]) for r in aid_rows]
            ys = [float(r["ego_speed_mps"]) for r in aid_rows]
            ax.plot(xs, ys, marker="o", markersize=2, label=aid)
        ax.set_title(f"Ego speed over steps ({mode})")
        ax.set_xlabel("Simulation step")
        ax.set_ylabel("Speed (m/s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "ego_speed_over_steps.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["ego_speed_png"] = str(p)

    # 3) Candidate availability over time (pool/selected/missing)
    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        steps, rows = steps_by_mode[mode]
        for metric, label in [("candidate_pool_size", "pool"), ("candidate_count_selected", "selected"), ("candidate_missing_slots", "missing")]:
            means = []
            for s in steps:
                vals = [r[metric] for r in rows if int(r["simulation_step"]) == s and r[metric] is not None]
                means.append(float(np.mean(vals)) if vals else np.nan)
            ax.plot(steps, means, marker="o", markersize=2, label=label)
        ax.set_title(f"Candidate availability ({mode})")
        ax.set_xlabel("Simulation step")
        ax.set_ylabel("Count (mean over agents)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "candidate_availability.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["candidate_availability_png"] = str(p)

    # 4) V2X reception over time
    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        steps, rows = steps_by_mode[mode]
        # per-step env-level counts are repeated per agent row; take one per step
        sent, delivered = [], []
        for s in steps:
            srows = [r for r in rows if int(r["simulation_step"]) == s]
            sent.append(int(srows[0]["v2x_step_sent"]) if srows else 0)
            delivered.append(int(srows[0]["v2x_step_delivered"]) if srows else 0)
        ax.plot(steps, sent, marker="o", markersize=2, label="sent")
        ax.plot(steps, delivered, marker="o", markersize=2, label="delivered")
        ax.set_title(f"V2X messages per step ({mode})")
        ax.set_xlabel("Simulation step")
        ax.set_ylabel("Messages")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "v2x_reception.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["v2x_reception_png"] = str(p)

    # 5) Selected candidate counts over time
    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        steps, rows = steps_by_mode[mode]
        for aid in sorted({r["agent_id"] for r in rows}):
            aid_rows = [r for r in rows if r["agent_id"] == aid]
            xs = [int(r["simulation_step"]) for r in aid_rows]
            ys = [int(r["candidate_count_selected"]) for r in aid_rows]
            ax.plot(xs, ys, marker="o", markersize=2, label=aid)
        ax.set_title(f"Selected candidate counts ({mode})")
        ax.set_xlabel("Simulation step")
        ax.set_ylabel("Selected count")
        ax.set_ylim(-0.1, 6.1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "selected_candidate_counts.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["selected_candidate_counts_png"] = str(p)

    # 6) Safety-related controller responses (brake/lane-change & nearest ahead)
    fig, axes = plt.subplots(2, len(modes), figsize=(7 * len(modes), 7), squeeze=False)
    for col, mode in enumerate(modes):
        steps, rows = steps_by_mode[mode]
        # safety actions count per step
        safety_counts = []
        for s in steps:
            srows = [r for r in rows if int(r["simulation_step"]) == s]
            safety_counts.append(sum(1 for r in srows if int(r["action_selected"]) in (2, 3, 4, 5)))
        axes[0, col].bar(steps, safety_counts)
        axes[0, col].set_title(f"Safety-related actions per step ({mode})")
        axes[0, col].set_xlabel("Simulation step")
        axes[0, col].set_ylabel("Count (decel/brake/lane)")
        axes[0, col].grid(alpha=0.3)

        for aid in sorted({r["agent_id"] for r in rows}):
            aid_rows = [r for r in rows if r["agent_id"] == aid]
            xs = [int(r["simulation_step"]) for r in aid_rows]
            ys = [r["safety_nearest_ahead_m"] if r["safety_nearest_ahead_m"] is not None else np.nan for r in aid_rows]
            axes[1, col].plot(xs, ys, marker="o", markersize=2, label=aid)
        axes[1, col].set_title(f"Nearest ahead candidate distance ({mode})")
        axes[1, col].set_xlabel("Simulation step")
        axes[1, col].set_ylabel("Distance (m)")
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(alpha=0.3)
    fig.tight_layout()
    p = fig_dir / "safety_responses.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    paths["safety_responses_png"] = str(p)

    paths["figures_dir"] = str(fig_dir)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "town": "Town03",
        "n_agents": 4,
        "spawn_mode": "cluster",
        "spawn_cluster_max_distance": 45.0,
        "candidate_source": "v2x",
        "v2x_config": {"range_m": 50.0, "delay_steps": 0, "loss_prob": 0.0},
        "selection_k": 2,
        "obs_dim": 56,
        "steps": args.steps,
        "seed": args.seed,
        "host": args.host,
        "port": args.port,
        "tm_port": args.tm_port,
        "controller": "marl.heuristic_controller.HeuristicController",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    summaries = []
    all_rows = []

    # Primary smoke: nearest_k
    print("=== Smoke: nearest_k ===", flush=True)
    controller = HeuristicController()
    env = make_env("nearest_k", args.host, args.port, args.tm_port, seed=args.seed)
    try:
        summary, rows = run_smoke_episode(env, controller, "nearest_k", reset_seed=args.seed, max_steps=args.steps)
    finally:
        env.close()
    summaries.append(summary)
    all_rows.extend(rows)
    print(f"nearest_k: steps={summary['steps_executed']} ended={summary['ended']} reason={summary['ending_reason']} "
          f"obs56={summary['all_obs_shape_56']} finite={summary['all_obs_finite']} v2x={summary['v2x_metadata_populated']}", flush=True)

    # Verify same controller handles random-k observations without parameter changes
    print("=== Smoke: random_k (same controller instance/params) ===", flush=True)
    env2 = make_env("random_k", args.host, args.port, args.tm_port, seed=args.seed)
    try:
        # Same controller object and parameters
        summary_r, rows_r = run_smoke_episode(env2, controller, "random_k", reset_seed=args.seed, max_steps=min(args.steps, 10))
    finally:
        env2.close()
    summaries.append(summary_r)
    all_rows.extend(rows_r)
    print(f"random_k: steps={summary_r['steps_executed']} ended={summary_r['ended']} reason={summary_r['ending_reason']} "
          f"obs56={summary_r['all_obs_shape_56']} finite={summary_r['all_obs_finite']} v2x={summary_r['v2x_metadata_populated']}", flush=True)

    paths = save_outputs(out_dir, summaries, all_rows, config)

    # Console evidence
    evidence = {
        "nearest_k": {k: summaries[0][k] for k in ("steps_executed","all_obs_shape_56","all_obs_float32","all_obs_finite","all_actions_valid","all_masks_respected","v2x_metadata_populated","ending_reason")},
        "random_k_same_controller": {k: summaries[1][k] for k in ("steps_executed","all_obs_shape_56","all_obs_float32","all_obs_finite","all_actions_valid","all_masks_respected","v2x_metadata_populated","ending_reason")},
        "outputs": paths,
    }
    (out_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, default=_json_default), encoding="utf-8")
    print("SMOKE_EVIDENCE:", json.dumps(evidence, indent=2, default=_json_default), flush=True)
    print("M8 SMOKE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
