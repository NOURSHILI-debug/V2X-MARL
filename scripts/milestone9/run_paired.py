"""Paired nearest_k vs random_k experiment runner for Milestone 9.

Additive infrastructure. Creates a fresh CARLA environment per episode,
applies explicit per-episode seeds (scenario/selection/V2X), runs both
strategies under equivalent scenario configuration with the identical
deterministic observation-conditioned controller, and logs standardized
artifacts under results/milestone9/.

Safety: does not launch large experiments by default. A single pilot pair is
executed only when explicitly requested via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.milestone9.config import (
    Milestone9Config,
    default_config,
    derive_selection_seed,
    derive_v2x_seed,
    episode_id as make_episode_id,
    pair_id as make_pair_id,
)
from scripts.milestone9.logging import (
    DEFAULT_RESULTS_ROOT,
    ExperimentLayout,
    atomic_write_csv,
    atomic_write_json,
    derive_ending_reason,
    make_experiment_id,
    scenario_signature_from_reset,
    selection_summary_from_info,
    sha256_file,
    utc_now_iso,
    v2x_received_from_info,
)
from scripts.milestone9.validate_pair import validate_pair

# CARLA-dependent imports are performed lazily inside runtime functions.


def _jsonable_location(loc: Any) -> Dict[str, float]:
    return {"x": float(loc.x), "y": float(loc.y), "z": float(loc.z)}


def _jsonable_rotation(rot: Any) -> Dict[str, float]:
    return {"pitch": float(rot.pitch), "yaw": float(rot.yaw), "roll": float(rot.roll)}


def _route_points(route: Any) -> List[Dict[str, float]]:
    points = []
    for wp in route or []:
        if hasattr(wp, "x") and hasattr(wp, "y"):
            points.append({"x": float(wp.x), "y": float(wp.y), "z": float(getattr(wp, "z", 0.0))})
        elif isinstance(wp, Mapping):
            points.append({
                "x": float(wp.get("x", 0.0)),
                "y": float(wp.get("y", 0.0)),
                "z": float(wp.get("z", 0.0)),
            })
    return points


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _safe_call(obj: Any, name: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    try:
        fn = getattr(obj, name)
        return fn(*args, **kwargs)
    except Exception:
        return default


def _vehicle_descriptor(actor: Any, managed: bool, experiment_agent_id: Optional[str] = None) -> Dict[str, Any]:
    desc: Dict[str, Any] = {
        "managed": bool(managed),
        "type_id": _safe_attr(actor, "type_id"),
        "carla_actor_id_diagnostic": _safe_attr(actor, "id"),
    }
    if experiment_agent_id is not None:
        desc["experiment_agent_id"] = experiment_agent_id
    try:
        tf = actor.get_transform()
        desc["transform"] = {"position": _jsonable_location(tf.location), "orientation": _jsonable_rotation(tf.rotation)}
    except Exception:
        desc["transform"] = None
    try:
        vel = actor.get_velocity()
        desc["velocity_mps"] = {"x": float(vel.x), "y": float(vel.y), "z": float(vel.z)}
        desc["speed_mps"] = math.sqrt(float(vel.x) ** 2 + float(vel.y) ** 2 + float(vel.z) ** 2)
    except Exception:
        desc["velocity_mps"] = None
        desc["speed_mps"] = None
    # best-effort stable-ish attributes
    attrs = {}
    try:
        for key in ("role_name", "color", "base_type", "object_type"):
            if hasattr(actor, "attributes") and actor.attributes is not None and key in actor.attributes:
                attrs[key] = actor.attributes.get(key)
    except Exception:
        attrs = {}
    if attrs:
        desc["attributes"] = attrs
    return desc


def _traffic_light_descriptor(light: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "carla_actor_id_diagnostic": _safe_attr(light, "id"),
        "type_id": _safe_attr(light, "type_id"),
    }
    try:
        tf = light.get_transform()
        d["transform"] = {"position": _jsonable_location(tf.location), "orientation": _jsonable_rotation(tf.rotation)}
    except Exception:
        d["transform"] = None
    try:
        state = _safe_call(light, "get_state")
        d["state"] = str(state) if state is not None else None
    except Exception:
        d["state"] = None
    # best-effort timing/phase info (may be unavailable or not reliable across CARLA APIs/versions)
    timing: Dict[str, Any] = {"available": False, "limitation": "Traffic-light phase/timing not reliably exposed; best-effort fields only."}
    for method, key in (
        ("get_elapsed_time", "elapsed_time_s"),
        ("get_time_is_frozen", "time_is_frozen"),
        ("get_green_time", "green_time_s"),
        ("get_yellow_time", "yellow_time_s"),
        ("get_red_time", "red_time_s"),
    ):
        try:
            val = getattr(light, method)()
            timing[key] = val
            timing["available"] = True
        except Exception:
            pass
    try:
        pole = _safe_call(light, "get_pole_index")
        if pole is not None:
            timing["pole_index"] = pole
            timing["available"] = True
    except Exception:
        pass
    d["timing"] = timing
    return d


def _collect_world_snapshot(env: Any) -> Dict[str, Any]:
    world = _safe_attr(env, "world")
    snap: Dict[str, Any] = {"map_name": _safe_attr(env, "town")}
    try:
        if world is not None:
            carla_map = _safe_call(world, "get_map")
            if carla_map is not None:
                snap["map_name"] = str(getattr(carla_map, "name", snap.get("map_name")))
    except Exception:
        pass
    settings = {}
    try:
        if world is not None:
            ws = _safe_call(world, "get_settings")
            if ws is not None:
                settings = {
                    "synchronous_mode": _safe_attr(ws, "synchronous_mode"),
                    "fixed_delta_seconds": _safe_attr(ws, "fixed_delta_seconds"),
                    "no_rendering_mode": _safe_attr(ws, "no_rendering_mode"),
                }
    except Exception:
        settings = {"error": "unavailable"}
    snap["world_settings"] = settings
    weather = {}
    try:
        if world is not None:
            w = _safe_call(world, "get_weather")
            if w is not None:
                for k in ("cloudiness", "precipitation", "precipitation_deposits", "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance", "wetness"):
                    if hasattr(w, k):
                        weather[k] = getattr(w, k)
    except Exception:
        weather = {"error": "unavailable"}
    snap["weather"] = weather
    return snap


def _collect_background_and_traffic_lights(env: Any) -> Dict[str, Any]:
    world = _safe_attr(env, "world")
    managed_ids = set()
    try:
        for v in (getattr(env, "vehicles", {}) or {}).values():
            if v is not None and hasattr(v, "id"):
                managed_ids.add(int(v.id))
    except Exception:
        managed_ids = set()
    vehicles = []
    traffic_lights = []
    try:
        if world is not None:
            actors = _safe_call(world, "get_actors")
            if actors is not None:
                for actor in actors.filter("vehicle.*"):
                    aid = int(getattr(actor, "id", -1))
                    vehicles.append(_vehicle_descriptor(actor, managed=aid in managed_ids))
                for tl in actors.filter("traffic.traffic_light*"):
                    traffic_lights.append(_traffic_light_descriptor(tl))
    except Exception as e:
        vehicles = [{"error": f"unavailable: {type(e).__name__}: {e}"}]
        traffic_lights = [{"error": f"unavailable: {type(e).__name__}: {e}"}]
    # deterministic ordering for comparison
    def _vkey(d):
        if not isinstance(d, dict):
            return str(d)
        t = d.get("transform") or {}
        p = (t.get("position") or {}) if isinstance(t, dict) else {}
        return (bool(d.get("managed")), str(d.get("type_id")), round(float(p.get("x", 0.0)), 3), round(float(p.get("y", 0.0)), 3), round(float(p.get("z", 0.0)), 3))
    vehicles_sorted = sorted(vehicles, key=_vkey) if vehicles and isinstance(vehicles[0], dict) else vehicles
    return {
        "managed_vehicle_actor_ids_diagnostic": sorted(list(managed_ids)),
        "vehicles": vehicles_sorted,
        "traffic_lights": traffic_lights,
        "background_capture_note": "Background/unmanaged actors and traffic-light phase are best-effort snapshots at reset; CARLA actor IDs are diagnostic and may be unstable across reloads.",
    }


def collect_initial_scenario(env: Any, reset_info: Mapping[str, Any]) -> Dict[str, Any]:
    agents = list(getattr(env, "agents", []))
    initial_state: Dict[str, Any] = {}
    controlled_agents: Dict[str, Any] = {}
    for aid in agents:
        vehicle = env.vehicles.get(aid)
        if vehicle is None:
            initial_state[aid] = {"missing": True}
            controlled_agents[aid] = {"experiment_agent_id": aid, "missing": True}
            continue
        tf = vehicle.get_transform()
        vel = vehicle.get_velocity()
        speed = math.sqrt(float(vel.x) ** 2 + float(vel.y) ** 2 + float(vel.z) ** 2)
        goal = env.goal.get(aid)
        route = env.route.get(aid)
        per_agent_reset = ((reset_info or {}).get("per_agent") or {}).get(aid, {}) if isinstance(reset_info, Mapping) else {}
        initial_state[aid] = {
            "position": _jsonable_location(tf.location),
            "orientation": _jsonable_rotation(tf.rotation),
            "speed_mps": float(speed),
            "destination": _jsonable_location(goal.location) if goal is not None else None,
            "destination_orientation": _jsonable_rotation(goal.rotation) if goal is not None else None,
            "route": _route_points(route),
            "route_length_points": len(route or []),
            "route_applied": per_agent_reset.get("route_applied") if isinstance(per_agent_reset, Mapping) else None,
        }
        controlled_agents[aid] = {
            **_vehicle_descriptor(vehicle, managed=True, experiment_agent_id=aid),
            "destination": _jsonable_location(goal.location) if goal is not None else None,
            "route_status": {"route_applied": per_agent_reset.get("route_applied") if isinstance(per_agent_reset, Mapping) else None, "route_length_points": len(route or [])},
        }
    sig = scenario_signature_from_reset(reset_info, agents)
    sig.update({
        "agents": agents,
        "n_agents": len(agents),
        "initial_state": initial_state,
        "world": _collect_world_snapshot(env),
        "controlled_agents": controlled_agents,
        "background_and_traffic_lights": _collect_background_and_traffic_lights(env),
        "snapshot_policy": {
            "must_match": [
                "route_applied (overall, strict True boolean)",
                "town/map name and controlled agent identities/count/physical initial states within tolerances",
                "world synchronous_mode/fixed_delta_seconds when present (pair consistency)",
            ],
            "allowed_to_differ": [
                "CARLA actor IDs and diagnostic IDs (unstable, especially across reloads)",
                "strategy and strategy-specific selection/v2x seeds",
            ],
            "diagnostic_only": [
                "background/unmanaged actor full set details (best-effort; may vary)",
                "traffic-light timing/phase details when unavailable or unstable",
                "weather details unless explicitly required by future protocol",
            ],
            "missing_required_invalidates_pair": True,
        },
    })
    return sig


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _extract_observation_summary(obs: Any) -> Dict[str, Any]:
    try:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    except Exception:
        return {"available": False}
    if arr.size != 56:
        return {"available": True, "obs_dim": int(arr.size)}
    # Compact, contract-based summary (no full dump by default).
    cand = arr[12:48].reshape(6, 6)
    valid = cand[:, 5] > 0.5
    return {
        "available": True,
        "speed_mps": float(arr[0]) * 30.0,
        "distance_to_goal_m": float(arr[6]) * 200.0,
        "progress_fraction": float(arr[7]),
        "valid_candidate_count_obs": int(np.count_nonzero(valid)),
        "selected_relative_positions_m": [
            {
                "rel_x_m": float(cand[i, 0]) * 50.0,
                "rel_y_m": float(cand[i, 1]) * 50.0,
                "rel_vx_mps": float(cand[i, 2]) * 30.0,
                "rel_vy_mps": float(cand[i, 3]) * 30.0,
            }
            for i in range(6)
            if bool(valid[i])
        ],
    }


def build_controller(cfg: Milestone9Config):
    module_name = cfg.controller.module
    class_name = cfg.controller.class_name
    mod = __import__(module_name, fromlist=[class_name])
    cls = getattr(mod, class_name)
    params = dict(cfg.controller.parameters)
    return cls(**params)


def build_v2x_config(cfg: Milestone9Config):
    from envs.v2x_channel import V2XConfig

    return V2XConfig(
        range_m=float(cfg.v2x.communication_range_m),
        delay_steps=int(cfg.v2x.delay_steps),
        loss_prob=float(cfg.v2x.packet_loss),
    )


def create_env(cfg: Milestone9Config, strategy: str, scenario_seed: int, selection_seed: int, v2x_seed: int):
    from envs.carla_env import CarlaMultiAgentEnv

    env = CarlaMultiAgentEnv(
        town=cfg.environment.town,
        n_agents=cfg.environment.n_agents,
        host=cfg.host,
        port=cfg.port,
        tm_port=cfg.tm_port,
        seed=int(scenario_seed),
        selection_mode=strategy,
        selection_k=int(cfg.environment.selection_k),
        selection_seed=int(selection_seed),
        candidate_source=cfg.environment.candidate_source,
        v2x_config=build_v2x_config(cfg),
        v2x_seed=int(v2x_seed),
        v2x_broadcasters=cfg.v2x.broadcasters,
        spawn_mode=cfg.environment.spawn_mode,
        spawn_cluster_max_distance=float(cfg.environment.spawn_cluster_max_distance_m),
        max_steps=int(cfg.environment.max_steps),
    )
    return env


def _episode_ended(terminated: Mapping[str, bool], truncated: Mapping[str, bool], info: Mapping[str, Any]) -> bool:
    if info and ("success" in info or "ending_reason" in info or "collision" in info):
        # Episode-level flags present on terminal step.
        if any(bool(v) for v in terminated.values()) or any(bool(v) for v in truncated.values()):
            return True
        if info.get("success") or info.get("collision") or info.get("off_road") or info.get("stuck"):
            return True
    return any(bool(v) for v in terminated.values()) or any(bool(v) for v in truncated.values())


def run_episode(
    cfg: Milestone9Config,
    *,
    experiment_id: str,
    scenario_seed: int,
    episode_index: int,
    strategy: str,
    controller: Any,
    layout: Optional[ExperimentLayout] = None,
) -> Dict[str, Any]:
    ep_id = make_episode_id(experiment_id, scenario_seed, episode_index, strategy)
    p_id = make_pair_id(experiment_id, scenario_seed, episode_index)
    selection_seed = int(derive_selection_seed(cfg, scenario_seed, episode_index, strategy))
    v2x_seed = int(derive_v2x_seed(cfg, scenario_seed, episode_index, strategy))
    realized = {
        "scenario_seed": int(scenario_seed),
        "selection_seed": selection_seed,
        "v2x_seed": v2x_seed,
        "reset_seed": int(scenario_seed),
    }

    episode_record: Dict[str, Any] = {
        "episode_id": ep_id,
        "pair_id": p_id,
        "strategy": strategy,
        "episode_index": int(episode_index),
        "scenario_seed": int(scenario_seed),
        "selection_seed": selection_seed,
        "v2x_seed": v2x_seed,
        "realized_seeds": realized,
        "execution_status": "running",
        "failure_reason": None,
        "config": cfg.to_dict(),
        "controller": cfg.controller.identity(),
        "environment": {
            "carla_version": cfg.carla_version,
            "town": cfg.environment.town,
            "n_agents": cfg.environment.n_agents,
            "episode_horizon": cfg.environment.max_steps,
        },
        "v2x_config": cfg.to_dict().get("v2x"),
        "created_utc": utc_now_iso(),
        "lifecycle": {
            "isolation_protocol": "fresh_env_per_episode_create_reset_close",
            "live_carla_isolation_verified": False,
            "readiness_gate": "live_carla_verification_required",
            "cleanup_attempted": False,
            "cleanup_status": "not_attempted",
            "cleanup_error": None,
            "close_called": False,
        },
        "persistence": {"status": "pending", "error": None, "artifacts_present": False},
    }

    step_rows: List[Dict[str, Any]] = []
    env = None
    obs = None
    info = {}
    terminated = {f"cav_{i}": False for i in range(cfg.environment.n_agents)}
    truncated = {f"cav_{i}": False for i in range(cfg.environment.n_agents)}
    rewards = {}
    team_reward_total = 0.0
    team_reward_steps = 0
    arrival_steps: Dict[str, Optional[int]] = {}
    try:
        env = create_env(cfg, strategy, scenario_seed, selection_seed, v2x_seed)
        obs, reset_info = env.reset(seed=int(scenario_seed))
        info = dict(reset_info or {})
        episode_record["reset_info"] = info
        episode_record["initial_scenario"] = collect_initial_scenario(env, info)
        episode_record["agents_order"] = list(getattr(env, "agents", []))
        arrival_steps = {aid: None for aid in episode_record["agents_order"]}

        # Strict route-validity precondition (do not proceed with an invalid route).
        route_applied = (episode_record.get("initial_scenario") or {}).get("route_applied", info.get("route_applied"))
        per_agent_reset = ((episode_record.get("initial_scenario") or {}).get("per_agent_reset")) or (info.get("per_agent") or {})
        per_agent_route_ok = True
        if isinstance(per_agent_reset, Mapping) and per_agent_reset:
            for aid, pdata in per_agent_reset.items():
                if isinstance(pdata, Mapping) and "route_applied" in pdata and pdata.get("route_applied") is not True:
                    per_agent_route_ok = False
                    break
        if type(route_applied) is not bool or route_applied is not True or not per_agent_route_ok:
            raise RuntimeError(
                f"Route validity precondition failed: route_applied={route_applied!r}, "
                f"per_agent_route_ok={per_agent_route_ok}, per_agent_reset={per_agent_reset}"
            )

        max_steps = int(getattr(env, "max_steps", cfg.environment.max_steps))
        # Run until environment termination/truncation. Do not add a conflicting cap.
        for step_idx in range(1, max_steps + 1):
            actions: Dict[str, int] = {}
            masks: Dict[str, Any] = {}
            for aid in episode_record["agents_order"]:
                a_obs = obs.get(aid)
                mask = None
                try:
                    # derive mask from env if available for logging accuracy
                    env_masks = env.action_masks()
                    mask = env_masks.get(aid)
                except Exception:
                    mask = None
                masks[aid] = mask
                actions[aid] = int(controller.act(a_obs, mask=mask)) if a_obs is not None else 0

            obs, rewards, terminated, truncated, info = env.step(actions)
            info = dict(info or {})
            team_reward = info.get("team_reward")
            if team_reward is not None:
                team_reward_total += float(team_reward)
                team_reward_steps += 1

            per_agent = info.get("per_agent", {}) or {}
            for aid in episode_record["agents_order"]:
                pa = per_agent.get(aid, {}) or {}
                if pa.get("goal_reached") and arrival_steps.get(aid) is None:
                    arrival_steps[aid] = pa.get("arrival_step", info.get("steps"))
                sel = selection_summary_from_info(pa)
                recv = v2x_received_from_info(pa)
                step_rows.append({
                    "episode_id": ep_id,
                    "pair_id": p_id,
                    "strategy": strategy,
                    "step": int(info.get("steps", step_idx)),
                    "agent_id": aid,
                    "action": int(actions.get(aid, 0)),
                    "action_mask": list(masks.get(aid)) if masks.get(aid) is not None else None,
                    "selected_candidate_ids": sel.get("selected_ids", []),
                    "candidate_pool_size": sel.get("pool_size"),
                    "selected_count": sel.get("selected_count"),
                    "missing_slots": sel.get("missing_slots"),
                    "valid_candidate_count": sel.get("selected_count"),
                    "v2x_received_count": recv.get("count"),
                    "v2x_received_ids": recv.get("sender_ids", []),
                    "v2x_received_ages": recv.get("ages", []),
                    "reward": _safe_float((rewards or {}).get(aid)),
                    "terminated": bool((terminated or {}).get(aid, False)),
                    "truncated": bool((truncated or {}).get(aid, False)),
                    "goal_reached": bool(pa.get("goal_reached", False)),
                    "collision": bool(pa.get("collision", False)),
                    "off_road": bool(pa.get("off_road", False)),
                    "stuck": bool(pa.get("stuck", False)),
                    "stop_reason": pa.get("stop_reason"),
                    "arrival_step": pa.get("arrival_step"),
                    "observation_summary": _extract_observation_summary((obs or {}).get(aid)),
                })

            if _episode_ended(terminated or {}, truncated or {}, info):
                break

        ending_reason = derive_ending_reason(info, getattr(env, "max_steps", cfg.environment.max_steps))
        episode_record.update({
            "execution_status": "success",
            "termination": {
                "ended": True,
                "ending_reason": ending_reason,
                "steps": int(info.get("steps", len({r['step'] for r in step_rows}))),
                "success": bool(info.get("success", False)),
                "collision": bool(info.get("collision", False)),
                "off_road": bool(info.get("off_road", False)),
                "stuck": bool(info.get("stuck", False)),
                "timeout": ending_reason == "timeout",
            },
            "arrival_steps": arrival_steps,
            "team_metrics": {
                "team_reward_total": float(team_reward_total),
                "team_reward_mean_step": (float(team_reward_total) / team_reward_steps) if team_reward_steps else 0.0,
                "min_ttc_s": _safe_float(info.get("min_ttc")),
                "jerk_mps3": _safe_float(info.get("jerk")),
                "duration_steps": int(info.get("steps", 0)),
            },
            "communication_metrics": {
                "v2x_episode": info.get("v2x"),
                "candidate_source": cfg.environment.candidate_source,
            },
            "per_agent_final": (info.get("per_agent") or {}),
            "completed_utc": utc_now_iso(),
        })
    except Exception as e:
        episode_record.update({
            "execution_status": "failed",
            "failure_reason": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "completed_utc": utc_now_iso(),
            "termination": episode_record.get("termination") or {"ended": False},
        })
    finally:
        # Explicit lifecycle/cleanup handling: failure invalidates the episode/pair.
        if env is not None:
            episode_record["lifecycle"]["cleanup_attempted"] = True
            episode_record["lifecycle"]["close_called"] = True
            try:
                env.close()
                episode_record["lifecycle"]["cleanup_status"] = "success"
            except Exception as cleanup_exc:
                episode_record["lifecycle"]["cleanup_status"] = "failed"
                episode_record["lifecycle"]["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                cleanup_msg = f"Cleanup failure: {episode_record['lifecycle']['cleanup_error']}"
                if episode_record.get("execution_status") == "success":
                    episode_record["execution_status"] = "failed"
                    episode_record["failure_reason"] = cleanup_msg
                else:
                    prev = episode_record.get("failure_reason")
                    episode_record["failure_reason"] = f"{prev}; {cleanup_msg}" if prev else cleanup_msg
                # preserve traceback of original failure; attach cleanup traceback separately
                episode_record["cleanup_traceback"] = traceback.format_exc()
        else:
            episode_record["lifecycle"]["cleanup_status"] = "skipped"

    # Persist episode artifacts if layout provided (best-effort; persistence failures recorded)
    episode_record["step_log_count"] = len(step_rows)
    if layout is not None:
        ep_json = layout.paths["episodes"] / f"{ep_id}.json"
        steps_csv = layout.paths["episodes"] / f"{ep_id}_steps.csv"
        raw_json = layout.paths["raw"] / f"{ep_id}_raw.json"
        persistence_errors = []
        csv_rows = []
        try:
            # Prepare step rows (stringify complex fields for CSV)
            for r in step_rows:
                row = dict(r)
                for k, v in list(row.items()):
                    if isinstance(v, (list, dict)):
                        row[k] = json.dumps(v, sort_keys=True, default=str)
                csv_rows.append(row)
            fieldnames = list(csv_rows[0].keys()) if csv_rows else [
                "episode_id", "pair_id", "strategy", "step", "agent_id", "action", "action_mask",
                "selected_candidate_ids", "candidate_pool_size", "selected_count", "missing_slots",
                "valid_candidate_count", "v2x_received_count", "v2x_received_ids", "v2x_received_ages",
                "reward", "terminated", "truncated", "goal_reached", "collision", "off_road", "stuck",
                "stop_reason", "arrival_step", "observation_summary",
            ]
            atomic_write_csv(steps_csv, csv_rows, fieldnames=fieldnames)
            episode_record["artifacts"] = {
                "episode_json": str(ep_json),
                "steps_csv": str(steps_csv),
                "raw_json": str(raw_json),
            }
            # Provisional success if writes below complete and files exist.
            episode_record["persistence"] = {"status": "success", "error": None, "artifacts_present": True}
            atomic_write_json(ep_json, {k: v for k, v in episode_record.items() if k != "traceback"})
            atomic_write_json(raw_json, episode_record)
            missing = [str(p) for p in (ep_json, steps_csv, raw_json) if not Path(p).exists()]
            if missing:
                persistence_errors.append(f"missing artifacts after write: {missing}")
        except Exception as pe:
            persistence_errors.append(f"{type(pe).__name__}: {pe}")

        if persistence_errors:
            episode_record["persistence"] = {"status": "failed", "error": "; ".join(persistence_errors), "artifacts_present": False}
            pmsg = f"Persistence failure: {'; '.join(persistence_errors)}"
            if episode_record.get("execution_status") == "success":
                episode_record["execution_status"] = "failed"
                episode_record["failure_reason"] = pmsg
            else:
                prev = episode_record.get("failure_reason")
                episode_record["failure_reason"] = f"{prev}; {pmsg}" if prev else pmsg
            episode_record["artifacts"] = episode_record.get("artifacts") or {
                "episode_json": str(ep_json),
                "steps_csv": str(steps_csv),
                "raw_json": str(raw_json),
            }
            # Best-effort: write/update records with failure reflected (may rewrite/replace).
            try:
                if not Path(steps_csv).exists():
                    atomic_write_csv(steps_csv, csv_rows, fieldnames=fieldnames if 'fieldnames' in locals() else None)
            except Exception:
                pass
            try:
                atomic_write_json(ep_json, {k: v for k, v in episode_record.items() if k != "traceback"})
            except Exception:
                pass
            try:
                atomic_write_json(raw_json, episode_record)
            except Exception:
                pass

        # Register files and manifest (best-effort). Checksums reflect final written content.
        try:
            if Path(ep_json).exists():
                layout.register_file(ep_json, "episode", extra={"episode_id": ep_id, "pair_id": p_id, "strategy": strategy})
            if Path(steps_csv).exists():
                layout.register_file(steps_csv, "episode_steps", extra={"episode_id": ep_id, "rows": len(csv_rows)})
            if Path(raw_json).exists():
                layout.register_file(raw_json, "raw_episode", extra={"episode_id": ep_id, "status": episode_record.get("execution_status")})
            layout.register_episode({
                "episode_id": ep_id,
                "pair_id": p_id,
                "strategy": strategy,
                "scenario_seed": scenario_seed,
                "episode_index": episode_index,
                "execution_status": episode_record.get("execution_status"),
                "ending_reason": (episode_record.get("termination") or {}).get("ending_reason"),
                "steps": (episode_record.get("termination") or {}).get("steps"),
                "persistence_status": (episode_record.get("persistence") or {}).get("status"),
                "cleanup_status": (episode_record.get("lifecycle") or {}).get("cleanup_status"),
            })
            layout.save_manifest()
        except Exception:
            # best-effort: do not discard in-memory record
            pass
    else:
        episode_record["steps"] = step_rows
        episode_record["persistence"] = {"status": "in_memory", "error": None, "artifacts_present": False}

    return episode_record


def choose_strategy_order(strategies: Sequence[str], rng: random.Random, counterbalanced: bool, pair_index: int) -> List[str]:
    strategies = list(strategies)
    if counterbalanced:
        # Deterministic alternation/counterbalance by pair index.
        return strategies if (pair_index % 2 == 0) else list(reversed(strategies))
    if rng is not None:
        shuffled = list(strategies)
        rng.shuffle(shuffled)
        return shuffled
    return strategies


def run_paired_experiment(
    cfg: Milestone9Config,
    *,
    experiment_id: Optional[str] = None,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    strategy_order_seed: int = 0,
    counterbalance_order: bool = True,
    layout: Optional[ExperimentLayout] = None,
) -> Dict[str, Any]:
    cfg = cfg.validate() if not isinstance(cfg, Milestone9Config) else cfg
    if experiment_id is None:
        experiment_id = make_experiment_id(cfg.to_dict(), label=cfg.experiment_label)
    if layout is None:
        layout = ExperimentLayout(Path(results_root), experiment_id)
    # Explicitly mark running state and persist config/metadata.
    layout.update_status("running", experiment_id=experiment_id, phase="start")
    layout.save_manifest()
    cfg_path = layout.paths["metadata"] / "config.json"
    atomic_write_json(cfg_path, cfg.to_dict())
    layout.register_file(cfg_path, "config")
    layout.save_manifest()

    if pairs is None:
        pairs = [(int(s), int(ep)) for s in cfg.seeds.scenario_seeds for ep in range(cfg.episodes_per_seed)]
    if not pairs:
        raise ValueError("At least one pair (scenario_seed, episode_index) is required")

    order_rng = random.Random(int(strategy_order_seed))
    # Single controller instance reused for all episodes/strategies with identical params.
    controller = build_controller(cfg)

    pair_results = []
    episodes = []
    interrupted = False
    fatal_error = None
    try:
        for pair_index, (scenario_seed, episode_index) in enumerate(pairs):
            p_id = make_pair_id(experiment_id, scenario_seed, episode_index)
            order = choose_strategy_order(cfg.strategies, order_rng, counterbalance_order, pair_index)
            pair_record: Dict[str, Any] = {
                "pair_id": p_id,
                "scenario_seed": int(scenario_seed),
                "episode_index": int(episode_index),
                "strategy_order": list(order),
                "episodes": {},
                "validation": None,
                "valid_for_analysis": False,
                "created_utc": utc_now_iso(),
            }
            for strategy in order:
                ep = run_episode(
                    cfg,
                    experiment_id=experiment_id,
                    scenario_seed=int(scenario_seed),
                    episode_index=int(episode_index),
                    strategy=strategy,
                    controller=controller,
                    layout=layout,
                )
                episodes.append(ep)
                pair_record["episodes"][strategy] = {
                    "episode_id": ep.get("episode_id"),
                    "execution_status": ep.get("execution_status"),
                    "ending_reason": (ep.get("termination") or {}).get("ending_reason"),
                    "artifact_episode_json": (ep.get("artifacts") or {}).get("episode_json"),
                    "persistence_status": (ep.get("persistence") or {}).get("status"),
                    "cleanup_status": (ep.get("lifecycle") or {}).get("cleanup_status"),
                }

            ep_a = next((e for e in episodes if e.get("pair_id") == p_id and e.get("strategy") == cfg.strategies[0]), None)
            ep_b = next((e for e in episodes if e.get("pair_id") == p_id and e.get("strategy") == cfg.strategies[1]), None)
            if ep_a is not None and ep_b is not None:
                validation = validate_pair(ep_a, ep_b)
                pair_record["validation"] = validation.to_dict()
                pair_record["valid_for_analysis"] = bool(validation.valid)
                # Failed member or lifecycle/persistence failure invalidates pair for paired analysis.
                if ep_a.get("execution_status") != "success" or ep_b.get("execution_status") != "success":
                    pair_record["valid_for_analysis"] = False
            else:
                pair_record["validation"] = {"valid": False, "issues": [{"category": "infrastructure_failure", "field": "pair", "message": "Missing episode records"}]}
                pair_record["valid_for_analysis"] = False

            pair_results.append(pair_record)
            # persist pair summary incrementally
            pair_path = layout.paths["summaries"] / f"{p_id}.json"
            atomic_write_json(pair_path, pair_record)
            layout.register_file(pair_path, "pair_summary", extra={"pair_id": p_id})
            layout.save_manifest()
    except Exception as e:
        interrupted = True
        fatal_error = f"{type(e).__name__}: {e}"
    finally:
        all_pairs_processed = len(pair_results) == len(pairs)
        pairs_valid = sum(1 for p in pair_results if p.get("valid_for_analysis"))
        all_pairs_valid = all_pairs_processed and pairs_valid == len(pairs) and len(pairs) > 0
        all_episode_success = all(ep.get("execution_status") == "success" for ep in episodes) if episodes else False
        summary = {
            "experiment_id": experiment_id,
            "created_utc": utc_now_iso(),
            "updated_utc": utc_now_iso(),
            "config": cfg.to_dict(),
            "pairs_requested": [{"scenario_seed": s, "episode_index": e} for s, e in pairs],
            "pairs": pair_results,
            "episodes_count": len(episodes),
            "pairs_processed": len(pair_results),
            "pairs_valid_for_analysis": pairs_valid,
            "all_pairs_processed": all_pairs_processed,
            "all_pairs_valid_for_analysis": all_pairs_valid,
            "interrupted": interrupted,
            "fatal_error": fatal_error,
        }
        try:
            summary_path = layout.paths["summaries"] / "experiment_summary.json"
            atomic_write_json(summary_path, summary)
            layout.register_file(summary_path, "experiment_summary")
        except Exception:
            pass

        if interrupted or not all_pairs_processed:
            final_status = "incomplete" if interrupted else "failed"
            layout.update_status(final_status, experiment_id=experiment_id, pairs=len(pair_results), fatal_error=fatal_error)
        elif all_pairs_valid and all_episode_success:
            layout.update_status("completed", experiment_id=experiment_id, pairs=len(pair_results), all_pairs_valid_for_analysis=True)
        else:
            layout.update_status(
                "failed",
                experiment_id=experiment_id,
                pairs=len(pair_results),
                pairs_valid_for_analysis=pairs_valid,
                all_pairs_valid_for_analysis=False,
                reason="one_or_more_pairs_invalid_or_episodes_failed",
            )
        try:
            layout.save_manifest()
        except Exception:
            pass
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Milestone 9 paired nearest_k/random_k episodes (pilot only when explicitly requested).")
    p.add_argument("--config-json", type=str, default=None, help="Path to Milestone9Config JSON. Defaults to baseline config.")
    p.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT), help="Results root (default results/milestone9).")
    p.add_argument("--experiment-id", type=str, default=None, help="Explicit experiment ID (must not exist/non-empty).")
    p.add_argument("--pilot", action="store_true", help="Run exactly one small pilot pair (requires explicit intent).")
    p.add_argument("--scenario-seed", type=int, default=7, help="Scenario seed for pilot pair.")
    p.add_argument("--episode-index", type=int, default=0, help="Episode index for pilot pair.")
    p.add_argument("--strategy-order-seed", type=int, default=0, help="Seed for strategy-order randomization when not counterbalanced.")
    p.add_argument("--randomize-order", action="store_true", help="Randomize strategy order instead of counterbalancing.")
    p.add_argument("--dry-run-config", action="store_true", help="Validate and print resolved config/pair plan, do not run CARLA.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg = Milestone9Config.from_dict(json.load(f))
    else:
        cfg = default_config()

    pairs = None
    if args.pilot:
        pairs = [(int(args.scenario_seed), int(args.episode_index))]
    elif not args.dry_run_config:
        print("Refusing to run: no explicit pilot requested. Use --pilot for one pair, or --dry-run-config to inspect.", file=sys.stderr)
        return 2

    if args.dry_run_config:
        plan = {
            "config": cfg.to_dict(),
            "pairs": pairs or [(int(s), int(ep)) for s in cfg.seeds.scenario_seeds for ep in range(cfg.episodes_per_seed)],
            "note": "Dry run only; no CARLA launched.",
        }
        print(json.dumps(plan, indent=2, default=str))
        return 0

    summary = run_paired_experiment(
        cfg,
        experiment_id=args.experiment_id,
        results_root=Path(args.results_root),
        pairs=pairs,
        strategy_order_seed=int(args.strategy_order_seed),
        counterbalance_order=not bool(args.randomize_order),
    )
    print(json.dumps({"experiment_id": summary.get("experiment_id"), "pairs_valid_for_analysis": summary.get("pairs_valid_for_analysis"), "episodes_count": summary.get("episodes_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
