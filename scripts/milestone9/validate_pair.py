"""Pair validation for Milestone 9 paired experiments.

Compares two episode records (members of the same pair) and checks initial
scenario/config equivalence required for defensible nearest_k vs random_k
comparison. Later trajectory divergence is expected when strategies/actions
differ and is NOT treated as a mismatch.

CARLA-free: operates on serialized episode/metadata dictionaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_TOLERANCES = {
    "position_m": 1e-3,
    "orientation_deg": 1e-2,
    "speed_mps": 1e-3,
    "route_point_m": 1e-3,
}


@dataclass
class ValidationIssue:
    category: str
    field: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairValidationReport:
    pair_id: Optional[str]
    episode_ids: Tuple[Optional[str], Optional[str]]
    strategies: Tuple[Optional[str], Optional[str]]
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    checked_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "episode_ids": list(self.episode_ids),
            "strategies": list(self.strategies),
            "valid": self.valid,
            "issues": [
                {
                    "category": i.category,
                    "field": i.field,
                    "message": i.message,
                    "details": i.details,
                }
                for i in self.issues
            ],
            "checked_fields": list(self.checked_fields),
        }


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _num_equal(a: Any, b: Any, tol: float) -> bool:
    if a is None and b is None:
        return True
    if not _is_number(a) or not _is_number(b):
        return a == b
    return abs(float(a) - float(b)) <= float(tol)


def _get(d: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _add_issue(report: PairValidationReport, category: str, field: str, message: str, **details: Any) -> None:
    report.issues.append(ValidationIssue(category=category, field=field, message=message, details=details))
    report.valid = False


def _compare_scalar(report: PairValidationReport, field: str, a: Any, b: Any, category: str) -> None:
    report.checked_fields.append(field)
    if a != b:
        _add_issue(report, category, field, f"Mismatch in {field}", a=a, b=b)


def _get_route_applied_value(episode: Mapping[str, Any]) -> tuple[Any, str]:
    """Return (value, source) for route_applied with explicit source tracking."""
    sa = episode.get("initial_scenario") or {}
    if "route_applied" in sa:
        return sa.get("route_applied"), "initial_scenario.route_applied"
    if "route_applied" in episode:
        return episode.get("route_applied"), "episode.route_applied"
    reset_info = episode.get("reset_info") or {}
    if isinstance(reset_info, Mapping) and "route_applied" in reset_info:
        return reset_info.get("route_applied"), "reset_info.route_applied"
    return None, "missing"


def _validate_route_applied(report: PairValidationReport, label: str, episode: Mapping[str, Any]) -> bool:
    """Strict route-validity precondition for one episode.

    Required: route_applied exists, is exactly bool, and is True.
    Missing/False/None/malformed values invalidate the pair.
    """
    value, source = _get_route_applied_value(episode)
    field = f"route_applied.{label}"
    report.checked_fields.append(field)

    if source == "missing":
        _add_issue(
            report,
            "infrastructure_failure",
            field,
            f"route_applied is missing for episode {label}",
            value=value,
            source=source,
            episode_id=episode.get("episode_id"),
        )
        return False
    if type(value) is not bool:
        _add_issue(
            report,
            "infrastructure_failure",
            field,
            f"route_applied must be a boolean for episode {label}",
            value=value,
            value_type=type(value).__name__,
            source=source,
            episode_id=episode.get("episode_id"),
        )
        return False
    if value is not True:
        _add_issue(
            report,
            "infrastructure_failure",
            field,
            f"route_applied must be True for episode {label}",
            value=value,
            source=source,
            episode_id=episode.get("episode_id"),
        )
        return False
    return True


def _compare_float(report: PairValidationReport, field: str, a: Any, b: Any, tol: float, category: str) -> None:
    report.checked_fields.append(field)
    if not _num_equal(a, b, tol):
        _add_issue(report, category, field, f"Mismatch in {field} beyond tolerance {tol}", a=a, b=b, tolerance=tol)


def _compare_point(report: PairValidationReport, field: str, a: Any, b: Any, tol: float, category: str) -> None:
    report.checked_fields.append(field)
    if a is None or b is None:
        if a != b:
            _add_issue(report, category, field, f"Mismatch in {field}", a=a, b=b)
        return
    keys = ("x", "y", "z")
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        _add_issue(report, category, field, f"Non-mapping point in {field}", a=a, b=b)
        return
    for k in keys:
        if k in a or k in b:
            if not _num_equal(a.get(k), b.get(k), tol):
                _add_issue(
                    report,
                    category,
                    field,
                    f"Mismatch in {field}.{k} beyond tolerance {tol}",
                    a=a,
                    b=b,
                    tolerance=tol,
                )
                return


def _compare_orientation(report: PairValidationReport, field: str, a: Any, b: Any, tol_deg: float, category: str) -> None:
    report.checked_fields.append(field)
    if a is None or b is None:
        if a != b:
            _add_issue(report, category, field, f"Mismatch in {field}", a=a, b=b)
        return
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        _add_issue(report, category, field, f"Non-mapping orientation in {field}", a=a, b=b)
        return
    # CARLA uses pitch/yaw/roll in degrees; compare cyclic yaw conservatively.
    for k in ("pitch", "roll"):
        if k in a or k in b:
            if not _num_equal(a.get(k), b.get(k), tol_deg):
                _add_issue(report, category, field, f"Mismatch in {field}.{k}", a=a, b=b, tolerance=tol_deg)
                return
    if "yaw" in a or "yaw" in b:
        ya, yb = a.get("yaw"), b.get("yaw")
        if not (_is_number(ya) and _is_number(yb)):
            if ya != yb:
                _add_issue(report, category, field, f"Mismatch in {field}.yaw", a=a, b=b)
            return
        diff = abs(float(ya) - float(yb)) % 360.0
        diff = min(diff, 360.0 - diff)
        if diff > tol_deg:
            _add_issue(report, category, field, f"Mismatch in {field}.yaw beyond tolerance {tol_deg}", a=a, b=b, tolerance=tol_deg, diff=diff)


def _compare_route(report: PairValidationReport, field: str, route_a: Any, route_b: Any, tol: float, category: str) -> None:
    report.checked_fields.append(field)
    if route_a is None or route_b is None:
        if route_a != route_b:
            _add_issue(report, category, field, f"Mismatch in {field}", a=route_a, b=route_b)
        return
    if not isinstance(route_a, Sequence) or isinstance(route_a, (str, bytes)) or not isinstance(route_b, Sequence) or isinstance(route_b, (str, bytes)):
        _add_issue(report, category, field, f"Routes must be sequences in {field}", a=route_a, b=route_b)
        return
    if len(route_a) != len(route_b):
        _add_issue(report, category, field, f"Route length mismatch in {field}", len_a=len(route_a), len_b=len(route_b))
        return
    for idx, (pa, pb) in enumerate(zip(route_a, route_b)):
        if not isinstance(pa, Mapping) or not isinstance(pb, Mapping):
            if pa != pb:
                _add_issue(report, category, f"{field}[{idx}]", f"Route point mismatch at {idx}", a=pa, b=pb)
                return
            continue
        for k in ("x", "y", "z"):
            if k in pa or k in pb:
                if not _num_equal(pa.get(k), pb.get(k), tol):
                    _add_issue(report, category, f"{field}[{idx}]", f"Route point mismatch at {idx}", a=pa, b=pb, tolerance=tol)
                    return


def _classify_initial_mismatch(field: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    # Initial scenario/config mismatches invalidate pairing; CARLA nondeterminism or
    # infrastructure issues are common causes but cannot be proven from records alone.
    # We classify as initial_scenario_mismatch for scenario/state/config fields and
    # let details guide interpretation.
    _ = (a, b)
    if field.startswith("execution") or field.endswith("status"):
        return "infrastructure_failure"
    return "initial_scenario_mismatch"


def validate_pair(
    episode_a: Mapping[str, Any],
    episode_b: Mapping[str, Any],
    *,
    tolerances: Optional[Mapping[str, float]] = None,
) -> PairValidationReport:
    """Validate that two paired episodes started from equivalent scenarios/config.

    The checker never marks a mismatched pair as valid. It does not compare
    later trajectories except to note that divergence is expected when actions
    differ (no failure issued for trajectory differences).
    """
    tol = dict(DEFAULT_TOLERANCES)
    if tolerances:
        tol.update({k: float(v) for k, v in tolerances.items()})

    report = PairValidationReport(
        pair_id=episode_a.get("pair_id") or episode_b.get("pair_id"),
        episode_ids=(episode_a.get("episode_id"), episode_b.get("episode_id")),
        strategies=(episode_a.get("strategy"), episode_b.get("strategy")),
        valid=True,
    )

    # Pair identity
    for field in ("pair_id", "scenario_seed", "episode_index"):
        va, vb = episode_a.get(field), episode_b.get(field)
        report.checked_fields.append(f"identity.{field}")
        if va != vb:
            _add_issue(report, "initial_scenario_mismatch", f"identity.{field}", f"Pair identity mismatch in {field}", a=va, b=vb)

    # Execution/infrastructure failures (recorded explicitly; pair invalid for analysis)
    for label, ep in (("a", episode_a), ("b", episode_b)):
        status = ep.get("execution_status")
        report.checked_fields.append(f"execution.{label}.status")
        if status not in (None, "success", "completed"):
            _add_issue(
                report,
                "infrastructure_failure",
                f"execution.{label}.status",
                f"Episode {label} did not complete successfully",
                status=status,
                failure_reason=ep.get("failure_reason"),
                episode_id=ep.get("episode_id"),
            )
        lifecycle = ep.get("lifecycle") or {}
        cleanup_status = lifecycle.get("cleanup_status")
        report.checked_fields.append(f"lifecycle.{label}.cleanup_status")
        if cleanup_status == "failed":
            _add_issue(
                report,
                "infrastructure_failure",
                f"lifecycle.{label}.cleanup_status",
                f"Episode {label} cleanup failed (lifecycle isolation not established)",
                cleanup_status=cleanup_status,
                cleanup_error=lifecycle.get("cleanup_error"),
                episode_id=ep.get("episode_id"),
            )
        persistence = ep.get("persistence") or {}
        persistence_status = persistence.get("status")
        report.checked_fields.append(f"persistence.{label}.status")
        if persistence_status == "failed":
            _add_issue(
                report,
                "infrastructure_failure",
                f"persistence.{label}.status",
                f"Episode {label} has persistence/artifact failure",
                persistence_status=persistence_status,
                persistence_error=persistence.get("error"),
                artifacts_present=persistence.get("artifacts_present"),
                episode_id=ep.get("episode_id"),
            )

    # Config equivalence (must be identical except strategy and strategy-specific selection seed)
    cfg_a = episode_a.get("config", {}) or {}
    cfg_b = episode_b.get("config", {}) or {}
    for field in ("town", "n_agents", "max_steps", "spawn_mode", "spawn_cluster_max_distance_m", "candidate_source"):
        va = _get(cfg_a, ["environment", field], cfg_a.get(field))
        vb = _get(cfg_b, ["environment", field], cfg_b.get(field))
        # fallback top-level
        if va is None:
            va = episode_a.get(field)
        if vb is None:
            vb = episode_b.get(field)
        _compare_scalar(report, f"config.{field}", va, vb, "initial_scenario_mismatch")

    _compare_scalar(report, "config.v2x", cfg_a.get("v2x", episode_a.get("v2x_config")), cfg_b.get("v2x", episode_b.get("v2x_config")), "initial_scenario_mismatch")
    _compare_scalar(report, "config.controller", cfg_a.get("controller", episode_a.get("controller")), cfg_b.get("controller", episode_b.get("controller")), "initial_scenario_mismatch")

    # Strict route-validity precondition: required, actual bool, and True for both members.
    route_a_ok = _validate_route_applied(report, "a", episode_a)
    route_b_ok = _validate_route_applied(report, "b", episode_b)
    # Only compare route_applied values for consistency when both are individually valid.
    if route_a_ok and route_b_ok:
        va, _ = _get_route_applied_value(episode_a)
        vb, _ = _get_route_applied_value(episode_b)
        _compare_scalar(report, "route_applied.consistency", va, vb, "infrastructure_failure")

    # Initial scenario
    sa = episode_a.get("initial_scenario", {}) or {}
    sb = episode_b.get("initial_scenario", {}) or {}
    agents_a = sa.get("agents") or list((sa.get("initial_state") or {}).keys()) or episode_a.get("agents_order") or []
    agents_b = sb.get("agents") or list((sb.get("initial_state") or {}).keys()) or episode_b.get("agents_order") or []
    report.checked_fields.append("initial_scenario.vehicle_count")
    if len(agents_a) != len(agents_b):
        _add_issue(report, "initial_scenario_mismatch", "initial_scenario.vehicle_count", "Vehicle count mismatch", a=len(agents_a), b=len(agents_b), agents_a=agents_a, agents_b=agents_b)
    report.checked_fields.append("initial_scenario.agent_ids")
    if list(agents_a) != list(agents_b):
        _add_issue(report, "initial_scenario_mismatch", "initial_scenario.agent_ids", "Agent ID/order mismatch", a=list(agents_a), b=list(agents_b))

    # Required snapshot schema/policy checks (missing required properties invalidate pair;
    # identical missing required information is not accepted).
    def _require_mapping(ep_label: str, container: Any, field: str):
        report.checked_fields.append(field)
        if not isinstance(container, Mapping):
            _add_issue(report, "infrastructure_failure", field, f"Required snapshot mapping missing/invalid for episode {ep_label}", value=container)
            return False
        return True

    for ep_label, s in (("a", sa), ("b", sb)):
        has_world = _require_mapping(ep_label, s.get("world"), f"snapshot.{ep_label}.world")
        if has_world:
            world = s.get("world") or {}
            for req in ("map_name", "world_settings", "weather"):
                report.checked_fields.append(f"snapshot.{ep_label}.world.{req}")
                if req not in world:
                    _add_issue(report, "infrastructure_failure", f"snapshot.{ep_label}.world.{req}", f"Required world field '{req}' missing for episode {ep_label}")
            ws = world.get("world_settings")
            if not isinstance(ws, Mapping):
                _add_issue(report, "infrastructure_failure", f"snapshot.{ep_label}.world.world_settings", f"world_settings must be a mapping for episode {ep_label}", value=ws)
            else:
                for req in ("synchronous_mode", "fixed_delta_seconds"):
                    report.checked_fields.append(f"snapshot.{ep_label}.world.world_settings.{req}")
                    if req not in ws:
                        _add_issue(report, "infrastructure_failure", f"snapshot.{ep_label}.world.world_settings.{req}", f"Required world setting '{req}' missing for episode {ep_label}")

        has_ctrl = _require_mapping(ep_label, s.get("controlled_agents"), f"snapshot.{ep_label}.controlled_agents")
        if has_ctrl:
            ca = s.get("controlled_agents") or {}
            agents_ref = agents_a if ep_label == "a" else agents_b
            for aid in list(agents_ref):
                report.checked_fields.append(f"snapshot.{ep_label}.controlled_agents.{aid}")
                if aid not in ca or not isinstance(ca.get(aid), Mapping):
                    _add_issue(report, "infrastructure_failure", f"snapshot.{ep_label}.controlled_agents.{aid}", f"Controlled agent descriptor missing for {aid} in episode {ep_label}")
                    continue
                cad = ca.get(aid) or {}
                if "experiment_agent_id" not in cad:
                    _add_issue(report, "infrastructure_failure", f"snapshot.{ep_label}.controlled_agents.{aid}.experiment_agent_id", f"experiment_agent_id missing for {aid} in episode {ep_label}")
                if "route_status" not in cad or not isinstance(cad.get("route_status"), Mapping):
                    _add_issue(report, "infrastructure_failure", f"snapshot.{ep_label}.controlled_agents.{aid}.route_status", f"route_status missing/invalid for {aid} in episode {ep_label}")

        has_bg = _require_mapping(ep_label, s.get("background_and_traffic_lights"), f"snapshot.{ep_label}.background_and_traffic_lights")
        if has_bg:
            bg = s.get("background_and_traffic_lights") or {}
            for req in ("vehicles", "traffic_lights"):
                report.checked_fields.append(f"snapshot.{ep_label}.background_and_traffic_lights.{req}")
                if req not in bg or not isinstance(bg.get(req), list):
                    _add_issue(report, "infrastructure_failure", f"snapshot.{ep_label}.background_and_traffic_lights.{req}", f"Required snapshot list '{req}' missing/invalid for episode {ep_label}")

    # Must-match world/simulator settings when required fields are present (no pass on identical missing required info handled above)
    wa = (sa.get("world") or {}) if isinstance(sa.get("world"), Mapping) else {}
    wb = (sb.get("world") or {}) if isinstance(sb.get("world"), Mapping) else {}
    if "map_name" in wa and "map_name" in wb:
        _compare_scalar(report, "snapshot.world.map_name", wa.get("map_name"), wb.get("map_name"), "initial_scenario_mismatch")
    wsa = wa.get("world_settings") if isinstance(wa.get("world_settings"), Mapping) else {}
    wsb = wb.get("world_settings") if isinstance(wb.get("world_settings"), Mapping) else {}
    for key in ("synchronous_mode", "fixed_delta_seconds"):
        if isinstance(wsa, Mapping) and isinstance(wsb, Mapping) and key in wsa and key in wsb:
            _compare_scalar(report, f"snapshot.world.world_settings.{key}", wsa.get(key), wsb.get(key), "initial_scenario_mismatch")

    # Controlled-agent route status must indicate applied (and match) when present.
    ca_a = sa.get("controlled_agents") if isinstance(sa.get("controlled_agents"), Mapping) else {}
    ca_b = sb.get("controlled_agents") if isinstance(sb.get("controlled_agents"), Mapping) else {}
    for agent_id in list(agents_a):
        ra = ((ca_a or {}).get(agent_id) or {}).get("route_status") if isinstance(ca_a, Mapping) else None
        rb = ((ca_b or {}).get(agent_id) or {}).get("route_status") if isinstance(ca_b, Mapping) else None
        if isinstance(ra, Mapping) and "route_applied" in ra:
            # per-agent route_applied must be strict True boolean as part of required snapshot
            va = ra.get("route_applied")
            report.checked_fields.append(f"snapshot.controlled_agents.{agent_id}.route_status.route_applied.a")
            if type(va) is not bool or va is not True:
                _add_issue(report, "infrastructure_failure", f"snapshot.controlled_agents.{agent_id}.route_status.route_applied.a", f"Per-agent route_applied must be True boolean for {agent_id} in episode a", value=va)
        if isinstance(rb, Mapping) and "route_applied" in rb:
            vb = rb.get("route_applied")
            report.checked_fields.append(f"snapshot.controlled_agents.{agent_id}.route_status.route_applied.b")
            if type(vb) is not bool or vb is not True:
                _add_issue(report, "infrastructure_failure", f"snapshot.controlled_agents.{agent_id}.route_status.route_applied.b", f"Per-agent route_applied must be True boolean for {agent_id} in episode b", value=vb)

    # Diagnostic-only/background policy: captured for audit; unstable CARLA actor IDs and
    # best-effort traffic-light/background details are not required to match identically.
    report.checked_fields.append("snapshot.background_and_traffic_lights.policy")
    report.checked_fields.append("snapshot.controlled_agents.carla_actor_id_diagnostic.policy")

    state_a = sa.get("initial_state") or {}
    state_b = sb.get("initial_state") or {}
    for agent_id in list(agents_a):
        pa = state_a.get(agent_id, {})
        pb = state_b.get(agent_id, {})
        _compare_point(report, f"initial_state.{agent_id}.position", pa.get("position"), pb.get("position"), tol["position_m"], "initial_scenario_mismatch")
        _compare_orientation(report, f"initial_state.{agent_id}.orientation", pa.get("orientation") or pa.get("rotation"), pb.get("orientation") or pb.get("rotation"), tol["orientation_deg"], "initial_scenario_mismatch")
        speed_a = pa.get("speed_mps", pa.get("speed"))
        speed_b = pb.get("speed_mps", pb.get("speed"))
        _compare_float(report, f"initial_state.{agent_id}.speed_mps", speed_a, speed_b, tol["speed_mps"], "initial_scenario_mismatch")
        _compare_point(report, f"initial_state.{agent_id}.destination", pa.get("destination") or pa.get("goal"), pb.get("destination") or pb.get("goal"), tol["route_point_m"], "initial_scenario_mismatch")
        _compare_route(report, f"initial_state.{agent_id}.route", pa.get("route"), pb.get("route"), tol["route_point_m"], "initial_scenario_mismatch")

    # Spawn metadata/config (route_applied is enforced as a strict required precondition above)
    for key in ("spawn_mode", "spawn_cluster_max_distance", "spawn_cluster_max_distance_m"):
        va = sa.get(key, episode_a.get(key))
        vb = sb.get(key, episode_b.get(key))
        if va is not None or vb is not None:
            _compare_scalar(report, f"spawn_metadata.{key}", va, vb, "initial_scenario_mismatch")

    # Episode horizon
    ha = _get(cfg_a, ["environment", "max_steps"], episode_a.get("episode_horizon", episode_a.get("max_steps")))
    hb = _get(cfg_b, ["environment", "max_steps"], episode_b.get("episode_horizon", episode_b.get("max_steps")))
    _compare_scalar(report, "config.episode_horizon", ha, hb, "initial_scenario_mismatch")

    # Note: later trajectory divergence is expected and intentionally not compared.
    report.checked_fields.append("later_trajectory")
    # no issue added

    # If there are mismatches, try to annotate plausible causes in details (non-exclusive)
    if not report.valid:
        for issue in report.issues:
            if issue.category == "initial_scenario_mismatch" and "plausible_causes" not in issue.details:
                issue.details["plausible_causes"] = [
                    "CARLA/TM nondeterminism",
                    "infrastructure or reset-order difference",
                    "spawn/route generation mismatch",
                ]
            if issue.category == "infrastructure_failure" and "plausible_causes" not in issue.details:
                issue.details["plausible_causes"] = [
                    "CARLA connection/world issue",
                    "runner exception or incomplete episode",
                ]

    return report


def pair_is_valid_for_analysis(report: PairValidationReport) -> bool:
    """True only when no mismatches/failures were detected."""
    return bool(report.valid)


__all__ = [
    "DEFAULT_TOLERANCES",
    "PairValidationReport",
    "ValidationIssue",
    "pair_is_valid_for_analysis",
    "validate_pair",
]
