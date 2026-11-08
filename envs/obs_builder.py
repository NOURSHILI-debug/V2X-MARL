"""Pure-NumPy construction of the normalized 56-value CARLA observation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np

EGO = 8
ROUTE = 6
M_CANDIDATES = 6
CAND_FEATURES = 6
CAND = M_CANDIDATES * CAND_FEATURES
MASK = 6
OBS_DIM = EGO + ROUTE + CAND + MASK

# All observation scaling is defined here so the CARLA adapter and tests share it.
V_MAX = 30.0  # m/s
A_MAX = 8.0  # m/s^2
D_MAX = 150.0  # m
LANE_WIDTH_MAX = 6.0  # m
MAX_LANES = 6.0
ROUTE_FRAME_MAX = 20.0  # m
CANDIDATE_RANGE = 50.0  # m

KEEP, ACCEL, DECEL, EMERGENCY_BRAKE, LANE_LEFT, LANE_RIGHT = range(MASK)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return float(np.clip(_finite(value), lower, upper))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(_finite(angle)), math.cos(_finite(angle)))


def to_ego_frame(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    yaw: float,
) -> tuple[float, float]:
    """Translate a world point and rotate it into the ego vehicle frame."""
    dx = _finite(x) - _finite(origin_x)
    dy = _finite(y) - _finite(origin_y)
    yaw = _finite(yaw)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def make_action_mask(
    left_lane_exists: bool,
    right_lane_exists: bool,
) -> np.ndarray:
    """Return validity for keep/accelerate/decelerate/brake/left/right."""
    mask = np.ones(MASK, dtype=bool)
    mask[LANE_LEFT] = bool(left_lane_exists)
    mask[LANE_RIGHT] = bool(right_lane_exists)
    mask[EMERGENCY_BRAKE] = True
    return mask


def _traffic_light_feature(value: object) -> float:
    if isinstance(value, str):
        state = value.strip().lower()
        if state == "red":
            return 1.0
        if state == "yellow":
            return 0.5
        return 0.0
    return float(np.clip(_finite(value), 0.0, 1.0))


def build_observation(
    ego: Mapping[str, object],
    route: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    action_mask: Sequence[bool],
) -> np.ndarray:
    """Build one normalized observation from CARLA-independent numeric records.

    Ego records provide world position/velocity/yaw plus normalized-feature inputs:
    speed, longitudinal_accel, heading_error, lateral_offset, lane_index,
    lane_count, distance_to_goal, and progress_fraction. Route points and candidate
    positions/velocities/headings are in world coordinates.
    """
    mask = np.asarray(action_mask, dtype=bool)
    if mask.shape != (MASK,):
        raise ValueError(f"action_mask must have shape ({MASK},)")
    mask = mask.copy()
    mask[EMERGENCY_BRAKE] = True

    yaw = _finite(ego.get("yaw", 0.0))
    origin_x = _finite(ego.get("x", 0.0))
    origin_y = _finite(ego.get("y", 0.0))
    ego_vx = _finite(ego.get("vx", 0.0))
    ego_vy = _finite(ego.get("vy", 0.0))
    lane_width = max(_finite(ego.get("lane_width", LANE_WIDTH_MAX)), 1e-6)

    ego_block = [
        _clip(_finite(ego.get("speed", 0.0)) / V_MAX, 0.0, 1.0),
        _clip(_finite(ego.get("longitudinal_accel", 0.0)) / A_MAX),
        _clip(_finite(ego.get("heading_error", 0.0)) / math.pi),
        _clip(_finite(ego.get("lateral_offset", 0.0)) / lane_width),
        _clip(_finite(ego.get("lane_index", 0.0)) / MAX_LANES, 0.0, 1.0),
        _clip(_finite(ego.get("lane_count", 1.0)) / MAX_LANES, 0.0, 1.0),
        _clip(_finite(ego.get("distance_to_goal", 0.0)) / D_MAX, 0.0, 1.0),
        _clip(_finite(ego.get("progress_fraction", 0.0)), 0.0, 1.0),
    ]

    next_x = _finite(route.get("next_waypoint_x", origin_x))
    next_y = _finite(route.get("next_waypoint_y", origin_y))
    route_x, route_y = to_ego_frame(next_x, next_y, origin_x, origin_y, yaw)
    route_block = [
        _clip(route_x / ROUTE_FRAME_MAX),
        _clip(route_y / ROUTE_FRAME_MAX),
        _clip(_finite(route.get("speed_limit", 0.0)) / V_MAX, 0.0, 1.0),
        _clip(float(bool(route.get("at_intersection", False))), 0.0, 1.0),
        _traffic_light_feature(route.get("traffic_light_state", 0.0)),
        _clip(_finite(route.get("distance_to_next", 0.0)) / ROUTE_FRAME_MAX, 0.0, 1.0),
    ]

    sortable_candidates: list[tuple[float, int, Mapping[str, object]]] = []
    for candidate in candidates:
        cx = _finite(candidate.get("x", origin_x))
        cy = _finite(candidate.get("y", origin_y))
        distance = math.hypot(cx - origin_x, cy - origin_y)
        if distance <= CANDIDATE_RANGE and bool(candidate.get("valid", True)):
            sortable_candidates.append(
                (distance, int(_finite(candidate.get("actor_id", 0))), candidate)
            )
    sortable_candidates.sort(key=lambda item: (item[0], item[1]))

    candidate_block = np.zeros((M_CANDIDATES, CAND_FEATURES), dtype=np.float32)
    for index, (_, _, candidate) in enumerate(sortable_candidates[:M_CANDIDATES]):
        cx, cy = _finite(candidate.get("x", origin_x)), _finite(candidate.get("y", origin_y))
        rel_x, rel_y = to_ego_frame(cx, cy, origin_x, origin_y, yaw)
        rel_vx, rel_vy = to_ego_frame(
            _finite(candidate.get("vx", 0.0)) - ego_vx,
            _finite(candidate.get("vy", 0.0)) - ego_vy,
            0.0,
            0.0,
            yaw,
        )
        rel_heading = _wrap_angle(_finite(candidate.get("yaw", yaw)) - yaw)
        candidate_block[index] = (
            _clip(rel_x / CANDIDATE_RANGE),
            _clip(rel_y / CANDIDATE_RANGE),
            _clip(rel_vx / V_MAX),
            _clip(rel_vy / V_MAX),
            _clip(rel_heading / math.pi),
            1.0,
        )

    observation = np.concatenate(
        (
            np.asarray(ego_block, dtype=np.float32),
            np.asarray(route_block, dtype=np.float32),
            candidate_block.reshape(-1),
            mask.astype(np.float32),
        )
    )
    return np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0).clip(
        -1.0, 1.0
    ).astype(np.float32, copy=False)


__all__ = [
    "A_MAX",
    "CAND",
    "CANDIDATE_RANGE",
    "D_MAX",
    "EGO",
    "LANE_WIDTH_MAX",
    "MASK",
    "MAX_LANES",
    "M_CANDIDATES",
    "OBS_DIM",
    "ROUTE",
    "ROUTE_FRAME_MAX",
    "V_MAX",
    "build_observation",
    "make_action_mask",
    "to_ego_frame",
]
