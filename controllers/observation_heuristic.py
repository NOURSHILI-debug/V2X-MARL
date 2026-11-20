"""Deterministic observation-conditioned heuristic controller (Milestone 8A).

Consumes the existing 56-D observation produced by ``envs.obs_builder`` and
returns one of the six canonical meta-actions defined by the environment action
interface (semantics unchanged).

The controller is deterministic for identical observations/configuration, has no
CARLA dependency, and uses no global random state. Candidate (selected teammate)
features are read explicitly and can influence decisions (collision-avoidance,
safe-following, and lane-clearance), enabling paired nearest_k/random_k
evaluation with identical controller logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from envs.obs_builder import (
    CAND_FEATURES,
    CANDIDATE_RANGE,
    EGO,
    KEEP,
    ACCEL,
    DECEL,
    EMERGENCY_BRAKE,
    LANE_LEFT,
    LANE_RIGHT,
    M_CANDIDATES,
    MASK,
    OBS_DIM,
    ROUTE,
    V_MAX,
)

# Observation indices (verified against envs/obs_builder.py)
# Ego [0:8]: speed/V_MAX, lon_accel/A_MAX, heading_error/pi,
# lateral_offset/lane_width, lane_index/MAX_LANES, lane_count/MAX_LANES,
# distance_to_goal/D_MAX, progress_fraction
IDX_EGO_SPEED = 0
IDX_EGO_LON_ACCEL = 1
IDX_EGO_HEADING_ERROR = 2
IDX_EGO_LATERAL_OFFSET = 3
IDX_EGO_LANE_INDEX = 4
IDX_EGO_LANE_COUNT = 5
IDX_EGO_DIST_GOAL = 6
IDX_EGO_PROGRESS = 7

# Route [8:14]: route_x/ROUTE_FRAME_MAX, route_y/ROUTE_FRAME_MAX,
# speed_limit/V_MAX, at_intersection, traffic_light(0/0.5/1),
# distance_to_next/ROUTE_FRAME_MAX
IDX_ROUTE_X = EGO + 0
IDX_ROUTE_Y = EGO + 1
IDX_ROUTE_SPEED_LIMIT = EGO + 2
IDX_ROUTE_AT_INTERSECTION = EGO + 3
IDX_ROUTE_TRAFFIC_LIGHT = EGO + 4
IDX_ROUTE_DIST_NEXT = EGO + 5

CAND_START = EGO + ROUTE
MASK_START = EGO + ROUTE + M_CANDIDATES * CAND_FEATURES

# Candidate slot: [rel_x/CANDIDATE_RANGE, rel_y/CANDIDATE_RANGE, rel_vx/V_MAX,
# rel_vy/V_MAX, rel_heading/pi, valid]
C_REL_X = 0
C_REL_Y = 1
C_REL_VX = 2
C_REL_VY = 3
C_REL_HEADING = 4
C_VALID = 5


@dataclass(frozen=True)
class Candidate:
    """Parsed view of one candidate slot in normalized ego-frame units."""

    slot: int
    rel_x: float
    rel_y: float
    rel_vx: float
    rel_vy: float
    rel_heading: float
    valid: bool


class ObservationHeuristicController:
    """Deterministic heuristic using ego, route, mask, and valid candidates.

    Decision logic (first applicable, mask-compliant):
      1) Emergency brake for immediate/critical closing threat ahead (candidate-based).
      2) Brake/decelerate for red/yellow traffic-light hazard or caution threat ahead.
      3) If blocked ahead, attempt lane change only to a mask-allowed side that is
         clear of valid candidates in the adjacent lane zone (safe-following/avoidance).
      4) Speed regulation vs route speed limit, gated by candidate proximity ahead.

    Thresholds are in normalized observation units (CANDIDATE_RANGE=50 m, V_MAX=30 m/s,
    ROUTE_FRAME_MAX=20 m).
    """

    SAME_LANE_HALF_WIDTH = 0.08   # ~4 m
    SIDE_LANE_INNER = 0.03        # ~1.5 m
    SIDE_LANE_OUTER = 0.18        # ~9 m

    AHEAD_MIN_X = -0.02           # -1 m
    EMERGENCY_X = 0.10            # 5 m
    BRAKE_X = 0.16                # 8 m
    CAUTION_X = 0.30              # 15 m
    CLEAR_AHEAD_X = 0.40          # 20 m

    CLOSING_FAST = -0.05          # -1.5 m/s
    CLOSING_MODERATE = -0.015     # -0.45 m/s

    RED_LIGHT = 0.75
    YELLOW_LIGHT = 0.25
    RED_LIGHT_DIST = 0.25
    YELLOW_LIGHT_DIST = 0.35
    SPEED_MARGIN = 0.03

    def __init__(self) -> None:
        self.last_action: int | None = None
        self.last_reason: str | None = None

    @staticmethod
    def validate_observation(obs: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return a finite float32 copy of a 56-D observation.

        Wrong shape raises ValueError. Non-finite numerical values are sanitized
        defensively to finite values so the controller remains safe and deterministic.
        """
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim != 1 or arr.shape != (OBS_DIM,):
            raise ValueError(f"observation must have shape ({OBS_DIM},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        arr = np.clip(arr, -1.0, 1.0)
        return arr.astype(np.float32, copy=True)

    @staticmethod
    def parse_action_mask(
        obs: np.ndarray,
        action_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a valid 6-action mask; emergency brake is always available."""
        if action_mask is None:
            mask = np.asarray(obs[MASK_START:MASK_START + MASK], dtype=np.float32) > 0.5
        else:
            mask = np.asarray(action_mask, dtype=bool).reshape(-1)
            if mask.shape != (MASK,):
                raise ValueError(f"action_mask must have shape ({MASK},), got {mask.shape}")
        mask = mask.copy()
        input_any = bool(mask.any())
        mask[EMERGENCY_BRAKE] = True
        if not input_any:
            mask[KEEP] = True
        return mask.astype(bool, copy=False)

    @staticmethod
    def parse_candidates(obs: np.ndarray) -> list[Candidate]:
        """Extract candidate slots; validity flag and finiteness are enforced."""
        out: list[Candidate] = []
        block = obs[CAND_START:MASK_START]
        for slot in range(M_CANDIDATES):
            vals = block[slot * CAND_FEATURES:(slot + 1) * CAND_FEATURES]
            finite = all(math.isfinite(float(v)) for v in vals)
            valid = bool(finite and float(vals[C_VALID]) > 0.5)
            out.append(
                Candidate(
                    slot=slot,
                    rel_x=float(vals[C_REL_X]),
                    rel_y=float(vals[C_REL_Y]),
                    rel_vx=float(vals[C_REL_VX]),
                    rel_vy=float(vals[C_REL_VY]),
                    rel_heading=float(vals[C_REL_HEADING]),
                    valid=valid,
                )
            )
        return out

    @staticmethod
    def _valid_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
        return [c for c in candidates if c.valid]

    def _ahead_same_lane(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        ahead = [
            c for c in self._valid_candidates(candidates)
            if c.rel_x >= self.AHEAD_MIN_X and abs(c.rel_y) <= self.SAME_LANE_HALF_WIDTH
        ]
        ahead.sort(key=lambda c: (c.rel_x, c.slot))
        return ahead

    def _lane_zone_clear(self, candidates: Sequence[Candidate], direction: int, x_limit: float) -> bool:
        """True if no valid candidate occupies the target adjacent lane zone ahead."""
        for c in self._valid_candidates(candidates):
            if c.rel_x < self.AHEAD_MIN_X or c.rel_x > x_limit:
                continue
            if direction == LANE_LEFT:
                in_zone = self.SIDE_LANE_INNER <= c.rel_y <= self.SIDE_LANE_OUTER
                straddle = 0.0 < c.rel_y < self.SIDE_LANE_OUTER and c.rel_x < self.CAUTION_X and abs(c.rel_y) > self.SAME_LANE_HALF_WIDTH * 0.5
            elif direction == LANE_RIGHT:
                in_zone = -self.SIDE_LANE_OUTER <= c.rel_y <= -self.SIDE_LANE_INNER
                straddle = -self.SIDE_LANE_OUTER < c.rel_y < 0.0 and c.rel_x < self.CAUTION_X and abs(c.rel_y) > self.SAME_LANE_HALF_WIDTH * 0.5
            else:
                return False
            if in_zone or straddle:
                return False
        return True

    def _has_immediate_threat(self, ahead: Sequence[Candidate]) -> bool:
        for c in ahead:
            if c.rel_x <= self.EMERGENCY_X:
                return True
            if c.rel_x <= self.BRAKE_X and c.rel_vx <= self.CLOSING_FAST:
                return True
        return False

    def _has_caution_threat(self, ahead: Sequence[Candidate]) -> bool:
        for c in ahead:
            if c.rel_x <= self.BRAKE_X:
                return True
            if c.rel_x <= self.CAUTION_X and c.rel_vx <= self.CLOSING_MODERATE:
                return True
            if c.rel_x <= self.CAUTION_X and c.rel_vx <= 0.0:
                return True
        return False

    @staticmethod
    def _choose_allowed(preferred: Sequence[int], mask: np.ndarray) -> int:
        for action in preferred:
            a = int(action)
            if 0 <= a < MASK and bool(mask[a]):
                return a
        for action in (EMERGENCY_BRAKE, DECEL, KEEP, ACCEL, LANE_LEFT, LANE_RIGHT):
            if bool(mask[action]):
                return int(action)
        return int(EMERGENCY_BRAKE)

    def act(self, obs: Sequence[float] | np.ndarray, action_mask: Sequence[bool] | np.ndarray | None = None) -> int:
        """Return a deterministic, mask-compliant action in {0..5}."""
        arr = self.validate_observation(obs)
        mask = self.parse_action_mask(arr, action_mask=action_mask)
        candidates = self.parse_candidates(arr)
        ahead = self._ahead_same_lane(candidates)

        ego_speed = float(arr[IDX_EGO_SPEED])
        route_y = float(arr[IDX_ROUTE_Y])
        speed_limit = float(arr[IDX_ROUTE_SPEED_LIMIT])
        traffic_light = float(arr[IDX_ROUTE_TRAFFIC_LIGHT])
        route_dist_next = float(arr[IDX_ROUTE_DIST_NEXT])

        if self._has_immediate_threat(ahead):
            action = self._choose_allowed([EMERGENCY_BRAKE, DECEL, KEEP], mask)
            self.last_action, self.last_reason = action, "immediate_threat"
            return action

        if traffic_light >= self.RED_LIGHT and route_dist_next <= self.RED_LIGHT_DIST:
            action = self._choose_allowed([EMERGENCY_BRAKE, DECEL, KEEP], mask)
            self.last_action, self.last_reason = action, "red_light"
            return action
        if traffic_light >= self.YELLOW_LIGHT and route_dist_next <= self.YELLOW_LIGHT_DIST:
            if ego_speed > 0.15 or self._has_caution_threat(ahead):
                action = self._choose_allowed([DECEL, EMERGENCY_BRAKE, KEEP], mask)
                self.last_action, self.last_reason = action, "yellow_light"
                return action

        if self._has_caution_threat(ahead):
            preferred_side = LANE_LEFT if route_y >= 0.0 else LANE_RIGHT
            other_side = LANE_RIGHT if preferred_side == LANE_LEFT else LANE_LEFT
            for side in (preferred_side, other_side):
                if bool(mask[side]) and self._lane_zone_clear(candidates, side, self.CAUTION_X):
                    self.last_action, self.last_reason = int(side), "lane_change_avoid"
                    return int(side)
            action = self._choose_allowed([DECEL, EMERGENCY_BRAKE, KEEP], mask)
            self.last_action, self.last_reason = action, "caution_decel"
            return action

        nearest_ahead_x = ahead[0].rel_x if ahead else float("inf")
        target_limit = speed_limit if speed_limit > 0.05 else 0.5
        if ego_speed > target_limit + self.SPEED_MARGIN:
            action = self._choose_allowed([DECEL, KEEP, ACCEL], mask)
            self.last_action, self.last_reason = action, "above_speed_limit"
            return action
        if ego_speed < target_limit - self.SPEED_MARGIN:
            if nearest_ahead_x > self.CLEAR_AHEAD_X:
                action = self._choose_allowed([ACCEL, KEEP, DECEL], mask)
            elif nearest_ahead_x > self.CAUTION_X:
                action = self._choose_allowed([KEEP, ACCEL, DECEL], mask)
            else:
                action = self._choose_allowed([KEEP, DECEL, ACCEL], mask)
            self.last_action, self.last_reason = action, "speed_regulation"
            return action

        action = self._choose_allowed([KEEP, DECEL, ACCEL], mask)
        self.last_action, self.last_reason = action, "keep"
        return action

    def __call__(self, obs: Sequence[float] | np.ndarray, action_mask: Sequence[bool] | np.ndarray | None = None) -> int:
        return self.act(obs, action_mask=action_mask)


__all__ = [
    "CAND_START",
    "C_REL_HEADING",
    "C_REL_VX",
    "C_REL_VY",
    "C_REL_X",
    "C_REL_Y",
    "C_VALID",
    "Candidate",
    "IDX_EGO_DIST_GOAL",
    "IDX_EGO_HEADING_ERROR",
    "IDX_EGO_LANE_COUNT",
    "IDX_EGO_LANE_INDEX",
    "IDX_EGO_LATERAL_OFFSET",
    "IDX_EGO_LON_ACCEL",
    "IDX_EGO_PROGRESS",
    "IDX_EGO_SPEED",
    "IDX_ROUTE_AT_INTERSECTION",
    "IDX_ROUTE_DIST_NEXT",
    "IDX_ROUTE_SPEED_LIMIT",
    "IDX_ROUTE_TRAFFIC_LIGHT",
    "IDX_ROUTE_X",
    "IDX_ROUTE_Y",
    "MASK_START",
    "ObservationHeuristicController",
]
