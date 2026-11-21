"""Deterministic observation-conditioned heuristic controller for CARLA.

Consumes the existing 56-D observation produced by ``envs.obs_builder`` and
returns one of the six canonical meta-actions. Candidate (teammate) features
genuinely influence decisions: threats and lane-clearance checks are computed
from the valid selected candidate slots, so different selected candidates can
produce different actions under the same ego/route/mask state.

No CARLA dependency, no global random state, pure NumPy/Python.
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

# Observation indices (verified against envs/obs_builder.py build_observation)
# Ego block (8): [speed/V_MAX, lon_accel/A_MAX, heading_error/pi,
#   lateral_offset/lane_width, lane_index/MAX_LANES, lane_count/MAX_LANES,
#   distance_to_goal/D_MAX, progress_fraction]
IDX_EGO_SPEED = 0
IDX_EGO_LON_ACCEL = 1
IDX_EGO_HEADING_ERROR = 2
IDX_EGO_LATERAL_OFFSET = 3
IDX_EGO_LANE_INDEX = 4
IDX_EGO_LANE_COUNT = 5
IDX_EGO_DIST_GOAL = 6
IDX_EGO_PROGRESS = 7

# Route block (6): [route_x/ROUTE_FRAME_MAX, route_y/ROUTE_FRAME_MAX,
#   speed_limit/V_MAX, at_intersection, traffic_light(0/0.5/1),
#   distance_to_next/ROUTE_FRAME_MAX]
IDX_ROUTE_X = EGO + 0
IDX_ROUTE_Y = EGO + 1
IDX_ROUTE_SPEED_LIMIT = EGO + 2
IDX_ROUTE_AT_INTERSECTION = EGO + 3
IDX_ROUTE_TRAFFIC_LIGHT = EGO + 4
IDX_ROUTE_DIST_NEXT = EGO + 5

CAND_START = EGO + ROUTE
MASK_START = EGO + ROUTE + M_CANDIDATES * CAND_FEATURES

# Candidate feature offsets within each 6-value slot:
# [rel_x/CANDIDATE_RANGE, rel_y/CANDIDATE_RANGE, rel_vx/V_MAX, rel_vy/V_MAX,
#  rel_heading/pi, valid]
C_REL_X = 0
C_REL_Y = 1
C_REL_VX = 2
C_REL_VY = 3
C_REL_HEADING = 4
C_VALID = 5


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class Candidate:
    """Parsed view of one candidate slot in ego-frame normalized units."""

    slot: int
    rel_x: float
    rel_y: float
    rel_vx: float
    rel_vy: float
    rel_heading: float
    valid: bool


class ObservationHeuristicController:
    """Deterministic heuristic that uses ego, route, mask and candidate features.

    All spatial/velocity quantities are in the observation's normalized units
    (rel_x, rel_y in multiples of ``CANDIDATE_RANGE`` = 50 m; rel_vx in
    multiples of ``V_MAX`` = 30 m/s). Thresholds are expressed in those units
    and documented below.

    Decision logic (first applicable rule wins, subject to action mask):
      1. **Emergency brake**: a valid candidate is directly ahead in the ego
         corridor and critically close, or close and closing.
      2. **Brake/decelerate for hazard**: red traffic light at short route
         distance, or a valid candidate ahead at caution distance (especially
         when closing).
      3. **Lane change to avoid ahead threat**: if an ahead candidate blocks
         progress and a lateral lane is permitted by the mask and clear of
         valid candidates in that lane zone, change toward the route's lateral
         direction when safe, else toward any safe side.
      4. **Speed regulation**: decelerate if above speed limit; accelerate if
         below speed limit and no relevant ahead threat; otherwise keep.
    """

    # Lateral zones (|rel_y|, normalized by 50 m)
    SAME_LANE_HALF_WIDTH = 0.08   # ~4 m corridor treated as ego path
    SIDE_LANE_INNER = 0.03        # ~1.5 m inner edge for adjacent lane zone
    SIDE_LANE_OUTER = 0.18        # ~9 m outer edge for adjacent lane zone

    # Longitudinal zones ahead (rel_x, normalized by 50 m; positive = ahead)
    AHEAD_MIN_X = -0.02           # small tolerance behind ego origin
    EMERGENCY_X = 0.10            # ~5 m: immediate danger
    BRAKE_X = 0.16                # ~8 m: hard braking / strong caution
    CAUTION_X = 0.30              # ~15 m: caution / consider lane change
    CLEAR_AHEAD_X = 0.40          # ~20 m: relevant for speed keeping

    # Relative forward speed (rel_vx, normalized by 30 m/s; negative = closing)
    CLOSING_FAST = -0.05          # closing by ~1.5 m/s
    CLOSING_MODERATE = -0.015     # closing by ~0.45 m/s

    # Route/traffic thresholds
    RED_LIGHT = 0.75
    YELLOW_LIGHT = 0.25
    RED_LIGHT_DIST = 0.25         # route distance_to_next / 20 m
    YELLOW_LIGHT_DIST = 0.35
    SPEED_MARGIN = 0.03           # tolerance around speed limit

    def __init__(self) -> None:
        self.last_action: int | None = None
        self.last_reason: str | None = None

    # ---- parsing ---------------------------------------------------------
    @staticmethod
    def validate_observation(obs: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return a finite float32 copy of a valid 56-D observation."""
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim != 1 or arr.shape != (OBS_DIM,):
            raise ValueError(f"observation must have shape ({OBS_DIM},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            # Defensive: sanitize rather than crashing the caller.
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        return arr.astype(np.float32, copy=True)

    @staticmethod
    def parse_action_mask(obs: np.ndarray) -> np.ndarray:
        mask = obs[MASK_START:MASK_START + MASK] > 0.5
        input_any = bool(mask.any())
        # Emergency brake must always be available per contract/builder.
        mask[EMERGENCY_BRAKE] = True
        if not input_any:
            # Degenerate/empty mask input: allow KEEP as a safe fallback.
            mask[KEEP] = True
        return mask

    @staticmethod
    def parse_candidates(obs: np.ndarray) -> list[Candidate]:
        candidates: list[Candidate] = []
        for slot in range(M_CANDIDATES):
            base = CAND_START + slot * CAND_FEATURES
            vals = obs[base:base + CAND_FEATURES]
            valid = bool(vals[C_VALID] > 0.5) and all(_is_finite_number(v) for v in vals)
            candidates.append(
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
        return candidates

    # ---- candidate reasoning (explicit use of candidate features) --------
    def _valid_candidates(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        return [c for c in candidates if c.valid]

    def _ahead_same_lane(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        out = []
        for c in self._valid_candidates(candidates):
            if c.rel_x >= self.AHEAD_MIN_X and abs(c.rel_y) <= self.SAME_LANE_HALF_WIDTH:
                out.append(c)
        out.sort(key=lambda c: c.rel_x)
        return out

    def _lane_zone_clear(
        self,
        candidates: Sequence[Candidate],
        direction: int,
        x_limit: float,
    ) -> bool:
        """Return True if no valid candidate occupies the target lateral zone ahead.

        ``direction`` is ``LANE_LEFT`` (+y) or ``LANE_RIGHT`` (-y). Candidates
        influence this check directly: any valid candidate inside the adjacent
        lane zone within ``x_limit`` makes the lane unsafe.
        """
        for c in self._valid_candidates(candidates):
            if c.rel_x < self.AHEAD_MIN_X or c.rel_x > x_limit:
                continue
            if direction == LANE_LEFT:
                in_zone = self.SIDE_LANE_INNER <= c.rel_y <= self.SIDE_LANE_OUTER
                # Also treat a vehicle straddling the boundary as unsafe.
                straddle = 0.0 < c.rel_y < self.SIDE_LANE_OUTER and abs(c.rel_y) > self.SAME_LANE_HALF_WIDTH * 0.5 and c.rel_x < self.CAUTION_X
            else:
                in_zone = -self.SIDE_LANE_OUTER <= c.rel_y <= -self.SIDE_LANE_INNER
                straddle = -self.SIDE_LANE_OUTER < c.rel_y < 0.0 and abs(c.rel_y) > self.SAME_LANE_HALF_WIDTH * 0.5 and c.rel_x < self.CAUTION_X
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

    # ---- action selection ------------------------------------------------
    @staticmethod
    def _choose_allowed(preferred: Sequence[int], mask: np.ndarray) -> int:
        for action in preferred:
            if 0 <= action < MASK and mask[action]:
                return int(action)
        for action in (EMERGENCY_BRAKE, DECEL, KEEP, ACCEL, LANE_LEFT, LANE_RIGHT):
            if mask[action]:
                return int(action)
        return int(KEEP)

    def act(self, obs: Sequence[float] | np.ndarray) -> int:
        """Return a deterministic action in {0..5} for the given observation."""
        arr = self.validate_observation(obs)
        mask = self.parse_action_mask(arr)
        candidates = self.parse_candidates(arr)
        ahead = self._ahead_same_lane(candidates)

        ego_speed = float(arr[IDX_EGO_SPEED])
        route_y = float(arr[IDX_ROUTE_Y])
        speed_limit = float(arr[IDX_ROUTE_SPEED_LIMIT])
        traffic_light = float(arr[IDX_ROUTE_TRAFFIC_LIGHT])
        route_dist_next = float(arr[IDX_ROUTE_DIST_NEXT])

        # 1) Immediate collision/threat ahead -> emergency brake.
        if self._has_immediate_threat(ahead):
            action = self._choose_allowed([EMERGENCY_BRAKE, DECEL, KEEP], mask)
            self.last_action, self.last_reason = action, "immediate_threat"
            return action

        # 2) Traffic light hazard (route-driven) or caution threat ahead.
        if traffic_light >= self.RED_LIGHT and route_dist_next <= self.RED_LIGHT_DIST:
            action = self._choose_allowed([EMERGENCY_BRAKE, DECEL, KEEP], mask)
            self.last_action, self.last_reason = action, "red_light"
            return action
        if traffic_light >= self.YELLOW_LIGHT and route_dist_next <= self.YELLOW_LIGHT_DIST:
            if ego_speed > 0.15 or self._has_caution_threat(ahead):
                action = self._choose_allowed([DECEL, EMERGENCY_BRAKE, KEEP], mask)
                self.last_action, self.last_reason = action, "yellow_light"
                return action

        caution = self._has_caution_threat(ahead)
        if caution:
            # 3) Try to change lane if blocked ahead; candidate clearance is decisive.
            preferred_side = LANE_LEFT if route_y >= 0.0 else LANE_RIGHT
            other_side = LANE_RIGHT if preferred_side == LANE_LEFT else LANE_LEFT
            for side in (preferred_side, other_side):
                if mask[side] and self._lane_zone_clear(candidates, side, self.CAUTION_X):
                    self.last_action, self.last_reason = int(side), "lane_change_avoid"
                    return int(side)
            action = self._choose_allowed([DECEL, EMERGENCY_BRAKE, KEEP], mask)
            self.last_action, self.last_reason = action, "caution_decel"
            return action

        # 4) Speed regulation with awareness of candidates ahead.
        nearest_ahead_x = ahead[0].rel_x if ahead else float("inf")
        target_limit = speed_limit if speed_limit > 0.05 else 0.5
        if ego_speed > target_limit + self.SPEED_MARGIN:
            action = self._choose_allowed([DECEL, KEEP, ACCEL], mask)
            self.last_action, self.last_reason = action, "above_speed_limit"
            return action

        if ego_speed < target_limit - self.SPEED_MARGIN:
            # Only accelerate if no relevant candidate is close ahead.
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

    def __call__(self, obs: Sequence[float] | np.ndarray) -> int:
        return self.act(obs)


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
