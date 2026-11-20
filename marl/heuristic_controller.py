"""Deterministic observation-conditioned heuristic controller (Milestone 8).

This module provides a CARLA-independent controller that consumes the existing
56-D observation (and action mask) defined by ``envs.obs_builder`` and returns
one of the six canonical meta-actions.

Design goals:
- Deterministic for identical inputs (no global random state).
- Genuinely uses ego/route features and valid selected candidate features.
- Respects action masks and preserves emergency-brake availability.
- Safe handling of empty/missing candidate slots and defensive validation.

Observation contract (56 values, float32, normalized/clipped to [-1, 1]):
  - Ego [0:8]:
      0 speed / V_MAX
      1 longitudinal_accel / A_MAX
      2 heading_error / pi
      3 lateral_offset / lane_width
      4 lane_index / MAX_LANES
      5 lane_count / MAX_LANES
      6 distance_to_goal / D_MAX
      7 progress_fraction
  - Route [8:14]:
      8 next_waypoint_x (ego frame) / ROUTE_FRAME_MAX
      9 next_waypoint_y (ego frame) / ROUTE_FRAME_MAX
      10 speed_limit / V_MAX
      11 at_intersection
      12 traffic_light_state (0 none/green, 0.5 yellow, 1 red)
      13 distance_to_next / ROUTE_FRAME_MAX
  - Candidates [14:50], 6 slots x 6 features:
      [rel_x/CANDIDATE_RANGE, rel_y/CANDIDATE_RANGE,
       rel_vx/V_MAX, rel_vy/V_MAX, rel_heading/pi, valid]
  - Action mask [50:56]: [KEEP, ACCEL, DECEL, EMERGENCY_BRAKE, LANE_LEFT, LANE_RIGHT]

Indices/constants mirror ``envs.obs_builder`` to avoid guessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from envs.obs_builder import (
    A_MAX,
    CAND_FEATURES,
    CANDIDATE_RANGE,
    D_MAX,
    EGO,
    KEEP,
    ACCEL,
    DECEL,
    EMERGENCY_BRAKE,
    LANE_LEFT,
    LANE_RIGHT,
    LANE_WIDTH_MAX,
    M_CANDIDATES,
    MASK,
    MAX_LANES,
    OBS_DIM,
    ROUTE,
    ROUTE_FRAME_MAX,
    V_MAX,
)

# ---- exact observation indices (verified against envs/obs_builder.py) ----
IDX_EGO_SPEED = 0
IDX_EGO_LON_ACCEL = 1
IDX_EGO_HEADING_ERROR = 2
IDX_EGO_LATERAL_OFFSET = 3
IDX_EGO_LANE_INDEX = 4
IDX_EGO_LANE_COUNT = 5
IDX_EGO_DIST_GOAL = 6
IDX_EGO_PROGRESS = 7

IDX_ROUTE_X = EGO + 0
IDX_ROUTE_Y = EGO + 1
IDX_ROUTE_SPEED_LIMIT = EGO + 2
IDX_ROUTE_AT_INTERSECTION = EGO + 3
IDX_ROUTE_TRAFFIC_LIGHT = EGO + 4
IDX_ROUTE_DIST_NEXT = EGO + 5

CAND_START = EGO + ROUTE
MASK_START = EGO + ROUTE + M_CANDIDATES * CAND_FEATURES

C_REL_X = 0
C_REL_Y = 1
C_REL_VX = 2
C_REL_VY = 3
C_REL_HEADING = 4
C_VALID = 5


@dataclass(frozen=True)
class ParsedCandidate:
    """One candidate slot in normalized ego-frame observation units."""

    slot: int
    rel_x: float  # forward distance / CANDIDATE_RANGE (positive = ahead)
    rel_y: float  # left distance / CANDIDATE_RANGE (positive = left)
    rel_vx: float  # relative forward speed / V_MAX (negative = closing)
    rel_vy: float  # relative lateral speed / V_MAX
    rel_heading: float  # relative heading / pi
    valid: bool


class HeuristicController:
    """Deterministic, observation-conditioned driving controller.

    Candidate (selected teammate) features are used explicitly in threat
    detection, speed gating, and lane-clearance checks. Different valid
    candidate observations can therefore produce different actions for the same
    ego/route/mask state, enabling valid nearest_k vs random_k evaluation when
    both modes use identical controller parameters and rules.
    """

    # Thresholds in normalized observation units (documented conversions):
    # CANDIDATE_RANGE=50 m, V_MAX=30 m/s, ROUTE_FRAME_MAX=20 m.
    SAME_LANE_HALF_WIDTH = 0.08   # |rel_y| <= 4 m: ego corridor
    SIDE_LANE_INNER = 0.03        # 1.5 m: inner edge of adjacent lane zone
    SIDE_LANE_OUTER = 0.18        # 9 m: outer edge of adjacent lane zone

    AHEAD_MIN_X = -0.02           # -1 m tolerance
    EMERGENCY_X = 0.10            # 5 m
    BRAKE_X = 0.16                # 8 m
    CAUTION_X = 0.30              # 15 m
    CLEAR_AHEAD_X = 0.40          # 20 m

    CLOSING_FAST = -0.05          # -1.5 m/s
    CLOSING_MODERATE = -0.015     # -0.45 m/s

    RED_LIGHT = 0.75
    YELLOW_LIGHT = 0.25
    RED_LIGHT_DIST = 0.25         # 5 m in route-distance units
    YELLOW_LIGHT_DIST = 0.35      # 7 m
    SPEED_MARGIN = 0.03

    def __init__(self) -> None:
        self.last_action: int | None = None
        self.last_reason: str | None = None

    # ----------------------------- validation ----------------------------- #
    @staticmethod
    def validate_observation(obs: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return a validated float32 copy of a 56-D observation.

        Raises ``ValueError`` for wrong shape or non-finite values.
        """
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim != 1 or arr.shape != (OBS_DIM,):
            raise ValueError(
                f"observation must have shape ({OBS_DIM},), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("observation contains non-finite values")
        return arr.astype(np.float32, copy=True)

    @staticmethod
    def parse_action_mask(
        obs: np.ndarray,
        action_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Parse/validate mask from obs tail or explicit argument.

        Emergency brake is always preserved. If the resulting mask would allow
        no actions, KEEP is enabled as a safe fallback.
        """
        if action_mask is None:
            mask = np.asarray(obs[MASK_START:MASK_START + MASK], dtype=np.float32) > 0.5
        else:
            mask = np.asarray(action_mask, dtype=bool).reshape(-1)
            if mask.shape != (MASK,):
                raise ValueError(f"action_mask must have shape ({MASK},), got {mask.shape}")
        # Preserve emergency-brake availability (contract/obs builder invariant).
        mask = mask.copy()
        mask[EMERGENCY_BRAKE] = True
        if not np.any(mask):
            mask[KEEP] = True
        return mask.astype(bool, copy=False)

    @staticmethod
    def parse_candidates(obs: np.ndarray) -> list[ParsedCandidate]:
        """Parse candidate slots; invalid/padded/non-finite slots are marked invalid."""
        out: list[ParsedCandidate] = []
        block = obs[CAND_START:MASK_START]
        for slot in range(M_CANDIDATES):
            vals = block[slot * CAND_FEATURES:(slot + 1) * CAND_FEATURES]
            finite = all(math.isfinite(float(v)) for v in vals)
            valid = bool(finite and float(vals[C_VALID]) > 0.5)
            out.append(
                ParsedCandidate(
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

    # ------------------------- candidate reasoning ------------------------ #
    @staticmethod
    def _valid(candidates: Sequence[ParsedCandidate]) -> list[ParsedCandidate]:
        return [c for c in candidates if c.valid]

    def _ahead_same_lane(self, candidates: Sequence[ParsedCandidate]) -> list[ParsedCandidate]:
        """Valid candidates ahead in ego corridor, sorted nearest-first."""
        ahead = [
            c for c in self._valid(candidates)
            if c.rel_x >= self.AHEAD_MIN_X and abs(c.rel_y) <= self.SAME_LANE_HALF_WIDTH
        ]
        ahead.sort(key=lambda c: (c.rel_x, c.slot))
        return ahead

    def _lane_zone_clear(
        self,
        candidates: Sequence[ParsedCandidate],
        direction: int,
        x_limit: float,
    ) -> bool:
        """True if the target adjacent lane zone is clear of valid candidates.

        Candidate positions/velocities directly determine clearance: if a valid
        candidate occupies the adjacent lane zone within ``x_limit`` ahead, the
        lane is unsafe. Missing/unselected candidates are absent and therefore do
        not block the lane in this observation, which is the mechanism by which
        selection can influence decisions.
        """
        for c in self._valid(candidates):
            if c.rel_x < self.AHEAD_MIN_X or c.rel_x > x_limit:
                continue
            if direction == LANE_LEFT:
                in_zone = self.SIDE_LANE_INNER <= c.rel_y <= self.SIDE_LANE_OUTER
                straddle = (
                    0.0 < c.rel_y < self.SIDE_LANE_OUTER
                    and c.rel_x < self.CAUTION_X
                    and abs(c.rel_y) > self.SAME_LANE_HALF_WIDTH * 0.5
                )
            elif direction == LANE_RIGHT:
                in_zone = -self.SIDE_LANE_OUTER <= c.rel_y <= -self.SIDE_LANE_INNER
                straddle = (
                    -self.SIDE_LANE_OUTER < c.rel_y < 0.0
                    and c.rel_x < self.CAUTION_X
                    and abs(c.rel_y) > self.SAME_LANE_HALF_WIDTH * 0.5
                )
            else:
                return False
            if in_zone or straddle:
                return False
        return True

    def _has_immediate_threat(self, ahead: Sequence[ParsedCandidate]) -> bool:
        for c in ahead:
            if c.rel_x <= self.EMERGENCY_X:
                return True
            if c.rel_x <= self.BRAKE_X and c.rel_vx <= self.CLOSING_FAST:
                return True
        return False

    def _has_caution_threat(self, ahead: Sequence[ParsedCandidate]) -> bool:
        for c in ahead:
            if c.rel_x <= self.BRAKE_X:
                return True
            if c.rel_x <= self.CAUTION_X and c.rel_vx <= self.CLOSING_MODERATE:
                return True
            if c.rel_x <= self.CAUTION_X and c.rel_vx <= 0.0:
                return True
        return False

    # --------------------------- action selection ------------------------- #
    @staticmethod
    def _choose_allowed(preferred: Sequence[int], mask: np.ndarray) -> int:
        for action in preferred:
            a = int(action)
            if 0 <= a < MASK and bool(mask[a]):
                return a
        # Deterministic fallback order preserving emergency brake availability.
        for action in (EMERGENCY_BRAKE, DECEL, KEEP, ACCEL, LANE_LEFT, LANE_RIGHT):
            if bool(mask[action]):
                return int(action)
        return int(EMERGENCY_BRAKE)

    def act(
        self,
        obs: Sequence[float] | np.ndarray,
        action_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> int:
        """Return a deterministic action in {0..5} for the given observation.

        Args:
            obs: 56-D observation vector.
            action_mask: Optional explicit 6-bool mask; if omitted, mask is read
                from ``obs[50:56]``.
        """
        arr = self.validate_observation(obs)
        mask = self.parse_action_mask(arr, action_mask=action_mask)
        candidates = self.parse_candidates(arr)
        ahead = self._ahead_same_lane(candidates)

        ego_speed = float(arr[IDX_EGO_SPEED])
        route_y = float(arr[IDX_ROUTE_Y])
        speed_limit = float(arr[IDX_ROUTE_SPEED_LIMIT])
        traffic_light = float(arr[IDX_ROUTE_TRAFFIC_LIGHT])
        route_dist_next = float(arr[IDX_ROUTE_DIST_NEXT])

        # 1) Immediate collision/closing threat ahead (candidate-driven).
        if self._has_immediate_threat(ahead):
            action = self._choose_allowed([EMERGENCY_BRAKE, DECEL, KEEP], mask)
            self.last_action, self.last_reason = action, "immediate_threat"
            return action

        # 2) Traffic-light / route hazard, or caution threat ahead.
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
            # 3) Attempt lane change only if target side is candidate-clear.
            preferred_side = LANE_LEFT if route_y >= 0.0 else LANE_RIGHT
            other_side = LANE_RIGHT if preferred_side == LANE_LEFT else LANE_LEFT
            for side in (preferred_side, other_side):
                if bool(mask[side]) and self._lane_zone_clear(candidates, side, self.CAUTION_X):
                    self.last_action, self.last_reason = int(side), "lane_change_avoid"
                    return int(side)
            action = self._choose_allowed([DECEL, EMERGENCY_BRAKE, KEEP], mask)
            self.last_action, self.last_reason = action, "caution_decel"
            return action

        # 4) Speed regulation, gated by candidate proximity ahead.
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

    def __call__(
        self,
        obs: Sequence[float] | np.ndarray,
        action_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> int:
        return self.act(obs, action_mask=action_mask)


# Backward/explicit naming convenience
ObservationHeuristicController = HeuristicController

__all__ = [
    "ACCEL",
    "A_MAX",
    "BRAKE_X",
    "CANDIDATE_RANGE",
    "CAND_FEATURES",
    "CAND_START",
    "CAUTION_X",
    "CLEAR_AHEAD_X",
    "CLOSING_FAST",
    "CLOSING_MODERATE",
    "C_REL_HEADING",
    "C_REL_VX",
    "C_REL_VY",
    "C_REL_X",
    "C_REL_Y",
    "C_VALID",
    "DECEL",
    "D_MAX",
    "EMERGENCY_BRAKE",
    "EMERGENCY_X",
    "EGO",
    "HeuristicController",
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
    "KEEP",
    "LANE_LEFT",
    "LANE_RIGHT",
    "LANE_WIDTH_MAX",
    "MASK",
    "MASK_START",
    "MAX_LANES",
    "M_CANDIDATES",
    "OBS_DIM",
    "ObservationHeuristicController",
    "ParsedCandidate",
    "ROUTE",
    "ROUTE_FRAME_MAX",
    "V_MAX",
]
