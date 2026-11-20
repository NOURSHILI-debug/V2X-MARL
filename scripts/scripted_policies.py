"""Shared scripted policies for scenario gates.

The competent controller reads only the canonical 56-D observation and the
six-value action mask. No environment internals are available to it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from envs.obs_builder import (
    ACCEL,
    CAND_FEATURES,
    CANDIDATE_RANGE,
    DECEL,
    EGO,
    EMERGENCY_BRAKE,
    KEEP,
    LANE_LEFT,
    LANE_RIGHT,
    M_CANDIDATES,
    ROUTE,
    V_MAX,
)

POLICIES = ("keep_speed", "random", "accelerate", "competent")

TTC_WARNING_S = 3.0
TTC_BRAKE_S = 1.5
SAFE_LANE_GAP_M = 12.0
LANE_WIDTH_M = 3.5
VEHICLE_LENGTH_M = 5.0
CRUISE_SPEED_NORM = 0.72
CLEAR_AHEAD_M = 35.0
MAX_TTC_S = 10.0

_CANDIDATE_BLOCK_START = EGO + ROUTE


def competent_action(observation: Any, mask: Any) -> int:
    """Return a safe action computed only from one observation and its mask."""
    speed_norm = float(observation[0])
    ego_ttc = float("inf")
    front_clearance = float("inf")
    target_clearance = {1: float("inf"), -1: float("inf")}

    for index in range(M_CANDIDATES):
        base = _CANDIDATE_BLOCK_START + index * CAND_FEATURES
        if float(observation[base + 5]) <= 0.0:
            continue
        rel_x = float(observation[base]) * CANDIDATE_RANGE
        rel_y = float(observation[base + 1]) * CANDIDATE_RANGE
        rel_vx = float(observation[base + 2]) * V_MAX
        gap = rel_x - VEHICLE_LENGTH_M
        closing = -rel_vx

        if abs(rel_y) <= LANE_WIDTH_M / 2.0 and gap > 0.0 and closing > 0.0:
            ego_ttc = min(ego_ttc, min(gap / closing, MAX_TTC_S))
        if gap > 0.0:
            front_clearance = min(front_clearance, gap)
        for direction in (1, -1):
            if abs(rel_y - direction * LANE_WIDTH_M) <= LANE_WIDTH_M / 2.0:
                clearance = (
                    rel_x - VEHICLE_LENGTH_M
                    if rel_x >= 0.0
                    else -rel_x - VEHICLE_LENGTH_M
                )
                target_clearance[direction] = min(target_clearance[direction], clearance)

    if ego_ttc < TTC_WARNING_S:
        options = []
        if mask[LANE_LEFT]:
            options.append((target_clearance[1], LANE_LEFT))
        if mask[LANE_RIGHT]:
            options.append((target_clearance[-1], LANE_RIGHT))
        safe_options = [option for option in options if option[0] >= SAFE_LANE_GAP_M]
        if safe_options:
            return int(max(safe_options)[1])
        if ego_ttc < TTC_BRAKE_S:
            return EMERGENCY_BRAKE
        return DECEL

    if (
        speed_norm < CRUISE_SPEED_NORM
        and front_clearance > CLEAR_AHEAD_M
        and mask[ACCEL]
    ):
        return ACCEL
    return KEEP


def choose_action(
    policy: str,
    observation: Any,
    mask: Any,
    rng: np.random.Generator,
) -> int:
    """Select one action for a policy, using observation only for competent."""
    if policy == "keep_speed":
        return KEEP
    if policy == "accelerate":
        return ACCEL
    if policy == "random":
        valid = np.flatnonzero(np.asarray(mask, dtype=bool))
        return int(rng.choice(valid))
    if policy == "competent":
        return competent_action(observation, mask)
    raise ValueError(f"unknown policy '{policy}'.")


def choose_actions(
    policy: str,
    observations: Mapping[str, Any],
    masks: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, int]:
    """Select one action per agent."""
    return {
        agent_id: choose_action(policy, observations[agent_id], masks[agent_id], rng)
        for agent_id in observations
    }


__all__ = [
    "POLICIES",
    "TTC_WARNING_S",
    "choose_action",
    "choose_actions",
    "competent_action",
]
