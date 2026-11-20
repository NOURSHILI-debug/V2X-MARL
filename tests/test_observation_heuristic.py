"""Tests for the deterministic observation-conditioned CARLA heuristic controller."""

from __future__ import annotations

import math

import numpy as np
import pytest

from envs.obs_builder import (
    CAND,
    CAND_FEATURES,
    CANDIDATE_RANGE,
    EGO,
    KEEP,
    ACCEL,
    DECEL,
    EMERGENCY_BRAKE,
    LANE_LEFT,
    LANE_RIGHT,
    MASK,
    M_CANDIDATES,
    OBS_DIM,
    ROUTE,
    V_MAX,
    build_observation,
    make_action_mask,
)
from policies.observation_heuristic import (
    CAND_START,
    C_REL_VX,
    C_REL_X,
    C_REL_Y,
    C_VALID,
    IDX_EGO_SPEED,
    IDX_ROUTE_DIST_NEXT,
    IDX_ROUTE_SPEED_LIMIT,
    IDX_ROUTE_TRAFFIC_LIGHT,
    IDX_ROUTE_Y,
    MASK_START,
    ObservationHeuristicController,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ego(**overrides):
    ego = {
        "x": 0.0,
        "y": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "yaw": 0.0,
        "speed": 0.0,
        "longitudinal_accel": 0.0,
        "heading_error": 0.0,
        "lateral_offset": 0.0,
        "lane_width": 3.5,
        "lane_index": 1.0,
        "lane_count": 3.0,
        "distance_to_goal": 100.0,
        "progress_fraction": 0.2,
    }
    ego.update(overrides)
    return ego


def _route(**overrides):
    route = {
        "next_waypoint_x": 20.0,
        "next_waypoint_y": 0.0,
        "speed_limit": 15.0,  # m/s
        "at_intersection": False,
        "traffic_light_state": "none",
        "distance_to_next": 15.0,
    }
    route.update(overrides)
    return route


def _candidate(x, y, vx=0.0, vy=0.0, yaw=0.0, actor_id=100, valid=True):
    return {
        "actor_id": actor_id,
        "x": float(x),
        "y": float(y),
        "vx": float(vx),
        "vy": float(vy),
        "yaw": float(yaw),
        "valid": bool(valid),
    }


def _obs(ego=None, route=None, candidates=None, mask=None):
    return build_observation(
        _ego() if ego is None else ego,
        _route() if route is None else route,
        [] if candidates is None else list(candidates),
        np.ones(MASK, dtype=bool) if mask is None else np.asarray(mask, dtype=bool),
    )


def _set_candidate(obs, slot, rel_x, rel_y, rel_vx, rel_vy, rel_heading, valid):
    base = CAND_START + slot * CAND_FEATURES
    obs[base + 0] = rel_x
    obs[base + 1] = rel_y
    obs[base + 2] = rel_vx
    obs[base + 3] = rel_vy
    obs[base + 4] = rel_heading
    obs[base + 5] = valid
    return obs


# --------------------------------------------------------------------------- #
# Shape / parsing / validation
# --------------------------------------------------------------------------- #
def test_rejects_wrong_observation_shape():
    controller = ObservationHeuristicController()
    with pytest.raises(ValueError):
        controller.act(np.zeros(55, dtype=np.float32))
    with pytest.raises(ValueError):
        controller.act(np.zeros((1, 56), dtype=np.float32))


def test_parses_ego_and_route_features():
    controller = ObservationHeuristicController()
    obs = _obs(
        ego=_ego(speed=10.0, distance_to_goal=50.0),
        route=_route(speed_limit=20.0, traffic_light_state="red", distance_to_next=5.0),
    )
    assert obs.shape == (OBS_DIM,)
    assert obs[IDX_EGO_SPEED] == pytest.approx(10.0 / V_MAX)
    assert obs[IDX_ROUTE_SPEED_LIMIT] == pytest.approx(20.0 / V_MAX)
    assert obs[IDX_ROUTE_TRAFFIC_LIGHT] == pytest.approx(1.0)
    # Controller should be able to act on parsed values without error.
    action = controller.act(obs)
    assert 0 <= action < MASK


def test_deterministic_behavior():
    controller = ObservationHeuristicController()
    obs = _obs(candidates=[_candidate(10.0, 0.0, actor_id=1)])
    actions = [controller.act(obs.copy()) for _ in range(20)]
    assert len(set(actions)) == 1
    # Fresh controller instance gives same result.
    assert ObservationHeuristicController().act(obs.copy()) == actions[0]


def test_non_finite_inputs_handled_defensively():
    controller = ObservationHeuristicController()
    obs = _obs()
    obs[IDX_EGO_SPEED] = np.nan
    obs[IDX_ROUTE_Y] = np.inf
    obs[CAND_START + 0] = -np.inf
    action = controller.act(obs)
    assert isinstance(action, int)
    assert 0 <= action < MASK


# --------------------------------------------------------------------------- #
# Candidate validity / empty / padded slots
# --------------------------------------------------------------------------- #
def test_empty_candidate_pool():
    controller = ObservationHeuristicController()
    obs = _obs(candidates=[])
    # All candidate slots padded/invalid.
    cand_block = obs[CAND_START:MASK_START].reshape(M_CANDIDATES, CAND_FEATURES)
    assert np.all(cand_block[:, C_VALID] == 0.0)
    action = controller.act(obs)
    assert 0 <= action < MASK


def test_invalid_and_padded_candidate_slots_ignored():
    controller = ObservationHeuristicController()
    # Build a clean observation, then inject a slot marked invalid but with
    # threatening coordinates: it must be ignored.
    obs_safe = _obs(ego=_ego(speed=5.0), candidates=[])
    obs_safe[IDX_ROUTE_SPEED_LIMIT] = 0.5  # 15 m/s target
    obs_safe[MASK_START:MASK_START + MASK] = 1.0
    action_safe = controller.act(obs_safe.copy())

    obs_padded_threat = obs_safe.copy()
    _set_candidate(obs_padded_threat, 0, rel_x=0.05, rel_y=0.0, rel_vx=-0.1,
                   rel_vy=0.0, rel_heading=0.0, valid=0.0)
    action_padded = controller.act(obs_padded_threat.copy())

    # A padded/invalid slot must not create an emergency brake threat.
    assert action_padded == action_safe
    parsed = controller.parse_candidates(obs_padded_threat)
    assert parsed[0].valid is False


def test_candidate_validity_mask_handling():
    controller = ObservationHeuristicController()
    obs = _obs(candidates=[
        _candidate(10.0, 0.0, actor_id=1, valid=True),
        _candidate(12.0, 1.0, actor_id=2, valid=False),
    ])
    parsed = controller.parse_candidates(obs)
    # build_observation filters by valid flag and range; invalid candidate not included.
    valid_slots = [c for c in parsed if c.valid]
    assert len(valid_slots) == 1


# --------------------------------------------------------------------------- #
# Action mask compliance
# --------------------------------------------------------------------------- #
def test_action_mask_compliance():
    controller = ObservationHeuristicController()
    obs = _obs(ego=_ego(speed=3.0), route=_route(speed_limit=15.0))
    # Disallow everything except DECEL and EMERGENCY_BRAKE.
    mask = np.array([False, False, True, True, False, False], dtype=bool)
    obs[MASK_START:MASK_START + MASK] = mask.astype(np.float32)
    action = controller.act(obs)
    assert mask[action] or action == EMERGENCY_BRAKE


def test_lane_change_only_when_mask_allows():
    controller = ObservationHeuristicController()
    # Create a caution situation: blocked ahead, left lane clear by candidates,
    # but left lane change disallowed by mask.
    obs = _obs(
        ego=_ego(speed=10.0),
        route=_route(speed_limit=15.0, next_waypoint_x=20.0, next_waypoint_y=5.0),
        candidates=[_candidate(8.0, 0.0, vx=-1.0, vy=0.0, actor_id=1)],  # ahead, closing
        mask=make_action_mask(left_lane_exists=False, right_lane_exists=True),
    )
    action = controller.act(obs)
    assert action != LANE_LEFT


# --------------------------------------------------------------------------- #
# Candidate information genuinely influences decisions
# --------------------------------------------------------------------------- #
def test_candidate_ahead_changes_decision_vs_no_candidates():
    controller = ObservationHeuristicController()
    mask = make_action_mask(left_lane_exists=False, right_lane_exists=False)

    # No candidates, below speed limit -> should accelerate (or keep), not brake.
    obs_clear = _obs(
        ego=_ego(speed=5.0),
        route=_route(speed_limit=15.0),
        candidates=[],
        mask=mask,
    )
    action_clear = controller.act(obs_clear)

    # Very close candidate directly ahead in same lane -> must brake/decelerate.
    obs_blocked = _obs(
        ego=_ego(speed=5.0),
        route=_route(speed_limit=15.0),
        candidates=[_candidate(4.0, 0.0, vx=-2.0, vy=0.0, actor_id=1)],
        mask=mask,
    )
    action_blocked = controller.act(obs_blocked)

    assert action_clear in (ACCEL, KEEP)
    assert action_blocked in (EMERGENCY_BRAKE, DECEL)
    assert action_clear != action_blocked


def test_side_candidate_blocks_lane_change_but_clear_side_allows_it():
    controller = ObservationHeuristicController()
    mask = make_action_mask(left_lane_exists=True, right_lane_exists=True)
    ego = _ego(speed=8.0)
    route = _route(speed_limit=15.0, next_waypoint_x=20.0, next_waypoint_y=0.0)
    ahead_blocker = _candidate(7.0, 0.0, vx=-1.0, vy=0.0, actor_id=1)

    # Left lane occupied by candidate, right lane clear -> should choose right.
    obs_left_blocked = _obs(
        ego=ego,
        route=route,
        candidates=[ahead_blocker, _candidate(7.0, 4.0, vx=0.0, vy=0.0, actor_id=2)],
        mask=mask,
    )
    action_left_blocked = ObservationHeuristicController().act(obs_left_blocked)

    # Right lane occupied, left lane clear -> should choose left.
    obs_right_blocked = _obs(
        ego=ego,
        route=route,
        candidates=[ahead_blocker, _candidate(7.0, -4.0, vx=0.0, vy=0.0, actor_id=3)],
        mask=mask,
    )
    action_right_blocked = ObservationHeuristicController().act(obs_right_blocked)

    assert action_left_blocked == LANE_RIGHT
    assert action_right_blocked == LANE_LEFT
    assert action_left_blocked != action_right_blocked


def test_different_selected_candidates_change_decision():
    """Nearest-k vs random-k can yield different selected sets: controller must react."""
    controller = ObservationHeuristicController()
    mask = make_action_mask(left_lane_exists=True, right_lane_exists=True)
    ego = _ego(speed=8.0)
    route = _route(speed_limit=15.0, next_waypoint_x=20.0, next_waypoint_y=0.0)
    ahead_blocker = _candidate(7.0, 0.0, vx=-1.0, vy=0.0, actor_id=1)
    left_lane_vehicle = _candidate(7.0, 4.0, vx=0.0, vy=0.0, actor_id=2)
    right_lane_vehicle = _candidate(7.0, -4.0, vx=0.0, vy=0.0, actor_id=3)

    # Selection A includes left-lane vehicle (blocks left), right is clear.
    obs_a = _obs(ego=ego, route=route, candidates=[ahead_blocker, left_lane_vehicle], mask=mask)
    # Selection B includes right-lane vehicle (blocks right), left is clear.
    obs_b = _obs(ego=ego, route=route, candidates=[ahead_blocker, right_lane_vehicle], mask=mask)

    action_a = ObservationHeuristicController().act(obs_a)
    action_b = ObservationHeuristicController().act(obs_b)
    assert action_a != action_b


# --------------------------------------------------------------------------- #
# Coverage across action choices (where applicable)
# --------------------------------------------------------------------------- #
def test_action_keep_when_at_speed_limit_and_clear():
    controller = ObservationHeuristicController()
    obs = _obs(
        ego=_ego(speed=15.0),
        route=_route(speed_limit=15.0),
        candidates=[],
        mask=make_action_mask(True, True),
    )
    assert controller.act(obs) == KEEP


def test_action_accelerate_when_below_speed_limit_and_clear():
    controller = ObservationHeuristicController()
    obs = _obs(
        ego=_ego(speed=5.0),
        route=_route(speed_limit=15.0),
        candidates=[],
        mask=make_action_mask(True, True),
    )
    assert controller.act(obs) == ACCEL


def test_action_decelerate_when_above_speed_limit():
    controller = ObservationHeuristicController()
    obs = _obs(
        ego=_ego(speed=20.0),
        route=_route(speed_limit=10.0),
        candidates=[],
        mask=make_action_mask(True, True),
    )
    assert controller.act(obs) == DECEL


def test_action_emergency_brake_for_immediate_threat():
    controller = ObservationHeuristicController()
    obs = _obs(
        ego=_ego(speed=10.0),
        route=_route(speed_limit=15.0),
        candidates=[_candidate(3.0, 0.0, vx=-2.0, vy=0.0, actor_id=1)],
        mask=make_action_mask(False, False),
    )
    assert controller.act(obs) == EMERGENCY_BRAKE


def test_action_lane_left_when_blocked_ahead_and_left_clear():
    controller = ObservationHeuristicController()
    obs = _obs(
        ego=_ego(speed=8.0),
        route=_route(speed_limit=15.0, next_waypoint_x=20.0, next_waypoint_y=2.0),
        candidates=[_candidate(7.0, 0.0, vx=-1.0, vy=0.0, actor_id=1)],
        mask=make_action_mask(left_lane_exists=True, right_lane_exists=True),
    )
    assert controller.act(obs) == LANE_LEFT


def test_action_lane_right_when_blocked_ahead_and_right_preferred_or_clear():
    controller = ObservationHeuristicController()
    obs = _obs(
        ego=_ego(speed=8.0),
        route=_route(speed_limit=15.0, next_waypoint_x=20.0, next_waypoint_y=-2.0),
        candidates=[_candidate(7.0, 0.0, vx=-1.0, vy=0.0, actor_id=1)],
        mask=make_action_mask(left_lane_exists=True, right_lane_exists=True),
    )
    assert controller.act(obs) == LANE_RIGHT


def test_red_traffic_light_causes_braking_behavior():
    controller = ObservationHeuristicController()
    obs = _obs(
        ego=_ego(speed=8.0),
        route=_route(speed_limit=15.0, traffic_light_state="red", distance_to_next=3.0),
        candidates=[],
        mask=make_action_mask(True, True),
    )
    assert controller.act(obs) in (EMERGENCY_BRAKE, DECEL)


def test_returns_valid_action_index():
    controller = ObservationHeuristicController()
    obs = _obs()
    action = controller.act(obs)
    assert isinstance(action, (int, np.integer))
    assert 0 <= int(action) < 6
