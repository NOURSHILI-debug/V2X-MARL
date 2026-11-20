"""Tests for Milestone 8 deterministic observation-conditioned heuristic controller."""

from __future__ import annotations

import numpy as np
import pytest

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
    MASK,
    M_CANDIDATES,
    OBS_DIM,
    ROUTE,
    V_MAX,
    build_observation,
    make_action_mask,
)
from marl.heuristic_controller import (
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
    HeuristicController,
)


# ----------------------------- helpers ----------------------------------- #


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
        "speed_limit": 15.0,
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


# 1. Correct observation shape handling
def test_correct_observation_shape_handling():
    controller = HeuristicController()
    obs = _obs()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    action = controller.act(obs)
    assert 0 <= int(action) < MASK


# 2. Invalid observation rejection
def test_invalid_observation_rejection():
    controller = HeuristicController()
    with pytest.raises(ValueError):
        controller.act(np.zeros(55, dtype=np.float32))
    with pytest.raises(ValueError):
        controller.act(np.zeros((1, 56), dtype=np.float32))
    bad = _obs().copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        controller.act(bad)
    bad2 = _obs().copy()
    bad2[10] = np.inf
    with pytest.raises(ValueError):
        controller.act(bad2)


# 3. Deterministic actions for identical observations
def test_deterministic_actions_for_identical_observations():
    obs = _obs(candidates=[_candidate(10.0, 0.0, actor_id=1)])
    c1 = HeuristicController()
    c2 = HeuristicController()
    a1 = [c1.act(obs.copy()) for _ in range(10)]
    a2 = [c2.act(obs.copy()) for _ in range(10)]
    assert len(set(a1)) == 1
    assert a1 == a2


# 4. Action-mask compliance
def test_action_mask_compliance():
    controller = HeuristicController()
    obs = _obs(ego=_ego(speed=3.0), route=_route(speed_limit=15.0))
    mask = np.array([False, False, True, True, False, False], dtype=bool)  # DECEL, BRAKE
    obs[MASK_START:MASK_START + MASK] = mask.astype(np.float32)
    action = controller.act(obs)
    assert mask[action] or action == EMERGENCY_BRAKE

    # Explicit mask argument should also be respected (intersected with safety).
    obs2 = _obs(ego=_ego(speed=3.0), route=_route(speed_limit=15.0))
    action2 = controller.act(obs2, action_mask=np.array([True, False, False, True, False, False]))
    assert action2 in (KEEP, EMERGENCY_BRAKE)


# 5. Emergency-brake fallback / availability
def test_emergency_brake_fallback_and_availability():
    controller = HeuristicController()
    # Immediate threat should choose emergency brake when available.
    obs = _obs(
        ego=_ego(speed=10.0),
        route=_route(speed_limit=15.0),
        candidates=[_candidate(3.0, 0.0, vx=-2.0, vy=0.0, actor_id=1)],
        mask=make_action_mask(False, False),
    )
    assert controller.act(obs) == EMERGENCY_BRAKE

    # Even if mask bit is off in obs, controller must preserve emergency brake.
    obs2 = _obs(
        ego=_ego(speed=10.0),
        route=_route(speed_limit=15.0),
        candidates=[_candidate(3.0, 0.0, vx=-2.0, vy=0.0, actor_id=1)],
    )
    obs2[MASK_START + EMERGENCY_BRAKE] = 0.0
    action = controller.act(obs2)
    assert action == EMERGENCY_BRAKE


# 6. Empty candidate handling
def test_empty_candidate_handling():
    controller = HeuristicController()
    obs = _obs(candidates=[])
    cand_block = obs[CAND_START:MASK_START].reshape(M_CANDIDATES, CAND_FEATURES)
    assert np.all(cand_block[:, C_VALID] == 0.0)
    action = controller.act(obs)
    assert 0 <= int(action) < MASK


# 7. Invalid candidate handling
def test_invalid_candidate_handling():
    controller = HeuristicController()
    # Safe baseline
    obs_safe = _obs(ego=_ego(speed=5.0), candidates=[])
    obs_safe[IDX_ROUTE_SPEED_LIMIT] = 15.0 / V_MAX
    obs_safe[MASK_START:MASK_START + MASK] = 1.0
    action_safe = controller.act(obs_safe.copy())

    # Threatening but invalid/padded slot must be ignored.
    obs_invalid = obs_safe.copy()
    _set_candidate(obs_invalid, 0, rel_x=0.05, rel_y=0.0, rel_vx=-0.2, rel_vy=0.0, rel_heading=0.0, valid=0.0)
    action_invalid = controller.act(obs_invalid.copy())
    assert action_invalid == action_safe

    parsed = controller.parse_candidates(obs_invalid)
    assert parsed[0].valid is False


# 8. Response to a nearby vehicle
def test_response_to_nearby_vehicle():
    controller = HeuristicController()
    mask = make_action_mask(False, False)
    obs_clear = _obs(ego=_ego(speed=5.0), route=_route(speed_limit=15.0), candidates=[], mask=mask)
    obs_near = _obs(
        ego=_ego(speed=5.0),
        route=_route(speed_limit=15.0),
        candidates=[_candidate(4.0, 0.0, vx=0.0, vy=0.0, actor_id=1)],  # ~4m ahead
        mask=mask,
    )
    a_clear = controller.act(obs_clear)
    a_near = controller.act(obs_near)
    assert a_clear in (ACCEL, KEEP)
    assert a_near in (EMERGENCY_BRAKE, DECEL)
    assert a_clear != a_near


# 9. Response to a closing vehicle based on relative velocity
def test_response_to_closing_vehicle_based_on_relative_velocity():
    controller = HeuristicController()
    mask = make_action_mask(False, False)
    # Same position (~7 m ahead, caution zone edge), not critically close by distance alone.
    obs_receding = _obs(
        ego=_ego(speed=8.0),
        route=_route(speed_limit=15.0),
        candidates=[_candidate(7.0, 0.0, vx=2.0, vy=0.0, actor_id=1)],  # pulling away
        mask=mask,
    )
    obs_closing = _obs(
        ego=_ego(speed=8.0),
        route=_route(speed_limit=15.0),
        candidates=[_candidate(7.0, 0.0, vx=-2.0, vy=0.0, actor_id=1)],  # closing fast
        mask=mask,
    )
    a_receding = controller.act(obs_receding)
    a_closing = controller.act(obs_closing)
    assert a_closing in (EMERGENCY_BRAKE, DECEL)
    assert a_receding != a_closing


# 10. Different valid candidate observations can produce different decisions
def test_different_valid_candidate_observations_can_produce_different_decisions():
    controller = HeuristicController()
    mask = make_action_mask(True, True)
    ego = _ego(speed=8.0)
    route = _route(speed_limit=15.0, next_waypoint_x=20.0, next_waypoint_y=0.0)
    ahead_blocker = _candidate(7.0, 0.0, vx=-1.0, vy=0.0, actor_id=1)
    left_lane_vehicle = _candidate(7.0, 4.0, vx=0.0, vy=0.0, actor_id=2)
    right_lane_vehicle = _candidate(7.0, -4.0, vx=0.0, vy=0.0, actor_id=3)

    obs_a = _obs(ego=ego, route=route, candidates=[ahead_blocker, left_lane_vehicle], mask=mask)
    obs_b = _obs(ego=ego, route=route, candidates=[ahead_blocker, right_lane_vehicle], mask=mask)

    a = HeuristicController().act(obs_a)
    b = HeuristicController().act(obs_b)
    assert a == LANE_RIGHT
    assert b == LANE_LEFT
    assert a != b


# 11. No use of global random state
def test_no_use_of_global_random_state():
    obs = _obs(
        ego=_ego(speed=8.0),
        route=_route(speed_limit=15.0),
        candidates=[_candidate(7.0, 0.0, vx=-1.0, vy=0.0, actor_id=1)],
        mask=make_action_mask(True, True),
    )
    np.random.seed(12345)
    state_before = np.random.get_state()[1].copy()
    controller = HeuristicController()
    out = [controller.act(obs.copy()) for _ in range(5)]
    state_after = np.random.get_state()[1].copy()
    assert len(set(out)) == 1
    assert np.array_equal(state_before, state_after)

    # Global python/np random state should not affect result.
    np.random.seed(0)
    c1 = HeuristicController().act(obs.copy())
    np.random.seed(999999)
    c2 = HeuristicController().act(obs.copy())
    assert c1 == c2


# 12. All returned actions belong to the existing action space
def test_all_returned_actions_belong_to_existing_action_space():
    controller = HeuristicController()
    cases = [
        _obs(ego=_ego(speed=15.0), route=_route(speed_limit=15.0), candidates=[]),  # keep
        _obs(ego=_ego(speed=5.0), route=_route(speed_limit=15.0), candidates=[]),   # accel
        _obs(ego=_ego(speed=20.0), route=_route(speed_limit=10.0), candidates=[]),  # decel
        _obs(ego=_ego(speed=10.0), candidates=[_candidate(3.0, 0.0, vx=-2.0, actor_id=1)], mask=make_action_mask(False, False)),  # brake
        _obs(ego=_ego(speed=8.0), route=_route(speed_limit=15.0, next_waypoint_y=2.0), candidates=[_candidate(7.0, 0.0, vx=-1.0, actor_id=1)], mask=make_action_mask(True, True)),  # left
        _obs(ego=_ego(speed=8.0), route=_route(speed_limit=15.0, next_waypoint_y=-2.0), candidates=[_candidate(7.0, 0.0, vx=-1.0, actor_id=1)], mask=make_action_mask(True, True)),  # right
    ]
    actions = [int(controller.act(c)) for c in cases]
    assert all(a in {KEEP, ACCEL, DECEL, EMERGENCY_BRAKE, LANE_LEFT, LANE_RIGHT} for a in actions)
    assert set(actions) == {KEEP, ACCEL, DECEL, EMERGENCY_BRAKE, LANE_LEFT, LANE_RIGHT}
