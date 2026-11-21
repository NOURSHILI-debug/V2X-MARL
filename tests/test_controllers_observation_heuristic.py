"""CARLA-free tests for controllers.observation_heuristic (Milestone 8A)."""

from __future__ import annotations

import numpy as np
import pytest

from envs.obs_builder import (
    CAND_FEATURES,
    CANDIDATE_RANGE,
    KEEP,
    ACCEL,
    DECEL,
    EMERGENCY_BRAKE,
    LANE_LEFT,
    LANE_RIGHT,
    MASK,
    M_CANDIDATES,
    OBS_DIM,
    V_MAX,
    build_observation,
    make_action_mask,
)
from controllers.observation_heuristic import (
    CAND_START,
    C_VALID,
    IDX_EGO_SPEED,
    IDX_ROUTE_SPEED_LIMIT,
    MASK_START,
    ObservationHeuristicController,
)


def _ego(**overrides):
    ego = {
        "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0,
        "speed": 0.0, "longitudinal_accel": 0.0, "heading_error": 0.0,
        "lateral_offset": 0.0, "lane_width": 3.5, "lane_index": 1.0,
        "lane_count": 3.0, "distance_to_goal": 100.0, "progress_fraction": 0.2,
    }
    ego.update(overrides)
    return ego


def _route(**overrides):
    route = {
        "next_waypoint_x": 20.0, "next_waypoint_y": 0.0,
        "speed_limit": 15.0, "at_intersection": False,
        "traffic_light_state": "none", "distance_to_next": 15.0,
    }
    route.update(overrides)
    return route


def _candidate(x, y, vx=0.0, vy=0.0, yaw=0.0, actor_id=100, valid=True):
    return {"actor_id": actor_id, "x": float(x), "y": float(y), "vx": float(vx), "vy": float(vy), "yaw": float(yaw), "valid": bool(valid)}


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


def test_observation_parsing():
    controller = ObservationHeuristicController()
    obs = _obs(ego=_ego(speed=10.0), route=_route(speed_limit=20.0, traffic_light_state="red", distance_to_next=5.0))
    arr = controller.validate_observation(obs)
    assert arr.shape == (OBS_DIM,)
    assert arr.dtype == np.float32
    assert np.all(np.isfinite(arr))
    assert arr[IDX_EGO_SPEED] == pytest.approx(10.0 / V_MAX)
    assert arr[IDX_ROUTE_SPEED_LIMIT] == pytest.approx(20.0 / V_MAX)
    action = controller.act(arr)
    assert 0 <= action < MASK


def test_observation_shape_rejected():
    controller = ObservationHeuristicController()
    with pytest.raises(ValueError):
        controller.validate_observation(np.zeros(55, dtype=np.float32))
    with pytest.raises(ValueError):
        controller.validate_observation(np.zeros((56, 1), dtype=np.float32))


def test_candidate_extraction():
    controller = ObservationHeuristicController()
    obs = _obs(candidates=[_candidate(10.0, 0.0, actor_id=1, valid=True), _candidate(12.0, 1.0, actor_id=2, valid=False)])
    candidates = controller.parse_candidates(obs)
    assert len(candidates) == M_CANDIDATES
    valid = [c for c in candidates if c.valid]
    assert len(valid) == 1
    assert valid[0].rel_x > 0.0  # 10 m ahead normalized
    assert abs(valid[0].rel_y) < 1e-6
    # padded slots invalid
    assert sum(1 for c in candidates if not c.valid) == M_CANDIDATES - 1


def test_action_mask_handling_always_respected():
    controller = ObservationHeuristicController()
    obs = _obs(ego=_ego(speed=3.0), route=_route(speed_limit=15.0))
    mask = np.array([False, False, True, True, False, False], dtype=bool)  # DECEL, BRAKE only
    obs[MASK_START:MASK_START + MASK] = mask.astype(np.float32)
    for _ in range(20):
        action = controller.act(obs.copy())
        assert action in (DECEL, EMERGENCY_BRAKE)
    # explicit mask argument respected
    obs2 = _obs(ego=_ego(speed=3.0), route=_route(speed_limit=15.0))
    action2 = controller.act(obs2, action_mask=np.array([True, False, False, True, False, False]))
    assert action2 in (KEEP, EMERGENCY_BRAKE)


def test_emergency_brake_always_available_and_never_invalid_action():
    controller = ObservationHeuristicController()
    obs = _obs(ego=_ego(speed=10.0), candidates=[_candidate(3.0, 0.0, vx=-2.0, actor_id=1)])
    obs[MASK_START:MASK_START + MASK] = 0.0  # degenerate input
    action = controller.act(obs)
    assert action == EMERGENCY_BRAKE
    # even if other mask bits claim allowed, returned action must be in allowed set (EBR always allowed)
    mask = controller.parse_action_mask(obs)
    assert mask[EMERGENCY_BRAKE]


def test_empty_candidate_handling_safe():
    controller = ObservationHeuristicController()
    obs = _obs(candidates=[])
    cand_block = obs[CAND_START:MASK_START].reshape(M_CANDIDATES, CAND_FEATURES)
    assert np.all(cand_block[:, C_VALID] == 0.0)
    action = controller.act(obs)
    assert 0 <= action < MASK
    # no errors across multiple calls
    for _ in range(10):
        assert 0 <= controller.act(obs.copy()) < MASK


def test_invalid_candidates_do_not_affect_decisions():
    controller = ObservationHeuristicController()
    base = _obs(ego=_ego(speed=5.0), route=_route(speed_limit=15.0), candidates=[])
    base[MASK_START:MASK_START + MASK] = 1.0
    a_base = controller.act(base.copy())

    # slot marked invalid but with threatening values -> must be ignored
    obs_invalid = base.copy()
    _set_candidate(obs_invalid, 0, rel_x=0.05, rel_y=0.0, rel_vx=-0.2, rel_vy=0.0, rel_heading=0.0, valid=0.0)
    a_invalid = controller.act(obs_invalid.copy())
    assert a_invalid == a_base

    # non-finite candidate values treated as invalid by parser
    obs_badvals = base.copy()
    _set_candidate(obs_badvals, 1, rel_x=np.nan, rel_y=0.0, rel_vx=0.0, rel_vy=0.0, rel_heading=0.0, valid=1.0)
    parsed = controller.parse_candidates(obs_badvals)
    assert parsed[1].valid is False
    a_badvals = controller.act(obs_badvals.copy())
    assert 0 <= a_badvals < MASK


def test_determinism():
    controller = ObservationHeuristicController()
    obs = _obs(candidates=[_candidate(10.0, 0.0, actor_id=1)])
    actions = [controller.act(obs.copy()) for _ in range(30)]
    assert len(set(actions)) == 1
    assert ObservationHeuristicController().act(obs.copy()) == actions[0]


def test_candidate_sensitive_behavior_same_ego_route_different_candidates():
    """Controlled evidence: teammate observations genuinely influence decisions."""
    ego = _ego(speed=8.0)
    route = _route(speed_limit=15.0, next_waypoint_x=20.0, next_waypoint_y=0.0)
    mask = make_action_mask(True, True)
    ahead_blocker = _candidate(7.0, 0.0, vx=-1.0, vy=0.0, actor_id=1)

    # Case A: left lane occupied by selected teammate, right clear -> prefer/go right
    obs_a = _obs(ego=ego, route=route, candidates=[ahead_blocker, _candidate(7.0, 4.0, actor_id=2)], mask=mask)
    # Case B: right lane occupied by selected teammate, left clear -> prefer/go left
    obs_b = _obs(ego=ego, route=route, candidates=[ahead_blocker, _candidate(7.0, -4.0, actor_id=3)], mask=mask)

    a = ObservationHeuristicController().act(obs_a)
    b = ObservationHeuristicController().act(obs_b)
    assert a == LANE_RIGHT
    assert b == LANE_LEFT
    assert a != b

    # Case C: no candidates visible (empty selection) -> different from blocked-ahead case
    obs_clear = _obs(ego=_ego(speed=5.0), route=_route(speed_limit=15.0), candidates=[], mask=make_action_mask(False, False))
    obs_near = _obs(ego=_ego(speed=5.0), route=_route(speed_limit=15.0), candidates=[_candidate(4.0, 0.0, vx=0.0, actor_id=1)], mask=make_action_mask(False, False))
    assert ObservationHeuristicController().act(obs_clear) in (ACCEL, KEEP)
    assert ObservationHeuristicController().act(obs_near) in (EMERGENCY_BRAKE, DECEL)
    assert ObservationHeuristicController().act(obs_clear) != ObservationHeuristicController().act(obs_near)


def test_invalid_numerical_values_handled_safely():
    controller = ObservationHeuristicController()
    obs = _obs(ego=_ego(speed=5.0), route=_route(speed_limit=15.0))
    obs[IDX_EGO_SPEED] = np.nan
    obs[IDX_ROUTE_SPEED_LIMIT] = np.inf
    obs[CAND_START + 0] = -np.inf
    # Should not raise; must return valid, mask-compliant action deterministically
    mask = make_action_mask(True, True)
    obs[MASK_START:MASK_START + MASK] = mask.astype(np.float32)
    a1 = controller.act(obs.copy(), action_mask=mask)
    a2 = controller.act(obs.copy(), action_mask=mask)
    assert a1 == a2
    assert 0 <= a1 < MASK
    assert bool(mask[a1])


def test_safety_related_decision_cases():
    controller = ObservationHeuristicController()
    # Immediate threat -> emergency brake
    obs_immediate = _obs(ego=_ego(speed=10.0), candidates=[_candidate(3.0, 0.0, vx=-2.0, actor_id=1)], mask=make_action_mask(False, False))
    assert controller.act(obs_immediate) == EMERGENCY_BRAKE

    # Closing vehicle by relative velocity escalates response
    mask = make_action_mask(False, False)
    obs_receding = _obs(ego=_ego(speed=8.0), route=_route(speed_limit=15.0), candidates=[_candidate(7.0, 0.0, vx=2.0, actor_id=1)], mask=mask)
    obs_closing = _obs(ego=_ego(speed=8.0), route=_route(speed_limit=15.0), candidates=[_candidate(7.0, 0.0, vx=-2.0, actor_id=1)], mask=mask)
    a_rec = controller.act(obs_receding)
    a_clo = controller.act(obs_closing)
    assert a_clo in (EMERGENCY_BRAKE, DECEL)
    assert a_rec != a_clo

    # Red light hazard -> brake/decel
    obs_red = _obs(ego=_ego(speed=8.0), route=_route(speed_limit=15.0, traffic_light_state="red", distance_to_next=3.0), candidates=[])
    assert controller.act(obs_red) in (EMERGENCY_BRAKE, DECEL)

    # Safe following when clear ahead -> can accelerate below limit
    obs_clear = _obs(ego=_ego(speed=5.0), route=_route(speed_limit=15.0), candidates=[], mask=make_action_mask(True, True))
    assert controller.act(obs_clear) == ACCEL


def test_no_global_random_state_dependency():
    obs = _obs(ego=_ego(speed=8.0), route=_route(speed_limit=15.0), candidates=[_candidate(7.0, 0.0, vx=-1.0, actor_id=1)], mask=make_action_mask(True, True))
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    controller = ObservationHeuristicController()
    outs = [controller.act(obs.copy()) for _ in range(5)]
    after = np.random.get_state()[1].copy()
    assert len(set(outs)) == 1
    assert np.array_equal(before, after)
    np.random.seed(0); a1 = ObservationHeuristicController().act(obs.copy())
    np.random.seed(999); a2 = ObservationHeuristicController().act(obs.copy())
    assert a1 == a2
