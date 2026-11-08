import math

import numpy as np
import pytest

from envs.obs_builder import (
    CAND,
    CAND_FEATURES,
    EGO,
    MASK,
    OBS_DIM,
    ROUTE,
    V_MAX,
    build_observation,
    make_action_mask,
    to_ego_frame,
)


def _build(ego=None, route=None, candidates=(), mask=None):
    ego_values = {
        "x": 0.0,
        "y": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "yaw": 0.0,
        "speed": 10.0,
        "longitudinal_accel": 1.0,
        "heading_error": 0.0,
        "lateral_offset": 0.0,
        "lane_width": 4.0,
        "lane_index": 1.0,
        "lane_count": 3.0,
        "distance_to_goal": 80.0,
        "progress_fraction": 0.4,
    }
    ego_values.update(ego or {})
    route_values = {
        "next_waypoint_x": 10.0,
        "next_waypoint_y": 0.0,
        "speed_limit": 15.0,
        "at_intersection": False,
        "traffic_light_state": "none",
        "distance_to_next": 10.0,
    }
    route_values.update(route or {})
    return build_observation(
        ego_values,
        route_values,
        candidates,
        np.ones(MASK, dtype=bool) if mask is None else mask,
    )


def test_observation_shape_dtype_and_block_layout():
    observation = _build()

    assert observation.shape == (56,) == (EGO + ROUTE + CAND + MASK,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()


def test_candidate_padding_sets_validity_to_zero():
    observation = _build()
    candidate_block = observation[EGO + ROUTE : EGO + ROUTE + CAND].reshape(
        6, CAND_FEATURES
    )

    assert np.array_equal(candidate_block, np.zeros((6, 6), dtype=np.float32))


def test_candidates_are_sorted_by_distance_then_actor_id():
    candidates = [
        {"actor_id": 8, "x": 5.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0},
        {"actor_id": 9, "x": 3.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0},
        {"actor_id": 2, "x": -3.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0},
    ]
    observation = _build(candidates=candidates)
    candidate_block = observation[EGO + ROUTE : EGO + ROUTE + CAND].reshape(
        6, CAND_FEATURES
    )

    assert candidate_block[:3, 0].tolist() == pytest.approx(
        [-3.0 / 50.0, 3.0 / 50.0, 5.0 / 50.0]
    )
    assert np.all(candidate_block[:3, 5] == 1.0)
    assert np.all(candidate_block[3:, 5] == 0.0)


def test_ego_frame_rotation_for_quarter_turn():
    local = to_ego_frame(0.0, 1.0, 0.0, 0.0, math.pi / 2)

    assert local == pytest.approx((1.0, 0.0), abs=1e-7)


def test_lane_masks_keep_emergency_brake_valid():
    no_left = make_action_mask(left_lane_exists=False, right_lane_exists=True)
    no_right = make_action_mask(left_lane_exists=True, right_lane_exists=False)

    assert no_left.shape == (MASK,)
    assert not no_left[4] and no_left[5]
    assert no_right[4] and not no_right[5]
    assert no_left[3] and no_right[3]
    assert no_left.any() and no_right.any()


def test_random_inputs_and_nonfinite_values_never_produce_nan():
    rng = np.random.default_rng(42)
    for _ in range(100):
        ego = {
            "x": rng.normal(),
            "y": rng.normal(),
            "vx": rng.normal() * V_MAX,
            "vy": rng.normal() * V_MAX,
            "yaw": rng.uniform(-math.pi, math.pi),
            "speed": rng.uniform(0.0, 60.0),
            "longitudinal_accel": rng.normal() * 20,
            "heading_error": rng.normal() * math.pi,
            "lateral_offset": rng.normal() * 10,
            "lane_width": rng.uniform(2.0, 6.0),
            "lane_index": rng.integers(0, 8),
            "lane_count": rng.integers(1, 8),
            "distance_to_goal": rng.uniform(0.0, 500.0),
            "progress_fraction": rng.uniform(-2.0, 2.0),
        }
        candidates = [
            {
                "actor_id": actor_id,
                "x": ego["x"] + rng.normal() * 50,
                "y": ego["y"] + rng.normal() * 50,
                "vx": rng.normal() * 30,
                "vy": rng.normal() * 30,
                "yaw": rng.uniform(-math.pi, math.pi),
            }
            for actor_id in range(8)
        ]
        observation = _build(ego=ego, candidates=candidates)
        assert observation.dtype == np.float32
        assert observation.shape == (OBS_DIM,)
        assert np.isfinite(observation).all()

    observation = _build(
        ego={"speed": float("nan"), "longitudinal_accel": float("inf")},
        route={"next_waypoint_x": float("nan"), "distance_to_next": float("inf")},
    )
    assert np.isfinite(observation).all()


def test_all_features_are_clipped_to_their_ranges():
    observation = _build(
        ego={
            "speed": 1000.0,
            "longitudinal_accel": 1000.0,
            "heading_error": math.pi * 10,
            "lateral_offset": 1000.0,
            "lane_index": 100.0,
            "lane_count": 100.0,
            "distance_to_goal": 10000.0,
            "progress_fraction": 100.0,
        },
        route={
            "next_waypoint_x": 10000.0,
            "next_waypoint_y": -10000.0,
            "speed_limit": 1000.0,
            "at_intersection": True,
            "traffic_light_state": "red",
            "distance_to_next": 1000.0,
        },
        candidates=[
            {
                "actor_id": 1,
                "x": 30.0,
                "y": -30.0,
                "vx": 1000.0,
                "vy": -1000.0,
                "yaw": math.pi * 4,
            }
        ],
    )

    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)
    assert observation[3] == 1.0
    assert observation[4] == 1.0
    assert observation[5] == 1.0
    assert observation[6] == 1.0
    assert observation[7] == 1.0
    assert observation[EGO + ROUTE + 5] == 1.0
