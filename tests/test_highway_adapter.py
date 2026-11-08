import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

import envs.highway_env as highway_env
import envs.obs_builder as obs_builder
from envs.highway_env import HighwayV2XEnv


LAYOUTS = ("kinematics25", "contract56")


@pytest.fixture
def contract_env():
    env = HighwayV2XEnv(render_mode=None, obs_layout="contract56")
    try:
        yield env
    finally:
        env.close()


def _make_env(layout):
    return HighwayV2XEnv(render_mode=None, obs_layout=layout)


def _step_actions(env):
    return {agent_id: 0 for agent_id in env.agents}


def _candidate_rows(observation):
    start = obs_builder.EGO + obs_builder.ROUTE
    stop = start + obs_builder.CAND
    return observation[start:stop].reshape(
        obs_builder.M_CANDIDATES,
        obs_builder.CAND_FEATURES,
    )


def _contract_module():
    path = Path(__file__).with_name("test_contract.py")
    spec = importlib.util.spec_from_file_location("highway_contract_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("layout", LAYOUTS)
def test_layout_dimensions_and_defaults(layout):
    env = _make_env(layout)
    try:
        expected = 56 if layout == "contract56" else 25
        assert env.obs_dim == expected
        assert env.n_actions == 6
        observations, _ = env.reset(seed=0)
        for observation in observations.values():
            assert observation.shape == (expected,)
            assert observation.dtype == np.float32
    finally:
        env.close()


def test_contract56_values_over_fifty_short_episodes(contract_env):
    for seed in range(50):
        observations, _ = contract_env.reset(seed=seed)
        for _ in range(3):
            observations, _, _, _, _ = contract_env.step(
                _step_actions(contract_env)
            )
            for observation in observations.values():
                assert observation.shape == (56,)
                assert observation.dtype == np.float32
                assert np.all(np.isfinite(observation))
                assert np.all(observation >= -1.0)
                assert np.all(observation <= 1.0)


def test_contract56_block_boundaries(contract_env):
    observations, _ = contract_env.reset(seed=0)
    for agent_id, observation in observations.items():
        ego = observation[: obs_builder.EGO]
        route = observation[obs_builder.EGO: obs_builder.EGO + obs_builder.ROUTE]
        candidates = _candidate_rows(observation)
        mask = observation[-obs_builder.MASK:]

        assert ego.shape == (8,)
        assert route.shape == (6,)
        assert candidates.shape == (6, 6)
        assert mask.shape == (6,)
        assert np.array_equal(
            mask.astype(bool),
            contract_env.action_masks()[agent_id],
        )


def test_candidates_are_within_range_sorted_and_at_most_six(contract_env):
    observations, _ = contract_env.reset(seed=0)
    for agent_id in contract_env.agents:
        candidates = contract_env.get_candidates(agent_id)
        distances = []
        ego = contract_env._vehicle_for_agent(agent_id)
        for candidate in candidates:
            distance = math.hypot(
                candidate["x"] - float(ego.position[0]),
                candidate["y"] - float(ego.position[1]),
            )
            distances.append(distance)
            assert distance <= obs_builder.CANDIDATE_RANGE
            assert candidate["valid"] is True
        assert distances == sorted(distances)
        rows = _candidate_rows(observations[agent_id])
        assert np.count_nonzero(rows[:, 5]) <= obs_builder.M_CANDIDATES


def test_candidate_padding_is_zero_with_invalid_validity(contract_env, monkeypatch):
    monkeypatch.setattr(contract_env, "get_candidates", lambda agent_id: [])
    observations, _ = contract_env.reset(seed=0)
    rows = _candidate_rows(observations["cav_0"])
    assert np.all(rows == 0.0)


def test_get_candidates_matches_encoded_rows(contract_env):
    observations, _ = contract_env.reset(seed=0)
    for agent_id in contract_env.agents:
        ego = contract_env._vehicle_for_agent(agent_id)
        ego_x = float(ego.position[0])
        ego_y = float(ego.position[1])
        yaw = float(ego.heading)
        ego_vx = float(ego.velocity[0])
        ego_vy = float(ego.velocity[1])
        rows = _candidate_rows(observations[agent_id])
        candidates = contract_env.get_candidates(agent_id)[: obs_builder.M_CANDIDATES]
        assert len(candidates) >= obs_builder.M_CANDIDATES
        for index, candidate in enumerate(candidates):
            rel_x, rel_y = obs_builder.to_ego_frame(
                candidate["x"],
                candidate["y"],
                ego_x,
                ego_y,
                yaw,
            )
            rel_vx, rel_vy = obs_builder.to_ego_frame(
                candidate["vx"] - ego_vx,
                candidate["vy"] - ego_vy,
                0.0,
                0.0,
                yaw,
            )
            rel_heading = math.atan2(
                math.sin(candidate["yaw"] - yaw),
                math.cos(candidate["yaw"] - yaw),
            )
            expected = np.array(
                [
                    np.clip(rel_x / obs_builder.CANDIDATE_RANGE, -1.0, 1.0),
                    np.clip(rel_y / obs_builder.CANDIDATE_RANGE, -1.0, 1.0),
                    np.clip(rel_vx / obs_builder.V_MAX, -1.0, 1.0),
                    np.clip(rel_vy / obs_builder.V_MAX, -1.0, 1.0),
                    np.clip(rel_heading / math.pi, -1.0, 1.0),
                    1.0,
                ],
                dtype=np.float32,
            )
            assert np.allclose(rows[index], expected, atol=1e-6)


def test_equal_distance_candidates_use_actor_id_tie_break(contract_env):
    observations, _ = contract_env.reset(seed=0)
    ego = contract_env._vehicle_for_agent("cav_0")
    x = float(ego.position[0]) + 10.0
    y = float(ego.position[1])
    yaw = float(ego.heading)
    candidates = [
        {
            "actor_id": 9,
            "x": x,
            "y": y,
            "vx": 1.0,
            "vy": 0.0,
            "yaw": yaw,
            "valid": True,
        },
        {
            "actor_id": 3,
            "x": x,
            "y": y,
            "vx": 2.0,
            "vy": 0.0,
            "yaw": yaw,
            "valid": True,
        },
    ]
    observation = obs_builder.build_observation(
        {
            "x": float(ego.position[0]),
            "y": float(ego.position[1]),
            "yaw": yaw,
            "lane_width": 4.0,
        },
        {"next_waypoint_x": x, "next_waypoint_y": y},
        candidates,
        contract_env.action_masks()["cav_0"],
    )
    rows = _candidate_rows(observation)
    assert np.allclose(rows[0, 5], 1.0)
    assert np.allclose(rows[1, 5], 1.0)
    assert rows[0, 2] == pytest.approx(2.0 / obs_builder.V_MAX)
    assert rows[1, 2] == pytest.approx(1.0 / obs_builder.V_MAX)


def test_action_masks_match_observation_and_edge_lanes(contract_env):
    observations, _ = contract_env.reset(seed=0)
    network = contract_env.env.unwrapped.road.network
    for agent_id in contract_env.agents:
        vehicle = contract_env._vehicle_for_agent(agent_id)
        target = vehicle.target_lane_index
        road_key = tuple(target[:2])
        lane_ids = [
            index[2]
            for index in network.lanes_dict()
            if tuple(index[:2]) == road_key
        ]
        mask = contract_env.action_masks()[agent_id]
        observed = observations[agent_id][-obs_builder.MASK:].astype(bool)
        assert np.array_equal(mask, observed)
        assert mask.any()
        assert mask[0] and mask[1] and mask[2] and mask[3]
        if int(target[2]) == min(lane_ids):
            assert not mask[4]
        if int(target[2]) == max(lane_ids):
            assert not mask[5]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_seed_determinism(layout):
    env = _make_env(layout)
    try:
        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        for agent_id in env.agents:
            assert np.array_equal(first[agent_id], second[agent_id])
        different, _ = env.reset(seed=8)
        assert any(
            not np.allclose(first[agent_id], different[agent_id])
            for agent_id in env.agents
        )
    finally:
        env.close()


def test_adapter_uses_carla_obs_builder(contract_env, monkeypatch):
    captured = []
    original = highway_env.build_observation

    def wrapper(ego, route, candidates, action_mask):
        captured.append((ego, route, candidates, action_mask))
        return original(ego, route, candidates, action_mask)

    monkeypatch.setattr(highway_env, "build_observation", wrapper)
    observations, _ = contract_env.reset(seed=0)
    assert captured
    for index, (ego, route, candidates, action_mask) in enumerate(
        captured[: len(contract_env.agents)]
    ):
        expected = original(ego, route, candidates, action_mask)
        agent_id = contract_env.agents[index]
        assert np.array_equal(observations[agent_id], expected)


def test_goal_distance_default_is_disabled():
    env = HighwayV2XEnv(render_mode=None, goal_distance_m=None)
    try:
        env.reset(seed=0)
        for _ in range(5):
            _, rewards, terminated, truncated, info = env.step(_step_actions(env))
            assert info["goal_reached"] is False
            assert info["success"] is False
            assert not any(terminated.values())
            assert not any(truncated.values())
            for agent_id in env.agents:
                assert info["per_agent"][agent_id]["goal_reached"] is False
                assert rewards[agent_id] == pytest.approx(
                    sum(info["per_agent"][agent_id]["reward_terms"].values())
                )
    finally:
        env.close()


def test_goal_distance_fires_once_and_success():
    env = HighwayV2XEnv(render_mode=None, goal_distance_m=1.0)
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(_step_actions(env))
        assert info["goal_reached"] is True
        assert info["success"] is True
        assert all(terminated.values())
        assert not any(truncated.values())
        for agent_id in env.agents:
            assert info["per_agent"][agent_id]["goal_reached"] is True
            assert info["per_agent"][agent_id]["reward_terms"]["goal"] == pytest.approx(
                10.0
            )
    finally:
        env.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_full_contract_suite_for_layout(layout):
    contract = _contract_module()
    env = _make_env(layout)
    try:
        contract.test_reset_format(env)
        contract.test_action_masks(env)
        contract.test_state(env)
        contract.test_step_format(env)
        contract.test_episode_terminates(env)
        contract.test_attributes(env)
        contract.test_seed_determinism(env)
        contract.test_seed_changes_episode(env)
        contract.test_info_keys(env)
    finally:
        env.close()
