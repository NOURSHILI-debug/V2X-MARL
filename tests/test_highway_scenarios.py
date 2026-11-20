import json
from pathlib import Path

import numpy as np
import pytest

from envs.base import Action, N_ACTIONS
from envs.highway_env import HighwayV2XEnv
from envs.obs_builder import (
    CANDIDATE_RANGE,
    CAND_FEATURES,
    EGO,
    M_CANDIDATES,
    OBS_DIM,
    ROUTE,
)
from envs.scenarios.highway_scenarios import (
    load_profiles,
    parse_profile,
    resolve_scenario,
)


BASELINE_PATH = Path(__file__).with_name("data") / "highway_no_scenario_baseline.json"
PROFILE_IDS = (
    "h1_moderate",
    "h1_dense",
    "h1_mixed_speed",
    "h1_forcing",
    "h2_lane_change",
)
VALID_PROFILE = {
    "scenario_id": "unit_profile",
    "lanes_count": 4,
    "background_vehicles": 12,
    "vehicles_density": 3.0,
    "duration_s": 60.0,
    "spacing_m": 20.0,
    "spacing_jitter_m": 0.5,
    "speeds_mps": {"mode": "uniform", "min": 20.0, "max": 25.0},
    "cav": {
        "lanes": [0, 1, 2, 3],
        "start_x_m": [70.0, 70.0, 70.0, 70.0],
        "speed_mps": 22.0,
    },
}


def jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        if "native_info" in value and isinstance(value["native_info"], dict):
            value = dict(value)
            value["native_info"] = {
                key: item
                for key, item in value["native_info"].items()
                if key != "action"
            }
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return f"<{type(value).__name__}:{value!r}>"


def _make_profile_env(profile_id, seed=100):
    env = HighwayV2XEnv(
        render_mode=None,
        obs_layout="contract56",
        scenario=profile_id,
    )
    observations, info = env.reset(seed=seed)
    return env, observations, info


def _candidate_rows(observation):
    start = EGO + ROUTE
    return observation[start : start + M_CANDIDATES * CAND_FEATURES].reshape(
        M_CANDIDATES, CAND_FEATURES
    )


def _background_vehicles(env):
    native = env.env.unwrapped
    controlled = set(native.controlled_vehicles)
    return [vehicle for vehicle in native.road.vehicles if vehicle not in controlled]


def _measured_density(profile_id, seed):
    env, _, _ = _make_profile_env(profile_id, seed)
    try:
        native = env.env.unwrapped
        vehicles = list(native.road.vehicles)
        gaps = []
        for lane_index in {vehicle.lane_index for vehicle in vehicles}:
            xs = sorted(
                float(vehicle.position[0])
                for vehicle in vehicles
                if vehicle.lane_index == lane_index
            )
            gaps.extend(float(end - start) for start, end in zip(xs, xs[1:]))
        speeds = [float(vehicle.speed) for vehicle in _background_vehicles(env)]
        return {
            "count": len(_background_vehicles(env)),
            "mean_gap_m": float(np.mean(gaps)),
            "speed_spread_mps": float(max(speeds) - min(speeds)),
        }
    finally:
        env.close()


def test_no_scenario_matches_prechange_baseline():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    for layout, expected_layout in baseline["layouts"].items():
        env = HighwayV2XEnv(render_mode=None, obs_layout=layout)
        try:
            assert env.scenario is None
            assert env.obs_dim == expected_layout["obs_dim"]
            assert env.n_actions == expected_layout["n_actions"]
            assert env.agents == expected_layout["agents"]
            for seed, expected_steps in expected_layout["seeds"].items():
                observations, info = env.reset(seed=int(seed))
                actual = {
                    "reset": {
                        "obs": {a: jsonable(observations[a]) for a in env.agents},
                        "info": jsonable(info),
                    }
                }
                assert actual == expected_steps[0]
                for index, expected_step in enumerate(expected_steps[1:], start=1):
                    observations, rewards, terminated, truncated, info = env.step(
                        {agent_id: 0 for agent_id in env.agents}
                    )
                    actual = {
                        "step": {
                            "obs": {
                                agent_id: jsonable(observations[agent_id])
                                for agent_id in env.agents
                            },
                            "rewards": jsonable(rewards),
                            "terminated": jsonable(terminated),
                            "truncated": jsonable(truncated),
                            "info": jsonable(info),
                        }
                    }
                    assert actual == expected_step, (layout, seed, index)
        finally:
            env.close()


def test_resolve_none_and_mapping_profile():
    assert resolve_scenario(None) is None
    profile = resolve_scenario(VALID_PROFILE)
    assert profile.scenario_id == "unit_profile"
    assert profile.background_vehicles == 12
    assert profile.env_config()["vehicles_count"] == 12


def test_profile_validation_rejects_contract_violations():
    too_few = dict(VALID_PROFILE, background_vehicles=11)
    with pytest.raises(ValueError, match="at least 12"):
        parse_profile(too_few)
    bad_cav = dict(
        VALID_PROFILE,
        cav={"lanes": [0, 1, 2], "start_x_m": [1, 2, 3, 4], "speed_mps": 20},
    )
    with pytest.raises(ValueError, match="four values"):
        parse_profile(bad_cav)


def test_config_file_defines_expected_profiles():
    profiles = load_profiles()
    assert set(profiles) == set(PROFILE_IDS)
    assert profiles["h1_moderate"].background_vehicles == 12
    assert profiles["h1_dense"].background_vehicles == 20
    assert profiles["h1_mixed_speed"].background_vehicles == 17
    assert profiles["h1_mixed_speed"].speeds.mode == "groups"
    assert len(profiles["h1_forcing"].forcing.roles) == 5
    assert len(profiles["h2_lane_change"].forcing.roles) == 7


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_profile_initializes_four_cavs_and_contract_observations(profile_id):
    env, observations, info = _make_profile_env(profile_id)
    try:
        native = env.env.unwrapped
        assert env.agents == ["cav_0", "cav_1", "cav_2", "cav_3"]
        assert env.obs_dim == OBS_DIM == 56
        assert env.n_actions == N_ACTIONS == 6
        assert info["scenario_id"] == profile_id
        assert len(native.controlled_vehicles) == 4
        assert len(_background_vehicles(env)) >= 12
        assert not any(vehicle.crashed for vehicle in native.road.vehicles)
        assert native.config["duration"] == 60.0
        assert native.config["policy_frequency"] == 5
        assert native.config["vehicles_density"] == load_profiles()[profile_id].vehicles_density
        for observation in observations.values():
            assert observation.shape == (56,)
            assert observation.dtype == np.float32
            assert np.all(np.isfinite(observation))
            assert np.all(np.abs(observation) <= 1.0)
    finally:
        env.close()


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_candidate_block_range_padding_and_masks(profile_id):
    env, observations, _ = _make_profile_env(profile_id)
    try:
        masks = env.action_masks()
        for agent_id in env.agents:
            observation = observations[agent_id]
            rows = _candidate_rows(observation)
            candidates = env.get_candidates(agent_id)
            assert rows.shape == (M_CANDIDATES, CAND_FEATURES)
            assert len(candidates) >= M_CANDIDATES
            valid_rows = int(np.count_nonzero(rows[:, -1]))
            assert valid_rows == M_CANDIDATES
            assert np.all(rows[valid_rows:] == 0.0)
            ego = env._vehicle_for_agent(agent_id)
            distances = [
                float(
                    np.hypot(
                        candidate["x"] - float(ego.position[0]),
                        candidate["y"] - float(ego.position[1]),
                    )
                )
                for candidate in candidates
            ]
            assert max(distances) <= CANDIDATE_RANGE
            assert distances == sorted(distances)
            mask = masks[agent_id]
            assert mask.shape == (6,)
            assert mask.dtype == bool
            assert mask[:4].all()
            assert mask[4] or mask[5]
            assert np.array_equal(mask, observation[-6:].astype(bool))
        assert tuple(int(action) for action in Action) == (0, 1, 2, 3, 4, 5)
    finally:
        env.close()


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_profile_seed_reproducibility(profile_id):
    env_a, obs_a, _ = _make_profile_env(profile_id, seed=100)
    env_b, obs_b, _ = _make_profile_env(profile_id, seed=100)
    env_c, obs_c, _ = _make_profile_env(profile_id, seed=101)
    try:
        for agent_id in env_a.agents:
            assert np.array_equal(obs_a[agent_id], obs_b[agent_id])
            assert not np.array_equal(obs_a[agent_id], obs_c[agent_id])
        states_a = env_a.state()
        states_b = env_b.state()
        assert np.array_equal(states_a, states_b)
    finally:
        env_a.close()
        env_b.close()
        env_c.close()


def test_profiles_differ_in_measured_count_spacing_and_speed_spread():
    measured = {
        profile_id: [
            _measured_density(profile_id, seed) for seed in (100, 101, 102)
        ]
        for profile_id in PROFILE_IDS
    }
    means = {
        profile_id: {
            key: float(np.mean([sample[key] for sample in samples]))
            for key in samples[0]
        }
        for profile_id, samples in measured.items()
    }
    assert means["h1_dense"]["count"] > means["h1_moderate"]["count"]
    assert means["h1_dense"]["mean_gap_m"] < means["h1_moderate"]["mean_gap_m"]
    assert (
        means["h1_mixed_speed"]["speed_spread_mps"]
        > means["h1_moderate"]["speed_spread_mps"]
    )
    assert (
        means["h1_mixed_speed"]["speed_spread_mps"]
        > means["h1_dense"]["speed_spread_mps"]
    )


def test_300_step_limit_is_respected():
    env, _, _ = _make_profile_env("h1_moderate", seed=100)
    try:
        ended = False
        for _ in range(300):
            _, _, terminated, truncated, info = env.step(
                {agent_id: 0 for agent_id in env.agents}
            )
            if all(terminated.values()) or any(truncated.values()):
                ended = True
                break
        assert ended
        assert 0 < int(info["steps"]) <= 300
    finally:
        env.close()


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_reward_behavior_is_unchanged_for_profiles(profile_id):
    env, _, _ = _make_profile_env(profile_id)
    try:
        for _ in range(3):
            _, rewards, _, _, info = env.step({agent_id: 0 for agent_id in env.agents})
            for agent_id in env.agents:
                terms = info["per_agent"][agent_id]["reward_terms"]
                assert set(terms) == {
                    "progress",
                    "collision",
                    "safety",
                    "comfort",
                    "goal",
                    "idle",
                }
                assert rewards[agent_id] == pytest.approx(sum(terms.values()))
            expected_team = float(np.mean(list(rewards.values())))
            if info["collision"]:
                expected_team -= 10.0
            assert info["team_reward"] == pytest.approx(expected_team)
    finally:
        env.close()


def test_invalid_scenario_selection_fails_explicitly():
    with pytest.raises(ValueError, match="unknown scenario"):
        HighwayV2XEnv(
            render_mode=None,
            obs_layout="contract56",
            scenario="not_a_profile",
        )
    with pytest.raises(ValueError, match="contract56"):
        HighwayV2XEnv(
            render_mode=None,
            obs_layout="kinematics25",
            scenario="h1_moderate",
        )
