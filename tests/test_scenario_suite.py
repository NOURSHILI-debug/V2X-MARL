import numpy as np
import pytest

from envs.base import N_ACTIONS
from envs.highway_env import HighwayV2XEnv
from envs.obs_builder import (
    CAND,
    CAND_FEATURES,
    EGO,
    M_CANDIDATES,
    OBS_DIM,
    ROUTE,
)
from envs.scenarios.highway_scenarios import load_profiles, parse_profile
from scripts.scripted_policies import competent_action

NEW_PROFILES = ("h1_forcing", "h2_lane_change")
VALID_PROFILE = {
    "scenario_id": "unit_forcing",
    "lanes_count": 4,
    "background_vehicles": 12,
    "vehicles_density": 3.0,
    "duration_s": 60.0,
    "spacing_m": 20.0,
    "spacing_jitter_m": 0.5,
    "speeds_mps": {"mode": "uniform", "min": 20.0, "max": 25.0},
    "cav": {
        "lanes": [1, 1, 0, 2],
        "start_x_m": 80.0,
        "start_gap_m": 20.0,
        "speed_mps": 22.0,
    },
    "forcing": {
        "scheduled_hazard": {
            "lane": 3,
            "x_m": 200.0,
            "speed_mps": 20.0,
            "brake_step": 2,
            "brake_speed_mps": 10.0,
        }
    },
}


class ObservationProbe:
    def __init__(self, values):
        object.__setattr__(self, "_values", values)

    def __getattribute__(self, name):
        if name in {"_values", "__getitem__"}:
            return object.__getattribute__(self, name)
        raise AssertionError(f"competent controller read observation.{name}")

    def __getitem__(self, key):
        return object.__getattribute__(self, "_values")[key]


def _make_env(profile, seed=100):
    env = HighwayV2XEnv(render_mode=None, obs_layout="contract56", scenario=profile)
    observations, _ = env.reset(seed=seed)
    return env, observations


def _background(env):
    native = env.env.unwrapped
    controlled = set(native.controlled_vehicles)
    return [vehicle for vehicle in native.road.vehicles if vehicle not in controlled]


def test_staggered_cav_starts_and_role_parsing():
    profile = parse_profile(VALID_PROFILE)
    assert profile.cav_x_m == (80.0, 100.0, 120.0, 140.0)
    assert len(profile.forcing.roles) == 1
    hazard = profile.forcing.scheduled_hazard
    assert hazard.brake_step == 2
    assert hazard.brake_speed_mps == 10.0


@pytest.mark.parametrize("profile_id", NEW_PROFILES)
def test_new_profile_contract_and_counts(profile_id):
    profile = load_profiles()[profile_id]
    env, observations = _make_env(profile_id)
    try:
        native = env.env.unwrapped
        assert env.agents == ["cav_0", "cav_1", "cav_2", "cav_3"]
        assert env.n_actions == N_ACTIONS == 6
        assert len(native.controlled_vehicles) == 4
        assert len(_background(env)) == profile.background_vehicles >= 12
        assert len(native._scenario_roles) == len(profile.forcing.roles)
        assert len(set(profile.cav_x_m)) == 4
        assert not any(vehicle.crashed for vehicle in native.road.vehicles)
        for observation in observations.values():
            assert observation.shape == (OBS_DIM,) == (56,)
            rows = observation[EGO + ROUTE : EGO + ROUTE + CAND].reshape(
                M_CANDIDATES, CAND_FEATURES
            )
            assert rows.shape == (M_CANDIDATES, 6)
            assert np.any(rows[:, 5])
        masks = env.action_masks()
        for mask in masks.values():
            assert mask.shape == (6,)
            assert mask.dtype == bool
    finally:
        env.close()


@pytest.mark.parametrize("profile_id", NEW_PROFILES)
def test_new_profile_seed_reproducibility(profile_id):
    env_a, obs_a = _make_env(profile_id, seed=100)
    env_b, obs_b = _make_env(profile_id, seed=100)
    env_c, obs_c = _make_env(profile_id, seed=101)
    try:
        for agent_id in env_a.agents:
            assert np.array_equal(obs_a[agent_id], obs_b[agent_id])
            assert not np.array_equal(obs_a[agent_id], obs_c[agent_id])
        assert np.array_equal(env_a.state(), env_b.state())
    finally:
        env_a.close()
        env_b.close()
        env_c.close()


def test_scheduled_hazard_fires_at_configured_step():
    env, _ = _make_env(VALID_PROFILE)
    try:
        native = env.env.unwrapped
        hazard = native._scenario_roles["scheduled_hazard"]
        assert hazard.target_speed == pytest.approx(20.0)
        actions = {agent_id: 0 for agent_id in env.agents}
        env.step(actions)
        assert hazard.target_speed == pytest.approx(20.0)
        env.step(actions)
        assert hazard.target_speed == pytest.approx(10.0)
    finally:
        env.close()


def test_competent_controller_only_reads_observation_and_mask():
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[0] = 0.7
    start = EGO + ROUTE
    observation[start + 5] = 1.0
    observation[start + 6] = 20.0 / 50.0
    observation[start + 8] = -12.0 / 30.0
    mask = np.ones(6, dtype=bool)
    action = competent_action(ObservationProbe(observation), mask)
    assert action in {0, 1, 2, 3, 4, 5}
