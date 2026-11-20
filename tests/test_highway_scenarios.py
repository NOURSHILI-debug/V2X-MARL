import json
from pathlib import Path

import numpy as np
import pytest

from envs.highway_env import HighwayV2XEnv
from envs.scenarios.highway_scenarios import (
    load_profiles,
    parse_profile,
    resolve_scenario,
)


BASELINE_PATH = Path(__file__).with_name("data") / "highway_no_scenario_baseline.json"
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
    bad_cav = dict(VALID_PROFILE, cav={"lanes": [0, 1, 2], "start_x_m": [1, 2, 3, 4], "speed_mps": 20})
    with pytest.raises(ValueError, match="four values"):
        parse_profile(bad_cav)


def test_scenario_config_file_loads_without_profiles():
    assert load_profiles() == {}
