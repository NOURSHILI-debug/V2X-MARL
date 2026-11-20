from __future__ import annotations

import json

import pytest

from scripts.milestone9.config import (
    CONTROLLER_CLASS,
    CONTROLLER_MODULE,
    Milestone9Config,
    default_config,
    derive_selection_seed,
    derive_v2x_seed,
    episode_id,
    pair_id,
    seed_key,
    stable_seed,
)


def test_default_config_valid_and_serializable():
    cfg = default_config()
    assert cfg.environment.town == "Town03"
    assert cfg.environment.n_agents == 4
    assert cfg.environment.spawn_mode == "cluster"
    assert cfg.environment.spawn_cluster_max_distance_m == 45.0
    assert cfg.environment.candidate_source == "v2x"
    assert cfg.environment.candidate_radius_m == 50.0
    assert cfg.environment.selection_k == 2
    assert cfg.environment.max_steps == 300
    assert cfg.environment.sim_hz == 20
    assert cfg.environment.decision_hz == 5
    assert cfg.v2x.communication_range_m == 50.0
    assert cfg.v2x.packet_loss == 0.0
    assert cfg.v2x.delay_steps == 0
    assert cfg.v2x.broadcasters == "cavs_and_background"
    assert cfg.controller.module == CONTROLLER_MODULE
    assert cfg.controller.class_name == CONTROLLER_CLASS
    assert tuple(cfg.strategies) == ("nearest_k", "random_k")
    assert tuple(cfg.seeds.scenario_seeds) == (7, 17, 27, 37, 47)

    d = cfg.to_dict()
    json.dumps(d)  # serializable
    restored = Milestone9Config.from_dict(d)
    assert restored.to_dict() == d


@pytest.mark.parametrize(
    "mutator",
    [
        lambda d: d["environment"].update({"town": "Town01"}),
        lambda d: d["environment"].update({"n_agents": 3}),
        lambda d: d["environment"].update({"spawn_mode": "default"}),
        lambda d: d["environment"].update({"spawn_cluster_max_distance_m": 40.0}),
        lambda d: d["environment"].update({"candidate_source": "ground_truth"}),
        lambda d: d["environment"].update({"candidate_radius_m": 40.0}),
        lambda d: d["environment"].update({"selection_k": 3}),
        lambda d: d["environment"].update({"max_steps": 250}),
        lambda d: d["v2x"].update({"packet_loss": 0.1}),
        lambda d: d["v2x"].update({"delay_steps": 1}),
        lambda d: d["v2x"].update({"communication_range_m": 40.0}),
        lambda d: d["v2x"].update({"broadcasters": "cavs_only"}),
        lambda d: d.update({"strategies": ["nearest_k"]}),
        lambda d: d["controller"].update({"class_name": "Other"}),
        lambda d: d["seeds"].update({"scenario_seeds": [7, 7]}),
    ],
)
def test_config_validation_rejects_invalid_values(mutator):
    d = default_config().to_dict()
    mutator(d)
    with pytest.raises(ValueError):
        Milestone9Config.from_dict(d)


def test_seed_derivation_is_deterministic_and_separated():
    cfg = default_config()
    a = derive_selection_seed(cfg, 7, 0, "random_k")
    b = derive_selection_seed(cfg, 7, 0, "random_k")
    assert a == b
    assert derive_v2x_seed(cfg, 7, 0, "random_k") == derive_v2x_seed(cfg, 7, 0, "random_k")
    # separated subsystems / identities
    assert derive_selection_seed(cfg, 7, 0, "random_k") != derive_v2x_seed(cfg, 7, 0, "random_k")
    assert derive_selection_seed(cfg, 7, 0, "random_k") != derive_selection_seed(cfg, 7, 0, "nearest_k")
    assert derive_selection_seed(cfg, 7, 0, "random_k") != derive_selection_seed(cfg, 17, 0, "random_k")
    assert derive_selection_seed(cfg, 7, 0, "random_k") != derive_selection_seed(cfg, 7, 1, "random_k")


def test_explicit_seed_overrides():
    d = default_config().to_dict()
    key = seed_key(7, 0, "random_k")
    d["seeds"]["explicit_selection_seeds"][key] = 12345
    d["seeds"]["explicit_v2x_seeds"][key] = 54321
    cfg = Milestone9Config.from_dict(d)
    assert derive_selection_seed(cfg, 7, 0, "random_k") == 12345
    assert derive_v2x_seed(cfg, 7, 0, "random_k") == 54321


def test_pair_and_episode_identity():
    exp = "m9_test"
    assert pair_id(exp, 7, 0) == "m9_test__seed7__ep0"
    assert episode_id(exp, 7, 0, "nearest_k") == "m9_test__seed7__ep0__nearest_k"
    assert pair_id(exp, 17, 2) != pair_id(exp, 17, 1)
    assert stable_seed("a", 1) == stable_seed("a", 1)
    assert stable_seed("a", 1) != stable_seed("a", 2)
