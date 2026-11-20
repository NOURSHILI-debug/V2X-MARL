from __future__ import annotations

from scripts.milestone9.validate_pair import DEFAULT_TOLERANCES, pair_is_valid_for_analysis, validate_pair


def _episode(strategy="nearest_k", **overrides):
    base = {
        "episode_id": f"pair__{strategy}",
        "pair_id": "pair1",
        "strategy": strategy,
        "scenario_seed": 7,
        "episode_index": 0,
        "execution_status": "success",
        "config": {
            "environment": {
                "town": "Town03",
                "n_agents": 2,
                "max_steps": 300,
                "spawn_mode": "cluster",
                "spawn_cluster_max_distance_m": 45.0,
                "candidate_source": "v2x",
            },
            "v2x": {"communication_range_m": 50.0, "packet_loss": 0.0, "delay_steps": 0},
            "controller": {"module": "controllers.observation_heuristic", "class": "ObservationHeuristicController", "parameters": {}},
        },
        "initial_scenario": {
            "agents": ["cav_0", "cav_1"],
            "spawn_mode": "cluster",
            "spawn_cluster_max_distance": 45.0,
            "route_applied": True,
            "world": {
                "map_name": "Town03",
                "world_settings": {"synchronous_mode": True, "fixed_delta_seconds": 0.05},
                "weather": {},
            },
            "controlled_agents": {
                "cav_0": {"experiment_agent_id": "cav_0", "route_status": {"route_applied": True}},
                "cav_1": {"experiment_agent_id": "cav_1", "route_status": {"route_applied": True}},
            },
            "background_and_traffic_lights": {
                "vehicles": [],
                "traffic_lights": [],
                "managed_vehicle_actor_ids_diagnostic": [],
            },
            "initial_state": {
                "cav_0": {
                    "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                    "orientation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0},
                    "speed_mps": 0.5,
                    "destination": {"x": 10.0, "y": 2.0, "z": 0.0},
                    "route": [{"x": 1.0, "y": 2.0, "z": 0.0}, {"x": 10.0, "y": 2.0, "z": 0.0}],
                },
                "cav_1": {
                    "position": {"x": 3.0, "y": 2.0, "z": 0.0},
                    "orientation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0},
                    "speed_mps": 0.4,
                    "destination": {"x": 12.0, "y": 2.0, "z": 0.0},
                    "route": [{"x": 3.0, "y": 2.0, "z": 0.0}, {"x": 12.0, "y": 2.0, "z": 0.0}],
                },
            },
        },
    }
    base.update(overrides)
    return base


def test_equivalent_pair_is_valid():
    a = _episode("nearest_k")
    b = _episode("random_k")
    report = validate_pair(a, b)
    assert report.valid is True
    assert pair_is_valid_for_analysis(report) is True
    assert report.issues == []


def test_float_tolerance_allows_small_differences():
    a = _episode("nearest_k")
    b = _episode("random_k")
    b["initial_scenario"]["initial_state"]["cav_0"]["position"]["x"] += 1e-6
    b["initial_scenario"]["initial_state"]["cav_0"]["orientation"]["yaw"] += 1e-6
    report = validate_pair(a, b, tolerances=DEFAULT_TOLERANCES)
    assert report.valid is True


def test_initial_scenario_mismatch_detected_and_never_silent():
    a = _episode("nearest_k")
    b = _episode("random_k")
    b["initial_scenario"]["initial_state"]["cav_0"]["position"]["x"] = 999.0
    report = validate_pair(a, b)
    assert report.valid is False
    assert pair_is_valid_for_analysis(report) is False
    assert any(i.category == "initial_scenario_mismatch" for i in report.issues)
    assert "initial_state.cav_0.position" in {i.field for i in report.issues}


def test_config_and_identity_mismatches_detected():
    a = _episode("nearest_k")
    b = _episode("random_k", scenario_seed=17)
    report = validate_pair(a, b)
    assert report.valid is False
    assert any(i.field == "identity.scenario_seed" for i in report.issues)

    b2 = _episode("random_k")
    b2["config"]["v2x"]["packet_loss"] = 0.1
    report2 = validate_pair(a, b2)
    assert report2.valid is False


def test_failed_episode_invalidates_pair_as_infrastructure_failure():
    a = _episode("nearest_k")
    b = _episode("random_k", execution_status="failed", failure_reason="CARLA disconnected")
    report = validate_pair(a, b)
    assert report.valid is False
    assert any(i.category == "infrastructure_failure" for i in report.issues)
    assert pair_is_valid_for_analysis(report) is False


def test_later_trajectory_not_required_to_match():
    a = _episode("nearest_k")
    b = _episode("random_k")
    # trajectory-like fields outside initial_scenario/config checks should not invalidate
    a["final_positions"] = {"cav_0": {"x": 1}}
    b["final_positions"] = {"cav_0": {"x": 100}}
    report = validate_pair(a, b)
    assert report.valid is True
    assert "later_trajectory" in report.checked_fields


def test_route_validity_both_applied_passes():
    a = _episode("nearest_k")
    b = _episode("random_k")
    report = validate_pair(a, b)
    assert report.valid is True
    assert "route_applied.a" in report.checked_fields
    assert "route_applied.b" in report.checked_fields
    assert not any(i.field.startswith("route_applied.") and i.message for i in report.issues)


def test_route_validity_first_false_invalidates_pair():
    a = _episode("nearest_k")
    b = _episode("random_k")
    a["initial_scenario"]["route_applied"] = False
    report = validate_pair(a, b)
    assert report.valid is False
    assert pair_is_valid_for_analysis(report) is False
    assert any(i.field == "route_applied.a" and i.category == "infrastructure_failure" for i in report.issues)


def test_route_validity_second_false_invalidates_pair():
    a = _episode("nearest_k")
    b = _episode("random_k")
    b["initial_scenario"]["route_applied"] = False
    report = validate_pair(a, b)
    assert report.valid is False
    assert any(i.field == "route_applied.b" and i.category == "infrastructure_failure" for i in report.issues)


def test_route_validity_both_false_invalidates_pair():
    a = _episode("nearest_k")
    b = _episode("random_k")
    a["initial_scenario"]["route_applied"] = False
    b["initial_scenario"]["route_applied"] = False
    report = validate_pair(a, b)
    assert report.valid is False
    assert any(i.field == "route_applied.a" and i.category == "infrastructure_failure" for i in report.issues)
    assert any(i.field == "route_applied.b" and i.category == "infrastructure_failure" for i in report.issues)


def test_route_validity_missing_invalidates_pair():
    a = _episode("nearest_k")
    b = _episode("random_k")
    del a["initial_scenario"]["route_applied"]
    report = validate_pair(a, b)
    assert report.valid is False
    assert any(i.field == "route_applied.a" and "missing" in i.message.lower() for i in report.issues)


def test_route_validity_none_invalidates_pair():
    a = _episode("nearest_k")
    b = _episode("random_k")
    b["initial_scenario"]["route_applied"] = None
    report = validate_pair(a, b)
    assert report.valid is False
    assert any(i.field == "route_applied.b" and i.category == "infrastructure_failure" for i in report.issues)


def test_route_validity_malformed_string_invalidates_pair():
    a = _episode("nearest_k")
    b = _episode("random_k")
    a["initial_scenario"]["route_applied"] = "true"
    report = validate_pair(a, b)
    assert report.valid is False
    assert any(i.field == "route_applied.a" and "boolean" in i.message.lower() for i in report.issues)
