from __future__ import annotations

from copy import deepcopy

from scripts.milestone9.validate_pair import pair_is_valid_for_analysis, validate_pair


def _base_episode(strategy="nearest_k"):
    return {
        "episode_id": f"pair__{strategy}",
        "pair_id": "pair1",
        "strategy": strategy,
        "scenario_seed": 7,
        "episode_index": 0,
        "execution_status": "success",
        "lifecycle": {"cleanup_status": "success"},
        "persistence": {"status": "success", "artifacts_present": True},
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
                "weather": {"cloudiness": 0.0},
            },
            "controlled_agents": {
                "cav_0": {"experiment_agent_id": "cav_0", "vehicle_type": "vehicle.tesla.model3", "route_status": {"route_applied": True}},
                "cav_1": {"experiment_agent_id": "cav_1", "vehicle_type": "vehicle.tesla.model3", "route_status": {"route_applied": True}},
            },
            "background_and_traffic_lights": {
                "managed_vehicle_actor_ids_diagnostic": [101, 102],
                "vehicles": [{"managed": True, "type_id": "vehicle.tesla.model3", "carla_actor_id_diagnostic": 101}],
                "traffic_lights": [{"carla_actor_id_diagnostic": 201, "state": "Green", "timing": {"available": False}}],
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


def test_snapshot_required_sections_present_and_valid_pair():
    a = _base_episode("nearest_k")
    b = _base_episode("random_k")
    report = validate_pair(a, b)
    assert report.valid is True
    assert pair_is_valid_for_analysis(report) is True


def test_missing_required_world_snapshot_invalidates_even_if_both_missing():
    a = _base_episode("nearest_k")
    b = _base_episode("random_k")
    del a["initial_scenario"]["world"]
    del b["initial_scenario"]["world"]
    report = validate_pair(a, b)
    assert report.valid is False
    assert any(i.field == "snapshot.a.world" for i in report.issues)
    assert any(i.field == "snapshot.b.world" for i in report.issues)


def test_missing_required_world_setting_invalidates():
    a = _base_episode("nearest_k")
    b = _base_episode("random_k")
    del a["initial_scenario"]["world"]["world_settings"]["fixed_delta_seconds"]
    report = validate_pair(a, b)
    assert report.valid is False
    assert any("fixed_delta_seconds" in i.field for i in report.issues)


def test_missing_controlled_agents_snapshot_invalidates():
    a = _base_episode("nearest_k")
    b = _base_episode("random_k")
    del b["initial_scenario"]["controlled_agents"]
    report = validate_pair(a, b)
    assert report.valid is False
    assert any(i.field == "snapshot.b.controlled_agents" for i in report.issues)


def test_missing_background_and_traffic_lights_snapshot_invalidates():
    a = _base_episode("nearest_k")
    b = _base_episode("random_k")
    del a["initial_scenario"]["background_and_traffic_lights"]["traffic_lights"]
    report = validate_pair(a, b)
    assert report.valid is False
    assert any("traffic_lights" in i.field for i in report.issues)


def test_world_settings_mismatch_invalidates():
    a = _base_episode("nearest_k")
    b = _base_episode("random_k")
    b["initial_scenario"]["world"]["world_settings"]["fixed_delta_seconds"] = 0.1
    report = validate_pair(a, b)
    assert report.valid is False
    assert any(i.field == "snapshot.world.world_settings.fixed_delta_seconds" for i in report.issues)


def test_cleanup_failure_invalidates_pair():
    a = _base_episode("nearest_k")
    b = _base_episode("random_k")
    b["lifecycle"]["cleanup_status"] = "failed"
    b["lifecycle"]["cleanup_error"] = "destroy failed"
    report = validate_pair(a, b)
    assert report.valid is False
    assert any(i.field == "lifecycle.b.cleanup_status" and i.category == "infrastructure_failure" for i in report.issues)


def test_persistence_failure_invalidates_pair():
    a = _base_episode("nearest_k")
    b = _base_episode("random_k")
    a["persistence"] = {"status": "failed", "error": "disk full", "artifacts_present": False}
    report = validate_pair(a, b)
    assert report.valid is False
    assert any(i.field == "persistence.a.status" and i.category == "infrastructure_failure" for i in report.issues)
