from types import SimpleNamespace

import pytest

from envs.termination import build_step_info, evaluate_termination, is_off_road


def test_missing_unprojected_waypoint_is_off_road():
    assert is_off_road(None)
    assert not is_off_road(SimpleNamespace(lane_id=1))


def test_step_info_keeps_per_agent_status_nested_and_boolean():
    info = build_step_info(
        ["cav_0", "cav_1"],
        {"cav_0": True, "cav_1": False},
        {"cav_0": False, "cav_1": True},
        {"cav_0": False, "cav_1": False},
        {"cav_0": True, "cav_1": False},
        {"cav_0": True, "cav_1": False},
        steps=4,
        episode_ended=False,
        success=False,
        town="Town03",
        seed=7,
        stop_reason={"cav_0": "goal", "cav_1": "red_light"},
    )

    assert info["per_agent"] == {
        "cav_0": {
            "collision": True,
            "goal_reached": False,
            "stuck": False,
            "off_road": True,
            "route_applied": True,
            "stop_reason": "goal",
            "arrival_step": None,
        },
        "cav_1": {
            "collision": False,
            "goal_reached": True,
            "stuck": False,
            "off_road": False,
            "route_applied": False,
            "stop_reason": "red_light",
            "arrival_step": None,
        },
    }
    assert all(
        isinstance(value, bool)
        for agent_info in info["per_agent"].values()
        for key, value in agent_info.items()
        if key not in {"stop_reason", "arrival_step"}
    )
    assert info["collision"] is True
    assert info["goal_reached"] is False
    assert info["stuck"] is False
    assert info["off_road"] is True
    assert info["route_applied"] is False
    assert info["invalid_action"] == 0
    assert {"collision", "goal_reached", "steps"}.issubset(info)
    assert not {"success", "min_ttc", "jerk", "scenario_id"}.intersection(info)
    assert "town" not in info and "seed" not in info


def test_terminal_step_info_includes_reset_metadata():
    info = build_step_info(
        ["cav_0"],
        {"cav_0": False},
        {"cav_0": True},
        {"cav_0": False},
        {"cav_0": False},
        {"cav_0": True},
        steps=12,
        episode_ended=True,
        success=True,
        town="Town03",
        seed=9,
    )

    assert info["town"] == "Town03"
    assert info["seed"] == 9
    assert info["success"] is True
    assert info["min_ttc"] == 10.0
    assert info["jerk"] == 0.0
    assert info["scenario_id"] == "free_drive"
    assert {"collision", "goal_reached", "steps", "success", "min_ttc", "jerk", "scenario_id", "town", "seed"}.issubset(info)


@pytest.mark.parametrize(
    ("reason", "collided", "reached", "off_road_steps", "stuck_steps", "expected_flag"),
    [
        (
            "collision",
            {"cav_0": True, "cav_1": False},
            {"cav_0": True, "cav_1": True},
            {"cav_0": 51, "cav_1": 0},
            {"cav_0": 51, "cav_1": 51},
            "collision",
        ),
        (
            "goal",
            {"cav_0": False, "cav_1": False},
            {"cav_0": True, "cav_1": True},
            {"cav_0": 0, "cav_1": 0},
            {"cav_0": 0, "cav_1": 0},
            "goal_reached",
        ),
        (
            "off_road",
            {"cav_0": False, "cav_1": False},
            {"cav_0": False, "cav_1": False},
            {"cav_0": 51, "cav_1": 0},
            {"cav_0": 51, "cav_1": 51},
            "off_road",
        ),
        (
            "stuck",
            {"cav_0": False, "cav_1": False},
            {"cav_0": False, "cav_1": False},
            {"cav_0": 0, "cav_1": 0},
            {"cav_0": 51, "cav_1": 51},
            "stuck",
        ),
        (
            "timeout",
            {"cav_0": False, "cav_1": False},
            {"cav_0": False, "cav_1": False},
            {"cav_0": 0, "cav_1": 0},
            {"cav_0": 0, "cav_1": 0},
            None,
        ),
    ],
)
def test_terminal_path_sets_only_its_matching_info_flag(
    reason, collided, reached, off_road_steps, stuck_steps, expected_flag
):
    agents = ["cav_0", "cav_1"]
    result = evaluate_termination(
        agents,
        collided,
        reached,
        off_road_steps,
        stuck_steps,
        step_count=300,
        max_steps=300,
        stuck_grace_steps=25,
    )
    assert result.reason == reason
    info = build_step_info(
        agents,
        collided,
        reached,
        result.stuck_by_agent,
        {agent_id: bool(off_road_steps[agent_id]) for agent_id in agents},
        dict.fromkeys(agents, True),
        steps=300,
        episode_ended=result.terminated or result.truncated,
        success=result.success,
        town="Town03",
        seed=0,
        ending_reason=result.reason,
    )

    ending_flags = ("collision", "goal_reached", "off_road", "stuck")
    set_flags = [flag for flag in ending_flags if info[flag]]
    assert set_flags == ([expected_flag] if expected_flag else [])
    assert result.truncated is (reason == "timeout")
    if reason == "goal":
        assert info["goal_reached"] and info["success"]
    else:
        assert not info["success"]
