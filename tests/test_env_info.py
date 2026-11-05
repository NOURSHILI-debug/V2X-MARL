from types import SimpleNamespace

from envs.termination import build_step_info, is_off_road


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
    )

    assert info["per_agent"] == {
        "cav_0": {
            "collision": True,
            "goal_reached": False,
            "stuck": False,
            "off_road": True,
            "route_applied": True,
        },
        "cav_1": {
            "collision": False,
            "goal_reached": True,
            "stuck": False,
            "off_road": False,
            "route_applied": False,
        },
    }
    assert all(
        isinstance(value, bool)
        for agent_info in info["per_agent"].values()
        for value in agent_info.values()
    )
    assert info["collision"] is True
    assert info["goal_reached"] is False
    assert info["stuck"] is False
    assert info["off_road"] is True
    assert info["route_applied"] is False
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
