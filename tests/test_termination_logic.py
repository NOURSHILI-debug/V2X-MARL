import pytest
import importlib
import sys
from types import ModuleType, SimpleNamespace

from envs.termination import (
    build_step_info,
    evaluate_termination,
    is_goal_reached,
    update_off_road_steps,
)


AGENTS = ("cav_0", "cav_1")


def _decision(
    *,
    collided=None,
    reached=None,
    off_road_steps=None,
    stuck_steps=None,
    step_count=1,
    max_steps=300,
):
    return evaluate_termination(
        AGENTS,
        collided or dict.fromkeys(AGENTS, False),
        reached or dict.fromkeys(AGENTS, False),
        off_road_steps or dict.fromkeys(AGENTS, 0),
        stuck_steps or dict.fromkeys(AGENTS, 0),
        step_count=step_count,
        max_steps=max_steps,
    )


def test_off_road_counter_requires_consecutive_steps():
    assert update_off_road_steps(0, True) == 1
    assert update_off_road_steps(9, True) == 10
    assert update_off_road_steps(9, False) == 0


@pytest.mark.parametrize(
    ("inputs", "reason"),
    [
        (
            {
                "collided": {"cav_0": True, "cav_1": False},
                "reached": {"cav_0": True, "cav_1": True},
                "off_road_steps": {"cav_0": 51, "cav_1": 0},
                "stuck_steps": {"cav_0": 51, "cav_1": 0},
                "step_count": 300,
            },
            "collision",
        ),
        (
            {
                "reached": {"cav_0": True, "cav_1": True},
                "off_road_steps": {"cav_0": 51, "cav_1": 0},
                "stuck_steps": {"cav_0": 51, "cav_1": 0},
                "step_count": 300,
            },
            "goal",
        ),
        (
            {
                "off_road_steps": {"cav_0": 51, "cav_1": 0},
                "stuck_steps": {"cav_0": 51, "cav_1": 0},
                "step_count": 300,
            },
            "off_road",
        ),
        ({"stuck_steps": {"cav_0": 51, "cav_1": 51}, "step_count": 300}, "stuck"),
        ({"step_count": 300}, "timeout"),
    ],
)
def test_termination_priority(inputs, reason):
    result = _decision(**inputs)
    assert result.reason == reason
    assert result.terminated is (reason != "timeout")
    assert result.truncated is (reason == "timeout")
    assert result.success is (reason == "goal")


def test_termination_continues_before_any_condition():
    result = _decision()

    assert result.reason is None
    assert not result.terminated
    assert not result.truncated
    assert not result.success


def test_stuck_requires_more_than_fifty_slow_steps():
    result = _decision(stuck_steps={"cav_0": 50, "cav_1": 0})

    assert result.reason is None
    assert result.stuck_by_agent == {"cav_0": False, "cav_1": False}

    result = _decision(stuck_steps={"cav_0": 51, "cav_1": 51}, step_count=51)
    assert result.reason == "stuck"
    one_stuck = _decision(
        stuck_steps={"cav_0": 51, "cav_1": 0}, step_count=51
    )
    assert one_stuck.reason is None


def test_off_road_requires_more_than_fifty_consecutive_steps():
    assert _decision(off_road_steps={"cav_0": 50, "cav_1": 0}).reason is None
    assert _decision(off_road_steps={"cav_0": 51, "cav_1": 0}).reason == "off_road"


def test_stuck_detection_waits_through_grace_period():
    result = evaluate_termination(
        AGENTS,
        dict.fromkeys(AGENTS, False),
        dict.fromkeys(AGENTS, False),
        dict.fromkeys(AGENTS, 0),
        {"cav_0": 51, "cav_1": 51},
        step_count=24,
        max_steps=300,
        stuck_grace_steps=25,
    )

    assert result.reason is None
    assert result.stuck_by_agent == {"cav_0": False, "cav_1": False}

    result = evaluate_termination(
        AGENTS,
        dict.fromkeys(AGENTS, False),
        dict.fromkeys(AGENTS, False),
        dict.fromkeys(AGENTS, 0),
        {"cav_0": 51, "cav_1": 51},
        step_count=25,
        max_steps=300,
        stuck_grace_steps=25,
    )
    assert result.reason == "stuck"


def test_goal_radius_includes_fake_vehicle_inside_radius():
    class FakeVehicle:
        def __init__(self, distance):
            self._distance = distance

        def get_location(self):
            return self

        def distance(self, _goal):
            return self._distance

    assert is_goal_reached(FakeVehicle(0.62), object(), radius_m=6.0)
    assert is_goal_reached(FakeVehicle(5.2), object(), radius_m=6.0)
    assert is_goal_reached(FakeVehicle(6.0), object(), radius_m=6.0)


def test_goal_radius_flags_inside_and_excludes_just_outside():
    class FakeVehicle:
        def __init__(self, distance):
            self._distance = distance

        def get_location(self):
            return self

        def distance(self, _goal):
            return self._distance

    radius = 8.0
    assert is_goal_reached(FakeVehicle(radius - 0.01), object(), radius)
    assert not is_goal_reached(FakeVehicle(radius + 0.01), object(), radius)


def test_arrival_step_is_exposed_once_in_per_agent_info():
    info = build_step_info(
        ["cav_0"], {"cav_0": False}, {"cav_0": True},
        {"cav_0": False}, {"cav_0": False}, {"cav_0": True},
        steps=7, episode_ended=False, success=False, town="Town03", seed=0,
        arrival_step={"cav_0": 4},
    )
    assert info["per_agent"]["cav_0"]["arrival_step"] == 4
    later = build_step_info(
        ["cav_0"], {"cav_0": False}, {"cav_0": True},
        {"cav_0": False}, {"cav_0": False}, {"cav_0": True},
        steps=8, episode_ended=False, success=False, town="Town03", seed=0,
        arrival_step={"cav_0": 4},
    )
    assert later["per_agent"]["cav_0"]["arrival_step"] == 4


def test_goal_is_15m_before_tm_path_endpoint():
    previous_module = sys.modules.pop("envs.carla_env", None)
    try:
        fake_carla = ModuleType("carla")
        fake_carla.Location = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
        fake_carla.LaneType = SimpleNamespace(Driving=1)
        previous_carla = sys.modules.get("carla")
        sys.modules["carla"] = fake_carla
        carla_env = importlib.import_module("envs.carla_env")
    finally:
        if previous_module is not None:
            sys.modules["envs.carla_env"] = previous_module
        if "previous_carla" in locals():
            if previous_carla is None:
                sys.modules.pop("carla", None)
            else:
                sys.modules["carla"] = previous_carla

    class FakeLocation:
        def __init__(self, x, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z

        def distance(self, other):
            return ((self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2)**0.5

    carla_env.carla.Location = lambda x, y, z: FakeLocation(x, y, z)
    path = [FakeLocation(0), FakeLocation(10), FakeLocation(20), FakeLocation(30)]
    goal = carla_env.CarlaMultiAgentEnv._goal_before_route_end(path)
    assert carla_env.GOAL_MARGIN == 15.0
    assert abs(goal.distance(path[-1]) - carla_env.GOAL_MARGIN) < 1e-6


def test_termination_requires_matching_agent_keys():
    with pytest.raises(ValueError, match="exactly the configured agents"):
        evaluate_termination(
            AGENTS,
            {"cav_0": False},
            dict.fromkeys(AGENTS, 0),
            dict.fromkeys(AGENTS, 0),
            dict.fromkeys(AGENTS, False),
            step_count=1,
            max_steps=300,
        )
