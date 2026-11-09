import importlib
import random
import sys
from types import ModuleType, SimpleNamespace


def _load_carla_env(monkeypatch):
    previous = sys.modules.pop("envs.carla_env", None)
    fake_carla = ModuleType("carla")
    fake_carla.Location = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
    fake_carla.LaneType = SimpleNamespace(Driving=1)
    monkeypatch.setitem(sys.modules, "carla", fake_carla)
    module = importlib.import_module("envs.carla_env")
    return module, previous


def test_route_start_validation_accepts_near_start_and_rejects_bad_start(monkeypatch):
    carla_env, previous = _load_carla_env(monkeypatch)
    try:
        class Point:
            def __init__(self, x):
                self.x, self.y, self.z = x, 0.0, 0.0

            def distance(self, other):
                return abs(self.x - other.x)

        route = [SimpleNamespace(transform=SimpleNamespace(location=Point(x))) for x in (0, 5, 10, 15)]
        assert carla_env.CarlaMultiAgentEnv._route_start_is_valid(Point(2), route)
        assert not carla_env.CarlaMultiAgentEnv._route_start_is_valid(Point(30), route)
    finally:
        sys.modules.pop("envs.carla_env", None)
        if previous is not None:
            sys.modules["envs.carla_env"] = previous


def test_per_agent_seeded_routes_can_diverge(monkeypatch):
    carla_env, previous = _load_carla_env(monkeypatch)
    try:
        class Waypoint:
            def __init__(self, x, branch):
                self.x, self.branch = x, branch

            def next(self, _distance):
                return [Waypoint(self.x + 5, branch) for branch in range(4)]

        env = object.__new__(carla_env.CarlaMultiAgentEnv)
        env._plan_route = carla_env.CarlaMultiAgentEnv._plan_route.__get__(env)
        env.seed = 0
        first = env._plan_route(Waypoint(0, 0), random.Random(env.seed + 0))
        second = env._plan_route(Waypoint(0, 0), random.Random(env.seed + 1))
        assert tuple(wp.branch for wp in first) != tuple(wp.branch for wp in second)
    finally:
        sys.modules.pop("envs.carla_env", None)
        if previous is not None:
            sys.modules["envs.carla_env"] = previous


def test_traffic_light_cycle_applies_to_each_group_and_resets(monkeypatch):
    carla_env, previous = _load_carla_env(monkeypatch)
    try:
        calls = []

        class FakeLight:
            def __init__(self, actor_id):
                self.id = actor_id
                self.durations = {}

            def get_group_traffic_lights(self):
                return lights

            def set_green_time(self, value):
                calls.append((self.id, "green", value))
                self.durations["green"] = value

            def set_yellow_time(self, value):
                calls.append((self.id, "yellow", value))
                self.durations["yellow"] = value

            def set_red_time(self, value):
                calls.append((self.id, "red", value))
                self.durations["red"] = value

            def reset_group(self):
                calls.append((self.id, "reset_group"))

            def freeze(self, value):
                calls.append((self.id, "freeze", value))

            def get_green_time(self):
                return self.durations["green"]

            def get_yellow_time(self):
                return self.durations["yellow"]

            def get_red_time(self):
                return self.durations["red"]

        lights = [FakeLight(1), FakeLight(2)]
        env = object.__new__(carla_env.CarlaMultiAgentEnv)
        env.reset_traffic_lights = True
        env.light_cycle = True
        env.world = SimpleNamespace(
            get_actors=lambda: SimpleNamespace(
                filter=lambda _pattern: lights
            )
        )
        env._configure_traffic_lights()

        for light in lights:
            assert light.durations == {"green": 10.0, "yellow": 2.0, "red": 5.0}
            assert (light.id, "freeze", False) in calls
        assert sum(call[1] == "reset_group" for call in calls) == 1
        assert calls.index((1, "reset_group")) > max(
            i for i, call in enumerate(calls) if call[1] == "red"
        )
    finally:
        sys.modules.pop("envs.carla_env", None)
        if previous is not None:
            sys.modules["envs.carla_env"] = previous
