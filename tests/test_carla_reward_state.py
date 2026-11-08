"""Exercise the real CARLA adapter with a small in-memory CARLA API double."""

from __future__ import annotations

import importlib
import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def distance(self, other):
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


class FakeWaypoint:
    road_id = 1
    lane_id = 1
    lane_width = 3.5
    lane_type = 1
    is_junction = False

    def __init__(self, location):
        self.transform = FakeTransform(location)

    def next(self, distance):
        location = self.transform.location
        return [FakeWaypoint(FakeLocation(location.x + distance, location.y, location.z))]

    def get_left_lane(self):
        return None

    def get_right_lane(self):
        return None


class FakeTransform:
    def __init__(self, location=None):
        self.location = location or FakeLocation()
        self.rotation = SimpleNamespace(yaw=0.0)

    def get_right_vector(self):
        return SimpleNamespace(x=0.0, y=1.0, z=0.0)


class FakeVehicle:
    def __init__(self, actor_id, transform):
        self.id = actor_id
        self.transform = FakeTransform(
            FakeLocation(transform.location.x, transform.location.y, transform.location.z)
        )
        self.velocity = FakeVector()
        self.acceleration = FakeVector()
        self.control = SimpleNamespace(throttle=0.0, brake=0.0, hand_brake=False)

    def set_autopilot(self, *_args):
        pass

    def get_location(self):
        return self.transform.location

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return self.velocity

    def get_acceleration(self):
        return self.acceleration

    def get_speed_limit(self):
        return 50.0

    def is_at_traffic_light(self):
        return False

    def get_traffic_light(self):
        return None

    def get_traffic_light_state(self):
        return 0

    def get_control(self):
        return self.control

    def apply_control(self, control):
        self.control = control

    def set_simulate_physics(self, _enabled):
        pass


class FakeSensor:
    def listen(self, _callback):
        pass

    def stop(self):
        pass

    def destroy(self):
        pass


class FakeActors(list):
    def filter(self, pattern):
        if pattern == "vehicle.*":
            return [actor for actor in self if isinstance(actor, FakeVehicle)]
        return []


class FakeWorld:
    def __init__(self):
        self.settings = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None)
        self.actors = FakeActors()
        self.spawn_points = [
            FakeTransform(FakeLocation(0.0, 0.0)),
            FakeTransform(FakeLocation(100.0, 0.0)),
        ]
        self.next_actor_id = 1

    def get_settings(self):
        return self.settings

    def apply_settings(self, settings):
        self.settings = settings

    def tick(self):
        return 1

    def get_map(self):
        return self

    @property
    def name(self):
        return "Town03"

    def get_spawn_points(self):
        return list(self.spawn_points)

    def get_waypoint(self, location, **_kwargs):
        return FakeWaypoint(FakeLocation(location.x, location.y, location.z))

    def get_blueprint_library(self):
        return FakeBlueprintLibrary()

    def try_spawn_actor(self, _blueprint, transform):
        vehicle = FakeVehicle(self.next_actor_id, transform)
        self.next_actor_id += 1
        self.actors.append(vehicle)
        return vehicle

    def spawn_actor(self, _blueprint, _transform, attach_to=None):
        sensor = FakeSensor()
        self.actors.append(sensor)
        return sensor

    def get_actors(self):
        return self.actors


class FakeBlueprintLibrary:
    def filter(self, _pattern):
        return [object()]

    def find(self, _name):
        return object()


class FakeTrafficManager:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class FakeClient:
    def __init__(self, _host, _port):
        self.world = FakeWorld()

    def set_timeout(self, _timeout):
        pass

    def get_world(self):
        return self.world

    def get_trafficmanager(self, _port):
        return FakeTrafficManager()

    def apply_batch(self, _commands):
        return []


def test_real_carla_env_reset_and_two_steps_initialize_reward_state(monkeypatch):
    previous_env_module = sys.modules.pop("envs.carla_env", None)
    fake_carla = ModuleType("carla")
    fake_carla.Client = FakeClient
    fake_carla.Location = FakeLocation
    fake_carla.Transform = FakeTransform
    fake_carla.LaneType = SimpleNamespace(Driving=1)
    fake_carla.TrafficLightState = SimpleNamespace(Red=1, Yellow=2)
    fake_carla.command = SimpleNamespace(DestroyActor=lambda actor_id: actor_id)
    monkeypatch.setitem(sys.modules, "carla", fake_carla)
    carla_env = importlib.import_module("envs.carla_env")
    env = carla_env.CarlaMultiAgentEnv(
        town="Town03", n_agents=2, reset_traffic_lights=False
    )
    try:
        _, _ = env.reset(seed=3)
        assert set(env.previous_route_arc) == set(env.agents)
        assert set(env.previous_longitudinal_acceleration) == set(env.agents)

        for _ in range(2):
            _, rewards, _, _, info = env.step({agent_id: carla_env.KEEP for agent_id in env.agents})
            assert all(isinstance(value, float) and math.isfinite(value) for value in rewards.values())
            assert isinstance(info["team_reward"], float)
            assert math.isfinite(info["team_reward"])
            for agent_id in env.agents:
                assert isinstance(info["per_agent"][agent_id]["reward_terms"], dict)
                assert math.isfinite(info["per_agent"][agent_id]["a_long"])
                assert math.isfinite(info["per_agent"][agent_id]["jerk"])
                assert all(
                    isinstance(value, float) and math.isfinite(value)
                    for value in info["per_agent"][agent_id]["reward_terms"].values()
                )
        assert all(isinstance(value, np.ndarray) for value in env._get_obs().values())
    finally:
        env.close()
        sys.modules.pop("envs.carla_env", None)
        if previous_env_module is not None:
            sys.modules["envs.carla_env"] = previous_env_module
