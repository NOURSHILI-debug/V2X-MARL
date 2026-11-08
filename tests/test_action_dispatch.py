import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import numpy as np


@pytest.mark.parametrize("action", range(6))
def test_apply_action_with_fake_carla(action, monkeypatch):
    previous_module = sys.modules.pop("envs.carla_env", None)
    fake_carla = ModuleType("carla")

    try:
        monkeypatch.setitem(sys.modules, "carla", fake_carla)
        carla_env = importlib.import_module("envs.carla_env")

        calls = []

        class FakeVehicle:
            def get_speed_limit(self):
                return 50.0

        class FakeTrafficManager:
            synchronous_modes = []

            def vehicle_percentage_speed_difference(self, _vehicle, percentage):
                calls.append(("speed", percentage))

            def force_lane_change(self, _vehicle, to_right):
                calls.append(("lane_change", to_right))

            def set_synchronous_mode(self, enabled):
                self.synchronous_modes.append(enabled)

        env = object.__new__(carla_env.CarlaMultiAgentEnv)
        env.vehicles = {"cav_0": FakeVehicle()}
        env.target_speed = {"cav_0": 9.0}
        env.tm = FakeTrafficManager()

        env._apply_action("cav_0", action)

        if action == carla_env.BRAKE:
            assert env.target_speed["cav_0"] == 9.0
            assert calls == [("speed", 100.0)]
        else:
            expected_speed = {0: 9.0, 1: 11.0, 2: 7.0, 4: 9.0, 5: 9.0}[action]
            assert env.target_speed["cav_0"] == expected_speed
            speed_call_index = 1 if action in (carla_env.LANE_LEFT, carla_env.LANE_RIGHT) else 0
            assert calls[speed_call_index][0] == "speed"
            assert calls[speed_call_index][1] == pytest.approx(
                (1.0 - expected_speed / (50.0 / 3.6)) * 100.0
            )
            if action in (carla_env.LANE_LEFT, carla_env.LANE_RIGHT):
                assert calls[0] == (
                    "lane_change",
                    action == carla_env.LANE_RIGHT,
                )
            else:
                assert len(calls) == 1
    finally:
        sys.modules.pop("envs.carla_env", None)
        if previous_module is not None:
            sys.modules["envs.carla_env"] = previous_module


def test_close_restores_async_modes_before_destroying_actors(monkeypatch):
    previous_module = sys.modules.pop("envs.carla_env", None)
    fake_carla = ModuleType("carla")

    try:
        monkeypatch.setitem(sys.modules, "carla", fake_carla)
        carla_env = importlib.import_module("envs.carla_env")
        calls = []

        class FakeSettings:
            synchronous_mode = True
            fixed_delta_seconds = 0.05

        class FakeWorld:
            def get_settings(self):
                return FakeSettings()

            def apply_settings(self, settings):
                calls.append(
                    ("world_settings", settings.synchronous_mode, settings.fixed_delta_seconds)
                )

            def tick(self):
                calls.append(("world_tick",))

        class FakeClient:
            def apply_batch(self, commands):
                calls.append(("destroy_actors", commands))

        class FakeTrafficManager:
            def set_synchronous_mode(self, enabled):
                calls.append(("traffic_manager_sync", enabled))

        class FakeVehicle:
            id = 42

        fake_carla.command = SimpleNamespace(
            DestroyActor=lambda actor_id: ("destroy", actor_id)
        )
        env = object.__new__(carla_env.CarlaMultiAgentEnv)
        env.world = FakeWorld()
        env.tm = FakeTrafficManager()
        env.client = FakeClient()
        env.vehicles = {"cav_0": FakeVehicle()}
        env.target_speed = {"cav_0": 9.0}
        env.route = {}
        env.goal = {}
        env.reached = {}
        env.stuck_steps = {}
        env.off_road_steps = {}
        env.route_applied = {}
        env.finished = {}
        env.monitor = None

        env.close()

        assert calls[:3] == [
            ("world_settings", False, None),
            ("traffic_manager_sync", False),
            ("destroy_actors", [("destroy", 42)]),
        ]
    finally:
        sys.modules.pop("envs.carla_env", None)
        if previous_module is not None:
            sys.modules["envs.carla_env"] = previous_module


def _step_env(carla_env, masks=None):
    env = object.__new__(carla_env.CarlaMultiAgentEnv)
    env.agents = ["cav_0", "cav_1"]
    env.reached = dict.fromkeys(env.agents, False)
    fake_vehicle = SimpleNamespace(
        get_location=lambda: SimpleNamespace(),
        get_acceleration=lambda: SimpleNamespace(length=lambda: 0.0),
        get_transform=lambda: SimpleNamespace(rotation=SimpleNamespace(yaw=0.0)),
    )
    env.vehicles = {agent: fake_vehicle for agent in env.agents}
    env.action_masks = lambda: masks or {
        "cav_0": np.array([1, 1, 1, 1, 0, 1], dtype=bool),
        "cav_1": np.array([1, 1, 1, 1, 1, 1], dtype=bool),
    }
    env._apply_action = lambda agent, action: env.dispatched.append((agent, action))
    env.dispatched = []
    env.world = SimpleNamespace(tick=lambda: None, get_map=lambda: SimpleNamespace(
        get_waypoint=lambda *args, **kwargs: object()
    ))
    env.monitor = SimpleNamespace(is_collided=lambda agent: False)
    env._update_goal_and_stuck = lambda: None
    env._measure_min_ttc = lambda: 10.0
    env.off_road_steps = dict.fromkeys(env.agents, 0)
    env.stuck_steps = dict.fromkeys(env.agents, 0)
    env.step_count = 0
    env.max_steps = 300
    env.route_applied = dict.fromkeys(env.agents, True)
    env.finished = dict.fromkeys(env.agents, False)
    env.stop_reason = dict.fromkeys(env.agents, "none")
    env._get_obs = lambda: {agent: np.zeros(56, dtype=np.float32) for agent in env.agents}
    env._step_vehicle_location = lambda vehicle: SimpleNamespace()
    env.town, env.seed = "Town03", 0
    env.min_ttc, env.jerk_samples = 10.0, []
    env.previous_acceleration = dict.fromkeys(env.agents)
    return env


def test_step_invalid_lane_change_becomes_keep_and_is_counted(monkeypatch):
    previous_module = sys.modules.pop("envs.carla_env", None)
    try:
        fake_carla = ModuleType("carla")
        fake_carla.LaneType = SimpleNamespace(Driving=1)
        monkeypatch.setitem(sys.modules, "carla", fake_carla)
        carla_env = importlib.import_module("envs.carla_env")
        env = _step_env(carla_env)
        env.max_steps = 1
        _, _, _, _, info = env.step({"cav_0": carla_env.LANE_LEFT, "cav_1": carla_env.KEEP})
        assert env.dispatched == [("cav_0", carla_env.KEEP), ("cav_1", carla_env.KEEP)]
        assert info["invalid_action"] == 1
    finally:
        sys.modules.pop("envs.carla_env", None)
        if previous_module is not None:
            sys.modules["envs.carla_env"] = previous_module


@pytest.mark.parametrize("actions", [
    {"cav_0": 0},
    {"cav_0": 0, "wrong": 1},
    {"cav_0": 6, "cav_1": 0},
    {"cav_0": True, "cav_1": 0},
])
def test_step_rejects_bad_agent_keys_or_action_ids(actions, monkeypatch):
    previous_module = sys.modules.pop("envs.carla_env", None)
    try:
        fake_carla = ModuleType("carla")
        fake_carla.LaneType = SimpleNamespace(Driving=1)
        monkeypatch.setitem(sys.modules, "carla", fake_carla)
        carla_env = importlib.import_module("envs.carla_env")
        env = _step_env(carla_env)
        with pytest.raises(ValueError):
            env.step(actions)
    finally:
        sys.modules.pop("envs.carla_env", None)
        if previous_module is not None:
            sys.modules["envs.carla_env"] = previous_module
