import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


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
