from types import SimpleNamespace

import pytest

from envs.safety import CollisionMonitor


class FakeBlueprintLibrary:
    def __init__(self, blueprint_names=None):
        self.blueprints = blueprint_names or {}

    def find(self, name):
        return self.blueprints.get(name)


class FakeSensor:
    def __init__(self):
        self.callback = None
        self.stopped = False
        self.destroyed = False

    def listen(self, callback):
        self.callback = callback

    def stop(self):
        self.stopped = True

    def destroy(self):
        self.destroyed = True

    def emit(self, event):
        if self.callback is not None:
            self.callback(event)


class FakeWorld:
    def __init__(self, blueprint_names=None):
        self.blueprint_library = FakeBlueprintLibrary(blueprint_names)
        self.spawned = []

    def get_blueprint_library(self):
        return self.blueprint_library

    def spawn_actor(self, blueprint, transform=None, attach_to=None):
        sensor = FakeSensor()
        self.spawned.append({
            "blueprint": blueprint,
            "transform": transform,
            "attach_to": attach_to,
            "sensor": sensor,
        })
        return sensor


class FakeCollisionEvent:
    def __init__(self, other_actor_id, frame):
        self.other_actor = SimpleNamespace(id=other_actor_id)
        self.frame = frame


def test_collision_monitor_records_collision():
    world = FakeWorld({"sensor.other.collision": object()})
    agents = {"agent_0": object(), "agent_1": object()}
    monitor = CollisionMonitor(world, agents)

    monitor.sensors["agent_0"].emit(FakeCollisionEvent(123, 42))

    assert monitor.is_collided("agent_0")
    assert not monitor.is_collided("agent_1")
    assert monitor.collision_events["agent_0"][0]["other_actor_id"] == 123
    assert monitor.collision_events["agent_0"][0]["frame"] == 42


def test_collision_monitor_reset_clears_state():
    world = FakeWorld({"sensor.other.collision": object()})
    monitor = CollisionMonitor(world, {"agent_0": object()})

    monitor.sensors["agent_0"].emit(FakeCollisionEvent(9, 10))
    assert monitor.is_collided("agent_0")

    monitor.reset()

    assert not monitor.is_collided("agent_0")
    assert monitor.collision_events["agent_0"] == []


def test_collision_monitor_rejects_unknown_agent_id():
    monitor = CollisionMonitor(FakeWorld({"sensor.other.collision": object()}), {"agent_0": object()})

    with pytest.raises(KeyError, match="Unknown agent ID"):
        monitor.is_collided("missing")


def test_collision_monitor_raises_for_missing_blueprint_and_cleanup_is_safe():
    world = FakeWorld({})
    with pytest.raises(ValueError, match="Missing CARLA collision sensor blueprint"):
        CollisionMonitor(world, {"agent_0": object()})

    monitor = CollisionMonitor(
        FakeWorld({"sensor.other.collision": object()}),
        {"agent_0": object()},
    )
    monitor.destroy()
    monitor.destroy()

    assert monitor.sensors == {}


def test_collision_monitor_ignores_callback_exceptions():
    world = FakeWorld({"sensor.other.collision": object()})
    monitor = CollisionMonitor(world, {"agent_0": object()})

    class ExplodingEvent:
        @property
        def other_actor(self):
            raise RuntimeError("boom")

        @property
        def frame(self):
            raise RuntimeError("boom")

    monitor._on_collision("agent_0", ExplodingEvent())

    assert not monitor.is_collided("agent_0")
