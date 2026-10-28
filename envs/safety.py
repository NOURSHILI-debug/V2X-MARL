"""Collision monitoring utilities for CARLA-based agent environments."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import carla


class CollisionMonitor:
    """Track per-agent collision events in a CARLA simulation."""

    def __init__(self, world: Any, vehicles: Dict[str, Any]):
        self.world = world
        self.vehicles = dict(vehicles)
        self._lock = threading.Lock()
        self._destroyed = False
        self._collision_flags: Dict[str, bool] = {
            agent_id: False for agent_id in self.vehicles
        }
        self.collision_events: Dict[str, List[Dict[str, Any]]] = {
            agent_id: [] for agent_id in self.vehicles
        }
        self.sensors: Dict[str, Any] = {}

        self._attach_sensors()

    def _attach_sensors(self) -> None:
        blueprint_library = getattr(self.world, "get_blueprint_library", None)
        if blueprint_library is None:
            raise ValueError(
                "World does not expose a blueprint library; cannot register collision sensors."
            )

        if callable(blueprint_library):
            blueprint_library = blueprint_library()

        collision_bp = blueprint_library.find("sensor.other.collision")
        if collision_bp is None:
            raise ValueError(
                "Missing CARLA collision sensor blueprint: 'sensor.other.collision'."
            )

        for agent_id, vehicle in self.vehicles.items():
            if vehicle is None:
                raise ValueError(f"Vehicle for agent '{agent_id}' is None.")

            try:
                sensor = self._spawn_sensor(agent_id, vehicle, collision_bp)
            except Exception as exc:
                self.destroy()
                raise ValueError(
                    f"Failed to spawn collision sensor for agent '{agent_id}'."
                ) from exc

            self.sensors[agent_id] = sensor

    def _spawn_sensor(self, agent_id: str, vehicle: Any, blueprint: Any) -> Any:
        try:
            import carla  # type: ignore
        except ImportError:  # pragma: no cover - used in non-CARLA test environments.
            carla = None

        try:
            if carla is not None:
                sensor = self.world.spawn_actor(
                    blueprint,
                    carla.Transform(),
                    attach_to=vehicle,
                )
            else:
                sensor = self.world.spawn_actor(
                    blueprint,
                    None,
                    attach_to=vehicle,
                )
        except Exception as exc:  # pragma: no cover - exercised via unit tests with fakes.
            raise exc

        if sensor is None:
            raise RuntimeError(f"Spawned sensor for agent '{agent_id}' is None.")

        try:
            sensor.listen(
                lambda event, agent_id=agent_id: self._on_collision(agent_id, event)
            )
        except Exception as exc:
            try:
                sensor.destroy()
            except Exception:
                pass
            raise exc

        return sensor

    def _on_collision(self, agent_id: str, event: Any) -> None:
        try:
            other_actor = getattr(event, "other_actor", None)
            frame = getattr(event, "frame", None)
            other_actor_id = getattr(other_actor, "id", None)
        except Exception:
            return

        try:
            with self._lock:
                if agent_id not in self._collision_flags:
                    return

                self._collision_flags[agent_id] = True
                self.collision_events[agent_id].append(
                    {
                        "frame": frame,
                        "other_actor_id": other_actor_id,
                    }
                )
        except Exception:
            return

    def is_collided(self, agent_id: str) -> bool:
        if agent_id not in self._collision_flags:
            raise KeyError(f"Unknown agent ID: '{agent_id}'")

        with self._lock:
            return bool(self._collision_flags[agent_id])

    def reset(self) -> None:
        with self._lock:
            for agent_id in self._collision_flags:
                self._collision_flags[agent_id] = False
                self.collision_events[agent_id].clear()

    def destroy(self) -> None:
        if self._destroyed:
            return

        self._destroyed = True

        for agent_id, sensor in list(self.sensors.items()):
            try:
                if hasattr(sensor, "stop"):
                    sensor.stop()
            except Exception:
                pass

            try:
                if hasattr(sensor, "destroy"):
                    sensor.destroy()
            except Exception:
                pass

            self.sensors.pop(agent_id, None)


__all__ = ["CollisionMonitor"]
