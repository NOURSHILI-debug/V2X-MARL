"""Small four-CAV adapter for the installed Highway-Env highway-v0."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import highway_env
import numpy as np

N_ACTIONS = 6
AGENT_IDS = tuple(f"cav_{index}" for index in range(4))

# Highway-Env DiscreteMetaAction: left, idle, right, faster, slower.
_NATIVE_ACTIONS = (1, 3, 4, 1, 0, 2)


class HighwayV2XEnv:
    """Expose the native multi-vehicle highway simulation as agent dictionaries."""

    def __init__(
        self,
        render_mode: str | None = "human",
        background_vehicles: int = 0,
        duration: float = 60,
    ) -> None:
        gym.register_envs(highway_env)
        config = {
            "controlled_vehicles": len(AGENT_IDS),
            "vehicles_count": background_vehicles,
            "lanes_count": 4,
            "duration": duration,
            "simulation_frequency": 20,
            "policy_frequency": 5,
            "ego_spacing": 0.5,
            "action": {
                "type": "MultiAgentAction",
                "action_config": {"type": "DiscreteMetaAction"},
            },
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {"type": "Kinematics"},
            },
            "screen_width": 1000,
            "screen_height": 300,
            "offscreen_rendering": False,
        }
        self.env = gym.make("highway-v0", render_mode=render_mode, config=config)
        self.agent_ids = AGENT_IDS
        self.agents = list(AGENT_IDS)
        self.steps = 0
        self._has_reset = False
        self._targets: dict[str, float] = {}

    def _vehicles(self) -> list[Any]:
        return list(self.env.unwrapped.controlled_vehicles)

    def _format_observations(self, native_observations: Any) -> dict[str, np.ndarray]:
        if not isinstance(native_observations, (tuple, list)):
            raise ValueError("Highway-Env did not return per-agent observations.")
        if len(native_observations) != len(self.agent_ids):
            raise ValueError(
                f"Expected {len(self.agent_ids)} observations; "
                f"received {len(native_observations)}."
            )
        return {
            agent_id: np.asarray(observation, dtype=np.float32).copy()
            for agent_id, observation in zip(
                self.agent_ids,
                native_observations,
                strict=True,
            )
        }

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        native_observations, native_info = self.env.reset(seed=seed)
        self.steps = 0
        self._has_reset = True
        self._targets = {
            agent_id: float(vehicle.target_speed)
            for agent_id, vehicle in zip(
                self.agent_ids,
                self._vehicles(),
                strict=True,
            )
        }
        observations = self._format_observations(native_observations)
        infos = {
            agent_id: {"seed": seed, "native_info": native_info}
            for agent_id in self.agent_ids
        }
        return observations, infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self._has_reset:
            raise RuntimeError("Call reset() before step().")
        if set(actions) != set(self.agent_ids):
            raise ValueError(f"Actions must contain exactly {self.agent_ids}.")

        native_actions = []
        emergency_brakes = set()
        for agent_id in self.agent_ids:
            action = actions[agent_id]
            if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
                raise TypeError(f"Action for {agent_id} must be an integer.")
            if not 0 <= int(action) < N_ACTIONS:
                raise ValueError(f"Invalid action for {agent_id}: {action}.")
            action = int(action)
            if not self.action_masks()[agent_id][action]:
                raise ValueError(f"Action {action} is invalid for {agent_id}.")
            if action == 1:
                self._targets[agent_id] += 5.0
            elif action == 2:
                self._targets[agent_id] = max(0.0, self._targets[agent_id] - 5.0)
            elif action == 3:
                emergency_brakes.add(agent_id)
            native_actions.append(_NATIVE_ACTIONS[action])

        vehicles = self._vehicles()
        for agent_id, vehicle in zip(self.agent_ids, vehicles, strict=True):
            vehicle.target_speed = (
                0.0 if agent_id in emergency_brakes else self._targets[agent_id]
            )
        observations, native_reward, native_terminated, native_truncated, native_info = (
            self.env.step(tuple(native_actions))
        )
        self.steps += 1
        for agent_id, vehicle in zip(self.agent_ids, vehicles, strict=True):
            if agent_id in emergency_brakes:
                vehicle.target_speed = self._targets[agent_id]

        crashed = {
            agent_id: bool(vehicle.crashed)
            for agent_id, vehicle in zip(self.agent_ids, vehicles, strict=True)
        }
        terminated_flag = bool(native_terminated) or any(crashed.values())
        truncated_flag = bool(native_truncated) and not terminated_flag
        infos = {}
        for agent_id, vehicle in zip(self.agent_ids, vehicles, strict=True):
            infos[agent_id] = {
                "x": float(vehicle.position[0]),
                "y": float(vehicle.position[1]),
                "speed_mps": float(vehicle.speed),
                "crashed": crashed[agent_id],
                "step": self.steps,
                "native_info": native_info,
            }
        rewards = {agent_id: float(native_reward) for agent_id in self.agent_ids}
        terminated = {
            agent_id: terminated_flag for agent_id in self.agent_ids
        }
        truncated = {
            agent_id: truncated_flag for agent_id in self.agent_ids
        }
        return self._format_observations(observations), rewards, terminated, truncated, infos

    def state(self) -> np.ndarray:
        values = []
        for vehicle in self._vehicles():
            values.extend(
                [
                    float(vehicle.position[0]),
                    float(vehicle.position[1]),
                    float(vehicle.velocity[0]),
                    float(vehicle.velocity[1]),
                    float(vehicle.heading),
                ]
            )
        return np.asarray(values, dtype=np.float32)

    def action_masks(self) -> dict[str, np.ndarray]:
        masks = {}
        for agent_id, vehicle in zip(
            self.agent_ids,
            self._vehicles(),
            strict=True,
        ):
            start, end, lane_id = vehicle.target_lane_index
            adjacent = {
                int(index[2])
                for index in self.env.unwrapped.road.network.side_lanes(
                    vehicle.target_lane_index
                )
            }
            mask = np.ones(N_ACTIONS, dtype=bool)
            mask[4] = int(lane_id) - 1 in adjacent
            mask[5] = int(lane_id) + 1 in adjacent
            masks[agent_id] = mask
        return masks

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()


__all__ = ["HighwayV2XEnv", "N_ACTIONS"]
