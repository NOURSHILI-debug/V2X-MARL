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

MAX_TTC_S = 10.0
DECISION_DT = 1.0 / 5.0
SCENARIO_ID = "highway-v0"
TOWN = "highway"


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
        first_space = self.env.observation_space[0]
        self.obs_dim = int(np.prod(first_space.shape))
        self.n_actions = N_ACTIONS
        self.steps = 0
        self._has_reset = False
        self._seed: int | None = None
        self._targets: dict[str, float] = {}
        self._prev_x: dict[str, float] = {}
        self._prev_speed: dict[str, float] = {}
        self._prev_accel: dict[str, float] = {}
        self._agent_jerk_sum: dict[str, float] = {}
        self._agent_jerk_count: dict[str, int] = {}
        self._agent_min_ttc: dict[str, float] = {}

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
            agent_id: self._flatten_observation(observation)
            for agent_id, observation in zip(
                self.agent_ids,
                native_observations,
                strict=True,
            )
        }

    def _flatten_observation(self, observation: Any) -> np.ndarray:
        flat = np.asarray(observation, dtype=np.float32).reshape(-1)
        if flat.shape != (self.obs_dim,):
            raise ValueError(
                f"Expected a flat observation of shape ({self.obs_dim},); "
                f"received {flat.shape}."
            )
        return np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        native_observations, native_info = self.env.reset(seed=seed)
        self.steps = 0
        self._has_reset = True
        self._seed = seed
        vehicles = self._vehicles()
        self._targets = {
            agent_id: float(vehicle.target_speed)
            for agent_id, vehicle in zip(self.agent_ids, vehicles, strict=True)
        }
        self._prev_x = {
            agent_id: float(vehicle.position[0])
            for agent_id, vehicle in zip(self.agent_ids, vehicles, strict=True)
        }
        self._prev_speed = {
            agent_id: float(vehicle.speed)
            for agent_id, vehicle in zip(self.agent_ids, vehicles, strict=True)
        }
        self._prev_accel = {agent_id: 0.0 for agent_id in self.agent_ids}
        self._agent_jerk_sum = {agent_id: 0.0 for agent_id in self.agent_ids}
        self._agent_jerk_count = {agent_id: 0 for agent_id in self.agent_ids}
        self._agent_min_ttc = {agent_id: MAX_TTC_S for agent_id in self.agent_ids}
        observations = self._format_observations(native_observations)
        infos = {
            "seed": seed,
            "town": TOWN,
            "scenario_id": SCENARIO_ID,
            "per_agent": {
                agent_id: {"seed": seed, "native_info": native_info}
                for agent_id in self.agent_ids
            },
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
        dict[str, Any],
    ]:
        if not self._has_reset:
            raise RuntimeError("Call reset() before step().")
        if set(actions) != set(self.agent_ids):
            raise ValueError(f"Actions must contain exactly {self.agent_ids}.")

        masks = self.action_masks()
        native_actions = []
        emergency_brakes = set()
        for agent_id in self.agent_ids:
            action = actions[agent_id]
            if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
                raise TypeError(f"Action for {agent_id} must be an integer.")
            if not 0 <= int(action) < N_ACTIONS:
                raise ValueError(f"Invalid action for {agent_id}: {action}.")
            action = int(action)
            if not masks[agent_id][action]:
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
        collision = any(crashed.values())
        terminated_flag = bool(native_terminated) or collision
        truncated_flag = bool(native_truncated) and not terminated_flag

        per_agent = {}
        for agent_id, vehicle in zip(self.agent_ids, vehicles, strict=True):
            x = float(vehicle.position[0])
            speed = float(vehicle.speed)
            acceleration = (speed - self._prev_speed[agent_id]) / DECISION_DT
            jerk = abs(acceleration - self._prev_accel[agent_id])
            ttc = self._lead_ttc(vehicle)
            self._agent_jerk_sum[agent_id] += jerk
            self._agent_jerk_count[agent_id] += 1
            self._agent_min_ttc[agent_id] = min(self._agent_min_ttc[agent_id], ttc)
            per_agent[agent_id] = {
                "collision": crashed[agent_id],
                "goal_reached": False,
                "x": x,
                "y": float(vehicle.position[1]),
                "speed_mps": speed,
                "crashed": crashed[agent_id],
                "min_ttc": self._agent_min_ttc[agent_id],
                "jerk": (
                    self._agent_jerk_sum[agent_id]
                    / self._agent_jerk_count[agent_id]
                ),
                "step": self.steps,
                "native_info": native_info,
            }
            self._prev_x[agent_id] = x
            self._prev_speed[agent_id] = speed
            self._prev_accel[agent_id] = acceleration

        total_jerk = sum(self._agent_jerk_sum.values())
        total_jerk_count = sum(self._agent_jerk_count.values())
        infos = {
            "collision": collision,
            "goal_reached": False,
            "success": False,
            "steps": self.steps,
            "min_ttc": min(self._agent_min_ttc.values()),
            "jerk": total_jerk / total_jerk_count if total_jerk_count else 0.0,
            "scenario_id": SCENARIO_ID,
            "town": TOWN,
            "seed": self._seed,
            "per_agent": per_agent,
        }
        rewards = {agent_id: float(native_reward) for agent_id in self.agent_ids}
        terminated = {agent_id: terminated_flag for agent_id in self.agent_ids}
        truncated = {agent_id: truncated_flag for agent_id in self.agent_ids}
        return (
            self._format_observations(observations),
            rewards,
            terminated,
            truncated,
            infos,
        )

    def _lead_ttc(self, ego: Any) -> float:
        best = MAX_TTC_S
        ego_x = float(ego.position[0])
        for other in self.env.unwrapped.road.vehicles:
            if other is ego:
                continue
            if tuple(other.lane_index) != tuple(ego.lane_index):
                continue
            gap = (
                float(other.position[0])
                - ego_x
                - 0.5 * (float(ego.LENGTH) + float(other.LENGTH))
            )
            if gap <= 0.0:
                continue
            closing = float(ego.speed) - float(other.speed)
            if closing > 0.0:
                best = min(best, gap / closing)
        return min(best, MAX_TTC_S)

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
