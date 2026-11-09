"""Small four-CAV adapter for the installed Highway-Env highway-v0."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import highway_env
import numpy as np

from envs.obs_builder import (
    CANDIDATE_RANGE,
    D_MAX,
    MAX_LANES,
    OBS_DIM as CONTRACT_OBS_DIM,
    ROUTE_FRAME_MAX,
    V_MAX,
    build_observation,
)
from envs.reward import RewardCalculator

N_ACTIONS = 6
AGENT_IDS = tuple(f"cav_{index}" for index in range(4))

# Highway-Env DiscreteMetaAction: left, idle, right, faster, slower.
_NATIVE_ACTIONS = (1, 3, 4, 1, 0, 2)

MAX_TTC_S = 10.0
DECISION_DT = 1.0 / 5.0
PROGRESS_DISTANCE = 30.0 * DECISION_DT
SCENARIO_ID = "highway-v0"
TOWN = "highway"

KINEMATICS25_LAYOUT = "kinematics25"
CONTRACT56_LAYOUT = "contract56"
OBS_LAYOUTS = (KINEMATICS25_LAYOUT, CONTRACT56_LAYOUT)
CONTRACT_MIN_VEHICLES = 12
CONTRACT_VEHICLES_DENSITY = 3.0
CAR_FOLLOW_MIN_GAP_M = 15.0
CAR_FOLLOW_HEADWAY_S = 1.0
ROUTE_LOOKAHEAD_M = min(10.0, ROUTE_FRAME_MAX)


class HighwayV2XEnv:
    """Expose the native multi-vehicle highway simulation as agent dictionaries."""

    def __init__(
        self,
        render_mode: str | None = "human",
        background_vehicles: int = 0,
        duration: float = 60,
        obs_layout: str = KINEMATICS25_LAYOUT,
        goal_distance_m: float | None = None,
    ) -> None:
        if obs_layout not in OBS_LAYOUTS:
            raise ValueError(f"obs_layout must be one of {OBS_LAYOUTS}.")
        if goal_distance_m is not None and goal_distance_m <= 0.0:
            raise ValueError("goal_distance_m must be positive when set.")
        self.obs_layout = obs_layout
        self.goal_distance_m = (
            None if goal_distance_m is None else float(goal_distance_m)
        )
        vehicles_count = background_vehicles
        if obs_layout == CONTRACT56_LAYOUT:
            vehicles_count = max(background_vehicles, CONTRACT_MIN_VEHICLES)
        gym.register_envs(highway_env)
        config = {
            "controlled_vehicles": len(AGENT_IDS),
            "vehicles_count": vehicles_count,
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
        if obs_layout == CONTRACT56_LAYOUT:
            config["vehicles_density"] = CONTRACT_VEHICLES_DENSITY
        self.env = gym.make("highway-v0", render_mode=render_mode, config=config)
        self.agent_ids = AGENT_IDS
        self.agents = list(AGENT_IDS)
        self.reward_calculator = RewardCalculator(agent_ids=self.agent_ids)
        first_space = self.env.observation_space[0]
        self.obs_dim = (
            CONTRACT_OBS_DIM
            if obs_layout == CONTRACT56_LAYOUT
            else int(np.prod(first_space.shape))
        )
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
        self._start_x: dict[str, float] = {}
        self._goal_reached: dict[str, bool] = {}
        self._vehicle_ids: dict[int, int] = {}

    def _vehicles(self) -> list[Any]:
        return list(self.env.unwrapped.controlled_vehicles)

    def _format_observations(self, native_observations: Any) -> dict[str, np.ndarray]:
        if self.obs_layout == CONTRACT56_LAYOUT:
            vehicles = self._vehicles()
            return {
                agent_id: self._contract_observation(agent_id, vehicle)
                for agent_id, vehicle in zip(
                    self.agent_ids,
                    vehicles,
                    strict=True,
                )
            }
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

    def _ensure_vehicle_ids(self) -> None:
        for vehicle in self.env.unwrapped.road.vehicles:
            key = id(vehicle)
            if key not in self._vehicle_ids:
                self._vehicle_ids[key] = len(self._vehicle_ids)

    def _vehicle_for_agent(self, agent_id: str) -> Any:
        try:
            index = self.agent_ids.index(agent_id)
        except ValueError:
            raise KeyError(f"Unknown agent ID: '{agent_id}'") from None
        return self._vehicles()[index]

    def get_candidates(self, agent_id: str) -> list[dict[str, Any]]:
        ego = self._vehicle_for_agent(agent_id)
        self._ensure_vehicle_ids()
        ego_x = float(ego.position[0])
        ego_y = float(ego.position[1])
        candidates = []
        for vehicle in self.env.unwrapped.road.vehicles:
            if vehicle is ego:
                continue
            x = float(vehicle.position[0])
            y = float(vehicle.position[1])
            distance = math.hypot(x - ego_x, y - ego_y)
            if distance > CANDIDATE_RANGE:
                continue
            candidates.append(
                (
                    distance,
                    self._vehicle_ids[id(vehicle)],
                    {
                        "actor_id": self._vehicle_ids[id(vehicle)],
                        "x": x,
                        "y": y,
                        "vx": float(vehicle.velocity[0]),
                        "vy": float(vehicle.velocity[1]),
                        "yaw": float(vehicle.heading),
                        "valid": True,
                    },
                )
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [candidate for _, _, candidate in candidates]

    def _lane_context(
        self,
        vehicle: Any,
    ) -> tuple[Any, float, float, int]:
        network = self.env.unwrapped.road.network
        lane = network.get_lane(vehicle.lane_index)
        longitudinal, lateral = lane.local_coordinates(vehicle.position)
        road_key = tuple(vehicle.lane_index[:2])
        lane_count = sum(
            1
            for lane_index in network.lanes_dict()
            if tuple(lane_index[:2]) == road_key
        )
        return lane, longitudinal, lateral, min(lane_count, int(MAX_LANES))

    def _contract_observation(
        self,
        agent_id: str,
        vehicle: Any,
    ) -> np.ndarray:
        lane, longitudinal, lateral, lane_count = self._lane_context(vehicle)
        lane_yaw = float(lane.heading_at(longitudinal))
        heading_error = math.atan2(
            math.sin(float(vehicle.heading) - lane_yaw),
            math.cos(float(vehicle.heading) - lane_yaw),
        )
        travelled = max(
            0.0,
            float(vehicle.position[0]) - self._start_x[agent_id],
        )
        if self.goal_distance_m is None:
            distance_to_goal = D_MAX
            progress_fraction = 0.0
        else:
            distance_to_goal = max(self.goal_distance_m - travelled, 0.0)
            progress_fraction = min(travelled / self.goal_distance_m, 1.0)
        lookahead = min(
            ROUTE_LOOKAHEAD_M,
            max(float(lane.length) - longitudinal, 0.0),
        )
        next_point = lane.position(longitudinal + lookahead, 0.0)
        ego = {
            "x": float(vehicle.position[0]),
            "y": float(vehicle.position[1]),
            "vx": float(vehicle.velocity[0]),
            "vy": float(vehicle.velocity[1]),
            "yaw": float(vehicle.heading),
            "speed": float(vehicle.speed),
            "longitudinal_accel": self._prev_accel.get(agent_id, 0.0),
            "heading_error": heading_error,
            "lateral_offset": float(lateral),
            "lane_width": float(lane.width),
            "lane_index": float(vehicle.lane_index[2]),
            "lane_count": float(lane_count),
            "distance_to_goal": float(distance_to_goal),
            "progress_fraction": float(progress_fraction),
        }
        route = {
            "next_waypoint_x": float(next_point[0]),
            "next_waypoint_y": float(next_point[1]),
            "speed_limit": float(lane.speed_limit),
            "at_intersection": False,
            "traffic_light_state": "none",
            "distance_to_next": float(lookahead),
        }
        return build_observation(
            ego,
            route,
            self.get_candidates(agent_id),
            self.action_masks()[agent_id],
        )

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        native_observations, native_info = self.env.reset(seed=seed)
        self.steps = 0
        self._has_reset = True
        self._seed = seed
        vehicles = self._vehicles()
        self._vehicle_ids = {}
        self._ensure_vehicle_ids()
        self._start_x = {
            agent_id: float(vehicle.position[0])
            for agent_id, vehicle in zip(self.agent_ids, vehicles, strict=True)
        }
        self._goal_reached = {agent_id: False for agent_id in self.agent_ids}
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

        pre_reached = dict(self._goal_reached)
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
            frozen = (
                self.goal_distance_m is not None
                and self._goal_reached.get(agent_id, False)
            )
            if agent_id in emergency_brakes or frozen:
                vehicle.target_speed = 0.0
                continue
            desired_speed = self._targets[agent_id]
            if self.obs_layout != CONTRACT56_LAYOUT:
                vehicle.target_speed = desired_speed
                continue
            action = int(actions[agent_id])
            native_delta = 5.0 if action == 1 else -5.0 if action == 2 else 0.0
            safe_speed = self._follow_safe_speed(
                vehicle,
                desired_speed + native_delta,
            )
            vehicle.target_speed = max(0.0, safe_speed - native_delta)
        observations, _native_reward, native_terminated, native_truncated, native_info = (
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

        per_agent = {}
        rewards = {}
        for agent_id, vehicle in zip(self.agent_ids, vehicles, strict=True):
            x = float(vehicle.position[0])
            speed = float(vehicle.speed)
            acceleration = (speed - self._prev_speed[agent_id]) / DECISION_DT
            jerk = abs(acceleration - self._prev_accel[agent_id])
            ttc = self._lead_ttc(vehicle)
            travelled = max(0.0, x - self._start_x[agent_id])
            newly_reached = False
            if (
                self.goal_distance_m is not None
                and not self._goal_reached[agent_id]
                and travelled >= self.goal_distance_m
            ):
                self._goal_reached[agent_id] = True
                newly_reached = True
            progress = (
                0.0
                if pre_reached.get(agent_id, False)
                else max(0.0, x - self._prev_x[agent_id]) / PROGRESS_DISTANCE
            )
            reward_terms = self.reward_calculator.compute_terms(
                agent_id,
                progress=progress,
                collision=crashed[agent_id],
                ttc=ttc,
                goal_reached=newly_reached,
                acceleration=acceleration,
                jerk=jerk,
                idle=speed < 1.0,
            )
            rewards[agent_id] = float(sum(reward_terms.values()))
            self._agent_jerk_sum[agent_id] += jerk
            self._agent_jerk_count[agent_id] += 1
            self._agent_min_ttc[agent_id] = min(self._agent_min_ttc[agent_id], ttc)
            per_agent[agent_id] = {
                "collision": crashed[agent_id],
                "goal_reached": self._goal_reached[agent_id],
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
                "reward_terms": reward_terms,
                "native_info": native_info,
            }
            self._prev_x[agent_id] = x
            self._prev_speed[agent_id] = speed
            self._prev_accel[agent_id] = acceleration

        all_reached = all(self._goal_reached.values())
        success_flag = all_reached and not collision
        terminated_flag = bool(native_terminated) or collision or all_reached
        truncated_flag = bool(native_truncated) and not terminated_flag

        total_jerk = sum(self._agent_jerk_sum.values())
        total_jerk_count = sum(self._agent_jerk_count.values())
        team_reward = self.reward_calculator.team_reward(
            rewards,
            collision=collision,
        )
        infos = {
            "collision": collision,
            "goal_reached": all_reached,
            "success": success_flag,
            "steps": self.steps,
            "min_ttc": min(self._agent_min_ttc.values()),
            "jerk": total_jerk / total_jerk_count if total_jerk_count else 0.0,
            "scenario_id": SCENARIO_ID,
            "town": TOWN,
            "seed": self._seed,
            "team_reward": team_reward,
            "per_agent": per_agent,
        }
        terminated = {agent_id: terminated_flag for agent_id in self.agent_ids}
        truncated = {agent_id: truncated_flag for agent_id in self.agent_ids}
        return (
            self._format_observations(observations),
            rewards,
            terminated,
            truncated,
            infos,
        )

    def _follow_safe_speed(self, vehicle: Any, aimed_speed: float) -> float:
        gap = None
        lead_speed = 0.0
        lanes = {tuple(vehicle.lane_index)}
        target_lane = getattr(vehicle, "target_lane_index", None)
        if target_lane is not None:
            lanes.add(tuple(target_lane))
        for other in self.env.unwrapped.road.vehicles:
            if other is vehicle:
                continue
            if tuple(other.lane_index) not in lanes:
                continue
            other_gap = (
                float(other.position[0])
                - float(vehicle.position[0])
                - 0.5 * (float(vehicle.LENGTH) + float(other.LENGTH))
            )
            if other_gap <= 0.0:
                continue
            if gap is None or other_gap < gap:
                gap = other_gap
                lead_speed = float(other.speed)
        if gap is None:
            return aimed_speed
        safe = lead_speed + (gap - CAR_FOLLOW_MIN_GAP_M) / CAR_FOLLOW_HEADWAY_S
        return max(0.0, min(aimed_speed, safe))

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
