"""
CARLA multi-agent env (env spec v1, contract in envs/base.py).

Implemented:
  - synchronous CARLA setup, seeded spawn, routes, and teardown
  - 56-D observations and lane-aware action masks
  - target-speed actions with temporary emergency-brake override
  - shared collision, goal, timeout, stuck, and off-road termination
TODO:
  - reward, min_ttc and jerk in info            (T1.3)
"""
import logging
import math
import random

import numpy as np
import carla

from envs.base import MultiAgentEnv
from envs.control import speed_limit_difference, update_target_speed
from envs.obs_builder import (
    CANDIDATE_RANGE,
    MAX_LANES,
    OBS_DIM as OBSERVATION_DIM,
    V_MAX,
    build_observation,
    make_action_mask,
)
from envs.safety import CollisionMonitor
from envs.termination import (
    OFF_ROAD_STEP_LIMIT,
    build_step_info,
    evaluate_termination,
    is_off_road,
    update_off_road_steps,
)

# ---- spec constants -------------------------------------------------------
N_AGENTS = 4
M_CANDIDATES = 6
OBS_DIM = OBSERVATION_DIM
N_ACTIONS = 6
SIM_DT = 0.05           # 20 Hz simulation
TICKS_PER_STEP = 4      # 5 Hz decision rate
MAX_STEPS = 300         # 60 s of sim time

# ---- route / termination constants (spec section 7) -----------------------
ROUTE_STEP_M = 5.0      # waypoint spacing of the planned route
ROUTE_LENGTH_M = 150.0  # start-to-goal route length
GOAL_RADIUS_M = 6.0     # agent counts as "at goal" inside this radius
STUCK_SPEED_MS = 0.5    # slower than this counts as not moving
STUCK_STEPS = 50        # 10 s at 5 Hz
STUCK_GRACE_STEPS = 25  # 5 s after reset before stuck detection starts
CRUISE_SPEED_MPS = 9.0
SPEED_STEP_MPS = 2.0

# meta-actions
KEEP, ACCEL, DECEL, BRAKE, LANE_LEFT, LANE_RIGHT = range(6)

logger = logging.getLogger(__name__)


class CarlaMultiAgentEnv(MultiAgentEnv):
    def __init__(self, town="Town03", n_agents=N_AGENTS, host="localhost",
                 port=2000, tm_port=8000, seed=0):
        self.town = town
        self.n_agents = n_agents
        self.agents = [f"cav_{i}" for i in range(n_agents)]
        self.seed = seed
        self.tm_port = tm_port
        self.obs_dim = OBS_DIM
        self.n_actions = N_ACTIONS
        self.max_steps = MAX_STEPS

        self.client = carla.Client(host, port)
        self.client.set_timeout(30)
        self.world = self._load_town(town)
        self._set_sync(True)

        self.tm = self.client.get_trafficmanager(tm_port)
        self.tm.set_synchronous_mode(True)

        self.vehicles = {}        # agent_id -> carla.Vehicle
        self.target_speed = {}    # agent_id -> m/s
        self.route = {}           # agent_id -> list of carla.Waypoint (start ... goal)
        self.goal = {}            # agent_id -> carla.Location
        self.reached = {}         # agent_id -> bool
        self.stuck_steps = {}     # agent_id -> consecutive slow steps
        self.off_road_steps = {}  # agent_id -> consecutive off-road decision steps
        self.route_applied = {}   # agent_id -> Traffic Manager accepted the route
        self.finished = {}        # agent_id -> terminal state zero-fill marker
        self.monitor = None       # CollisionMonitor (envs/safety.py)
        self.step_count = 0

    # ---- setup helpers ----------------------------------------------------
    def _load_town(self, town):
        world = self.client.get_world()
        if world.get_map().name.split("/")[-1] != town:
            world = self.client.load_world(town)   # slow, only when needed
        return world

    def _set_sync(self, enabled):
        s = self.world.get_settings()
        s.synchronous_mode = enabled
        s.fixed_delta_seconds = SIM_DT if enabled else None
        self.world.apply_settings(s)

    # ---- teardown ---------------------------------------------------------
    def _destroy_actors(self):
        try:
            if self.monitor is not None:          # sensors first, then their vehicles
                self.monitor.destroy()
        finally:
            try:
                if self.vehicles:
                    self.client.apply_batch(
                        [carla.command.DestroyActor(v.id) for v in self.vehicles.values()]
                    )
            finally:
                self.monitor = None
                self.vehicles, self.target_speed = {}, {}
                self.route, self.goal, self.reached, self.stuck_steps = {}, {}, {}, {}
                self.off_road_steps, self.route_applied = {}, {}
                self.finished = {}
                self.world.tick()

    def close(self):
        self._stop_sync_and_destroy()

    def _stop_sync_and_destroy(self):
        try:
            self._set_sync(False)
        finally:
            try:
                self.tm.set_synchronous_mode(False)
            finally:
                self._destroy_actors()

    # ---- routes -----------------------------------------------------------
    def _plan_route(self, start_wp, rng):
        """Seeded lane-following route of ~ROUTE_LENGTH_M; random (seeded) choice at junctions."""
        route, dist = [start_wp], 0.0
        while dist < ROUTE_LENGTH_M:
            nxt = route[-1].next(ROUTE_STEP_M)
            if not nxt:
                break
            route.append(rng.choice(nxt))
            dist += ROUTE_STEP_M
        return route

    def _follow_route(self, agent_id, vehicle, route):
        """Make the Traffic Manager follow the planned route (sparse points, last = goal)."""
        path = [wp.transform.location for wp in route[4:-1:4]] + [route[-1].transform.location]
        try:
            self.tm.set_path(vehicle, path)
        except Exception as exc:
            logger.warning("Traffic Manager did not accept route for %s: %s", agent_id, exc)
            return False
        return True

    # ---- API --------------------------------------------------------------
    def reset(self, seed=None, town=None):
        if seed is not None:
            self.seed = seed
        if town is not None and town != self.town:
            self.close()
            self.town = town
            self.world = self._load_town(town)
            self.tm = self.client.get_trafficmanager(self.tm_port)

        self._stop_sync_and_destroy()
        self._set_sync(True)
        self.tm.set_synchronous_mode(True)
        reset_complete = False
        try:
            self.tm.set_random_device_seed(self.seed)
            rng = random.Random(self.seed)

            spawn_points = self.world.get_map().get_spawn_points()
            rng.shuffle(spawn_points)
            bp = self.world.get_blueprint_library().filter("vehicle.tesla.model3")[0]

            for sp in spawn_points:               # skip blocked points until all agents exist
                if len(self.vehicles) == self.n_agents:
                    break
                v = self.world.try_spawn_actor(bp, sp)
                if v is None:
                    continue
                agent_id = self.agents[len(self.vehicles)]
                self.vehicles[agent_id] = v
                self.target_speed[agent_id] = CRUISE_SPEED_MPS
                v.set_autopilot(True, self.tm_port)
                self._set_tm_target_speed(agent_id)
                self.tm.auto_lane_change(v, False)      # lane changes only by action
                self.tm.ignore_lights_percentage(v, 0)

            spawned_count = len(self.vehicles)
            if spawned_count != self.n_agents:
                self._destroy_actors()
                assert spawned_count == self.n_agents, "not enough free spawn points"

            carla_map = self.world.get_map()
            for agent_id, v in self.vehicles.items():
                route = self._plan_route(carla_map.get_waypoint(v.get_location()), rng)
                self.route[agent_id] = route
                self.goal[agent_id] = route[-1].transform.location
                self.reached[agent_id] = False
                self.stuck_steps[agent_id] = 0
                self.off_road_steps[agent_id] = 0
                self.finished[agent_id] = False
                self.route_applied[agent_id] = self._follow_route(agent_id, v, route)

            for _ in range(5):                          # let physics settle
                self.world.tick()
            self.monitor = CollisionMonitor(self.world, self.vehicles)   # after settling

            self.step_count = 0
            reset_info = {
                "seed": self.seed,
                "town": self.town,
                "route_applied": bool(all(self.route_applied.values())),
                "per_agent": {
                    agent_id: {"route_applied": applied}
                    for agent_id, applied in self.route_applied.items()
                },
            }
            reset_complete = True
            return self._get_obs(), reset_info
        finally:
            if not reset_complete:
                self._destroy_actors()

    def step(self, actions):
        """actions: dict agent_id -> int in [0, 5]"""
        for agent_id, a in actions.items():
            if not self.reached[agent_id]:          # agents parked at their goal ignore actions
                self._apply_action(agent_id, a)
        for _ in range(TICKS_PER_STEP):
            self.world.tick()
        self.step_count += 1

        collided = {a: self.monitor.is_collided(a) for a in self.agents}
        self._update_goal_and_stuck()

        off_road_by_agent = {}
        for agent_id in self.agents:
            waypoint = self.world.get_map().get_waypoint(
                self.vehicles[agent_id].get_location(),
                project_to_road=False,
                lane_type=carla.LaneType.Driving,
            )
            off_road_by_agent[agent_id] = is_off_road(waypoint)
            self.off_road_steps[agent_id] = update_off_road_steps(
                self.off_road_steps[agent_id],
                off_road_by_agent[agent_id],
            )

        result = evaluate_termination(
            self.agents,
            collided,
            self.reached,
            self.off_road_steps,
            self.stuck_steps,
            step_count=self.step_count,
            max_steps=self.max_steps,
            off_road_step_limit=OFF_ROAD_STEP_LIMIT,
            stuck_step_limit=STUCK_STEPS,
            stuck_grace_steps=STUCK_GRACE_STEPS,
        )
        for agent_id in self.agents:
            if (
                self.reached[agent_id]
                or collided[agent_id]
                or self.off_road_steps[agent_id] >= OFF_ROAD_STEP_LIMIT
                or result.stuck_by_agent[agent_id]
                or result.terminated
                or result.truncated
            ):
                self.finished[agent_id] = True

        obs = self._get_obs()
        rewards = {a: 0.0 for a in self.agents}                 # TODO (T1.3)
        terminated = {a: result.terminated for a in self.agents}
        truncated = {a: result.truncated for a in self.agents}
        info = build_step_info(
            self.agents,
            collided,
            self.reached,
            result.stuck_by_agent,
            off_road_by_agent,
            self.route_applied,
            steps=self.step_count,
            episode_ended=result.terminated or result.truncated,
            success=result.success,
            town=self.town,
            seed=self.seed,
        )
        return obs, rewards, terminated, truncated, info

    def _update_goal_and_stuck(self):
        for a, v in self.vehicles.items():
            if self.reached[a]:
                continue
            if v.get_location().distance(self.goal[a]) < GOAL_RADIUS_M:
                self.reached[a] = True
                self.tm.vehicle_percentage_speed_difference(v, 100.0)   # park at the goal
                continue
            if self.step_count <= STUCK_GRACE_STEPS:
                self.stuck_steps[a] = 0
                continue
            slow = v.get_velocity().length() < STUCK_SPEED_MS
            self.stuck_steps[a] = self.stuck_steps[a] + 1 if (slow and not v.is_at_traffic_light()) else 0

    # ---- actions ----------------------------------------------------------
    def _set_tm_target_speed(self, agent_id):
        vehicle = self.vehicles[agent_id]
        speed_limit_kmh = vehicle.get_speed_limit() or 50.0
        self.tm.vehicle_percentage_speed_difference(
            vehicle,
            speed_limit_difference(
                self.target_speed[agent_id],
                speed_limit_kmh,
            ),
        )

    def _apply_action(self, agent_id, action):
        v = self.vehicles[agent_id]
        if action == BRAKE:
            self.tm.vehicle_percentage_speed_difference(v, 100.0)
            return

        self.target_speed[agent_id] = update_target_speed(
            self.target_speed[agent_id],
            action,
            speed_step=SPEED_STEP_MPS,
            max_speed=V_MAX,
        )
        if action == LANE_LEFT:
            self.tm.force_lane_change(v, False)                 # False = left
        elif action == LANE_RIGHT:
            self.tm.force_lane_change(v, True)                  # True = right
        self._set_tm_target_speed(agent_id)

    # ---- observations / masks --------------------------------------------
    @staticmethod
    def _same_direction_driving_lane(waypoint, adjacent):
        return (
            adjacent is not None
            and adjacent.lane_type == carla.LaneType.Driving
            and adjacent.road_id == waypoint.road_id
            and adjacent.lane_id * waypoint.lane_id > 0
        )

    def _adjacent_lane(self, waypoint, direction):
        adjacent = getattr(waypoint, f"get_{direction}_lane")()
        if self._same_direction_driving_lane(waypoint, adjacent):
            return adjacent
        return None

    def _lane_context(self, waypoint):
        left = self._adjacent_lane(waypoint, "left")
        right = self._adjacent_lane(waypoint, "right")
        left_count = 0
        cursor = left
        while cursor is not None and left_count < int(MAX_LANES) - 1:
            left_count += 1
            cursor = self._adjacent_lane(cursor, "left")
        right_count = 0
        cursor = right
        while cursor is not None and left_count + right_count < int(MAX_LANES) - 1:
            right_count += 1
            cursor = self._adjacent_lane(cursor, "right")
        lane_count = min(left_count + right_count + 1, int(MAX_LANES))
        return left, right, left_count, lane_count

    def get_candidates(self, agent_id):
        """Return nearby controlled/background vehicles in stable distance/ID order."""
        ego = self.vehicles[agent_id]
        ego_location = ego.get_location()
        candidates = []
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == ego.id:
                continue
            location = actor.get_location()
            distance = math.hypot(
                location.x - ego_location.x,
                location.y - ego_location.y,
            )
            if distance > CANDIDATE_RANGE:
                continue
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            candidates.append(
                (
                    distance,
                    int(actor.id),
                    {
                        "actor_id": int(actor.id),
                        "x": float(location.x),
                        "y": float(location.y),
                        "vx": float(velocity.x),
                        "vy": float(velocity.y),
                        "yaw": math.radians(float(transform.rotation.yaw)),
                    },
                )
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [candidate for _, _, candidate in candidates]

    def _extract_observation_inputs(self, agent_id):
        vehicle = self.vehicles[agent_id]
        transform = vehicle.get_transform()
        location = transform.location
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()
        yaw = math.radians(float(transform.rotation.yaw))
        waypoint = self.world.get_map().get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        left_lane, right_lane, lane_index, lane_count = self._lane_context(waypoint)
        lane_transform = waypoint.transform
        lane_center = lane_transform.location
        lane_right = lane_transform.get_right_vector()
        lateral_offset = (
            (location.x - lane_center.x) * lane_right.x
            + (location.y - lane_center.y) * lane_right.y
        )
        lane_yaw = math.radians(float(lane_transform.rotation.yaw))
        heading_error = math.atan2(
            math.sin(yaw - lane_yaw),
            math.cos(yaw - lane_yaw),
        )
        forward_x, forward_y = math.cos(yaw), math.sin(yaw)
        longitudinal_accel = acceleration.x * forward_x + acceleration.y * forward_y

        route = self.route[agent_id]
        route_index = min(
            range(len(route)),
            key=lambda index: route[index].transform.location.distance(location),
        )
        next_index = min(route_index + 1, len(route) - 1)
        next_waypoint = route[next_index]
        distance_to_next = location.distance(next_waypoint.transform.location)
        remaining_distance = distance_to_next + sum(
            route[index].transform.location.distance(
                route[index + 1].transform.location
            )
            for index in range(next_index, len(route) - 1)
        )
        route_length = sum(
            route[index].transform.location.distance(
                route[index + 1].transform.location
            )
            for index in range(len(route) - 1)
        )
        goal_distance = location.distance(self.goal[agent_id])
        progress_fraction = (
            1.0 if self.reached[agent_id]
            else 1.0 - remaining_distance / route_length
            if route_length > 0.0
            else 0.0
        )
        traffic_light = vehicle.get_traffic_light()
        if traffic_light is None:
            traffic_light_state = "none"
        else:
            state = vehicle.get_traffic_light_state()
            if state == carla.TrafficLightState.Red:
                traffic_light_state = "red"
            elif state == carla.TrafficLightState.Yellow:
                traffic_light_state = "yellow"
            else:
                traffic_light_state = "green"

        mask = make_action_mask(left_lane is not None, right_lane is not None)
        return {
            "ego": {
                "x": float(location.x),
                "y": float(location.y),
                "vx": float(velocity.x),
                "vy": float(velocity.y),
                "yaw": yaw,
                "speed": math.hypot(velocity.x, velocity.y),
                "longitudinal_accel": float(longitudinal_accel),
                "heading_error": heading_error,
                "lateral_offset": float(lateral_offset),
                "lane_width": float(waypoint.lane_width),
                "lane_index": float(lane_index),
                "lane_count": float(lane_count),
                "distance_to_goal": float(goal_distance),
                "progress_fraction": float(progress_fraction),
            },
            "route": {
                "next_waypoint_x": float(next_waypoint.transform.location.x),
                "next_waypoint_y": float(next_waypoint.transform.location.y),
                "speed_limit": float(vehicle.get_speed_limit()) / 3.6,
                "at_intersection": bool(waypoint.is_junction or next_waypoint.is_junction),
                "traffic_light_state": traffic_light_state,
                "distance_to_next": float(distance_to_next),
            },
            "candidates": self.get_candidates(agent_id),
            "action_mask": mask,
        }

    def _get_obs(self):
        observations = {}
        for agent_id in self.agents:
            data = self._extract_observation_inputs(agent_id)
            observations[agent_id] = build_observation(
                data["ego"],
                data["route"],
                data["candidates"],
                data["action_mask"],
            )
        return observations

    def state(self):
        """Global state; finished agents keep their six-value zero-filled slot."""
        s = np.zeros((self.n_agents, 6), dtype=np.float32)
        for a in self.agents:
            if a in self.finished and self.finished[a]:
                continue
            if a not in self.vehicles:
                continue
            v = self.vehicles[a]
            loc, vel = v.get_location(), v.get_velocity()
            index = self.agents.index(a)
            s[index] = [
                loc.x,
                loc.y,
                vel.x,
                vel.y,
                math.radians(float(v.get_transform().rotation.yaw)),
                loc.distance(self.goal[a]),
            ]
        return s.reshape(-1)

    def action_masks(self):
        masks = {}
        for agent_id in self.agents:
            if agent_id not in self.vehicles:
                masks[agent_id] = make_action_mask(False, False)
                continue
            waypoint = self.world.get_map().get_waypoint(
                self.vehicles[agent_id].get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            left, right, _, _ = self._lane_context(waypoint)
            masks[agent_id] = make_action_mask(
                left_lane_exists=left is not None,
                right_lane_exists=right is not None,
            )
        return masks