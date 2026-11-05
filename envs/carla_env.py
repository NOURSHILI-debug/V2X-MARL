"""
CARLA multi-agent env (env spec v1, contract in envs/base.py).

Implemented:
  - connection, synchronous mode, fixed delta, seeded Traffic Manager
  - seeded spawn, one seeded route + goal per agent, clean teardown
  - 6 meta-actions on top of Traffic Manager control
  - collision monitor (envs/safety.py) and termination (T1.1, spec section 7):
      collision (any CAV), all CAVs at goal, timeout, stuck, off-road
TODO:
  - full observation vector + action masks      (T1.2)
  - reward, min_ttc and jerk in info            (T1.3)
"""
import random
import logging
import numpy as np
import carla

from envs.base import MultiAgentEnv
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
OBS_DIM = 56            # ego 8 + route 6 + candidates 6*6 + action mask 6
N_ACTIONS = 6
SIM_DT = 0.05           # 20 Hz simulation
TICKS_PER_STEP = 4      # 5 Hz decision rate
MAX_STEPS = 300         # 60 s of sim time
SPEED_STEP_KMH = 5.0

# ---- route / termination constants (spec section 7) -----------------------
ROUTE_STEP_M = 5.0      # waypoint spacing of the planned route
ROUTE_LENGTH_M = 150.0  # start-to-goal route length
GOAL_RADIUS_M = 6.0     # agent counts as "at goal" inside this radius
STUCK_SPEED_MS = 0.5    # slower than this counts as not moving
STUCK_STEPS = 50        # 10 s at 5 Hz

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
        self.client.set_timeout(30.0)
        self.world = self._load_town(town)
        self._set_sync(True)

        self.tm = self.client.get_trafficmanager(tm_port)
        self.tm.set_synchronous_mode(True)

        self.vehicles = {}        # agent_id -> carla.Vehicle
        self.target_speed = {}    # agent_id -> km/h
        self.route = {}           # agent_id -> list of carla.Waypoint (start ... goal)
        self.goal = {}            # agent_id -> carla.Location
        self.reached = {}         # agent_id -> bool
        self.stuck_steps = {}     # agent_id -> consecutive slow steps
        self.off_road_steps = {}  # agent_id -> consecutive off-road decision steps
        self.route_applied = {}   # agent_id -> Traffic Manager accepted the route
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
                self.world.tick()

    def close(self):
        try:
            self._destroy_actors()
        finally:
            try:
                self._set_sync(False)             # never leave the server stuck in sync
            finally:
                self.tm.set_synchronous_mode(False)

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
            self._set_sync(True)
            self.tm = self.client.get_trafficmanager(self.tm_port)
            self.tm.set_synchronous_mode(True)

        self._destroy_actors()
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
                self.target_speed[agent_id] = 30.0
                v.set_autopilot(True, self.tm_port)
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
        )

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
            slow = v.get_velocity().length() < STUCK_SPEED_MS
            self.stuck_steps[a] = self.stuck_steps[a] + 1 if (slow and not v.is_at_traffic_light()) else 0

    # ---- actions ----------------------------------------------------------
    def _apply_action(self, agent_id, action):
        v = self.vehicles[agent_id]
        limit = v.get_speed_limit() or 50.0                     # km/h
        t = self.target_speed[agent_id]

        if action == ACCEL:
            t = min(t + SPEED_STEP_KMH, limit)
        elif action == DECEL:
            t = max(t - SPEED_STEP_KMH, 0.0)
        elif action == BRAKE:
            t = 0.0
        elif action == LANE_LEFT:
            self.tm.force_lane_change(v, False)                 # False = left
        elif action == LANE_RIGHT:
            self.tm.force_lane_change(v, True)                  # True = right
        # KEEP: nothing

        self.target_speed[agent_id] = t
        # TM expresses speed as % below the limit
        self.tm.vehicle_percentage_speed_difference(v, (1.0 - t / limit) * 100.0)

    # ---- observations / masks --------------------------------------------
    def _get_obs(self):
        # TODO (T1.2): ego(8) + route(6) + candidates(6x6) + mask(6)
        return {a: np.zeros(OBS_DIM, dtype=np.float32) for a in self.agents}

    def state(self):
        """Global state for the QMIX mixer (training only)."""
        s = []
        for a in self.agents:
            v = self.vehicles[a]
            loc, vel = v.get_location(), v.get_velocity()
            s += [loc.x, loc.y, vel.x, vel.y, v.get_transform().rotation.yaw]
        return np.array(s, dtype=np.float32)

    def action_masks(self):
        # TODO (T1.2): mask lane changes when there is no adjacent driving lane
        return {a: np.ones(N_ACTIONS, dtype=bool) for a in self.agents}