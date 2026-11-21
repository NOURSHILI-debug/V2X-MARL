"""
CARLA multi-agent env (env spec v1, contract in envs/base.py).

Implemented:
  - synchronous CARLA setup, seeded spawn, routes, and teardown
  - 56-D observations and lane-aware action masks
  - target-speed actions with temporary emergency-brake override
  - shared collision, goal, timeout, stuck, and off-road termination
TODO:
  - reward integration and team reward           (T1.3)
"""
import logging
import math
import random
from itertools import combinations

import numpy as np
import carla

from envs.base import MultiAgentEnv
from envs.control import speed_limit_difference, update_target_speed
from envs.v2x_adapter import received_to_candidates
from envs.v2x_channel import V2XChannel, V2XConfig
from grouping import make_selector
from envs.obs_builder import (
    CANDIDATE_RANGE,
    MAX_LANES,
    OBS_DIM as OBSERVATION_DIM,
    V_MAX,
    build_observation,
    make_action_mask,
)
from envs.metrics import (
    DECISION_DT_SECONDS,
    TTC_CAP_SECONDS,
    longitudinal_dynamics,
    longitudinal_ttc,
)
from envs.reward import RewardCalculator
from envs.reward_integration import compute_agent_step_reward
from envs.safety import CollisionMonitor
from envs.termination import (
    OFF_ROAD_STEP_LIMIT,
    build_step_info,
    evaluate_termination,
    is_off_road,
    is_goal_reached,
    update_off_road_steps,
)

# ---- spec constants -------------------------------------------------------
N_AGENTS = 4
M_CANDIDATES = 6
OBS_DIM = OBSERVATION_DIM
N_ACTIONS = 6
SIM_DT = 0.05           # 20 Hz simulation
TICKS_PER_STEP = int(DECISION_DT_SECONDS / SIM_DT)  # 5 Hz decision rate
MAX_STEPS = 300         # 60 s of sim time

# ---- route / termination constants (spec section 7) -----------------------
ROUTE_STEP_M = 5.0      # waypoint spacing of the planned route
ROUTE_LENGTH_M = 150.0  # start-to-goal route length
GOAL_RADIUS = 8.0       # agent counts as "at goal" inside this radius
GOAL_MARGIN = 15.0      # goal is this far before the route endpoint
GOAL_RADIUS_M = GOAL_RADIUS  # compatibility alias for diagnostics
ROUTE_START_TOLERANCE_M = 3.0
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
                 port=2000, tm_port=8000, seed=0, reset_traffic_lights=True,
                 light_cycle=True, selection_mode="legacy", selection_k=2,
                 selection_seed=None, candidate_source="ground_truth",
                 v2x_config=None, v2x_seed=None,
                 v2x_broadcasters="cavs_and_background",
                 spawn_mode="default", spawn_cluster_max_distance=45.0,
                 max_steps=None):
        self.town = town
        self.n_agents = n_agents
        self.agents = [f"cav_{i}" for i in range(n_agents)]
        self.seed = seed
        self.tm_port = tm_port
        self.obs_dim = OBS_DIM
        self.n_actions = N_ACTIONS
        # Backward compatible: default preserves existing MAX_STEPS behavior.
        if max_steps is None:
            self.max_steps = MAX_STEPS
        else:
            if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)):
                raise ValueError("max_steps must be a positive integer or None")
            max_steps = int(max_steps)
            if max_steps < 1:
                raise ValueError("max_steps must be a positive integer or None")
            self.max_steps = max_steps
        self.light_cycle = bool(light_cycle)
        self.reset_traffic_lights = bool(reset_traffic_lights)

        # ---- teammate selection configuration -----------------------------
        self._configure_selection(
            selection_mode=selection_mode,
            selection_k=selection_k,
            selection_seed=selection_seed,
        )

        # ---- candidate source configuration --------------------------------
        self._configure_candidate_source(
            candidate_source=candidate_source,
            v2x_config=v2x_config,
            v2x_seed=v2x_seed,
            v2x_broadcasters=v2x_broadcasters,
            seed=seed,
        )

        # ---- spawn configuration (experimental, default unchanged) ---------
        self._configure_spawn(
            spawn_mode=spawn_mode,
            spawn_cluster_max_distance=spawn_cluster_max_distance,
        )

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
        self.arrival_step = {agent_id: None for agent_id in self.agents}
        self.stuck_steps = {}     # agent_id -> consecutive slow steps
        self.off_road_steps = {}  # agent_id -> consecutive off-road decision steps
        self.route_applied = {}   # agent_id -> Traffic Manager accepted the route
        self.finished = {}        # agent_id -> terminal state zero-fill marker
        self.stop_reason = {}     # agent_id -> goal/red_light/blocked/stuck/none
        self.monitor = None       # CollisionMonitor (envs/safety.py)
        self.step_count = 0
        self.min_ttc = TTC_CAP_SECONDS
        self.jerk_samples = []
        self.previous_longitudinal_acceleration = {
            agent_id: None for agent_id in self.agents
        }
        self.previous_velocity = {agent_id: None for agent_id in self.agents}
        self.previous_route_arc = {agent_id: 0.0 for agent_id in self.agents}
        self.reward_calculator = RewardCalculator(self.agents)

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

    # ---- configuration helpers (also used by CARLA-free tests) -----------
    def _configure_selection(self, selection_mode, selection_k, selection_seed):
        """Validate and store the teammate-selection configuration.

        ``selection_mode="legacy"`` (default) keeps the existing behaviour: the
        full eligible candidate pool is passed straight to build_observation.
        Any other mode routes candidates through a grouping selector first.
        """
        self.selection_mode = selection_mode
        self.selection_k = int(selection_k)
        if self.selection_k < 1:
            raise ValueError("selection_k must be >= 1")
        self._selector = make_selector(selection_mode)
        # Explicit Generator for reproducible random selection; never the
        # global NumPy RNG. reset(seed=...) controls CARLA/background
        # randomness separately from selection_seed.
        self._selection_rng = np.random.default_rng(selection_seed)
        # Selection result per agent from the most recent _get_obs() call,
        # used to write per-agent selection metadata into info.
        self._last_selection = {}

    def _configure_candidate_source(
        self,
        candidate_source,
        v2x_config,
        v2x_seed,
        v2x_broadcasters,
        seed,
    ):
        """Validate and store the candidate-source configuration.

        ``candidate_source="ground_truth"`` (default) keeps the existing
        behaviour: candidates come from CARLA world state via get_candidates().
        ``"v2x"`` routes candidates through the V2X channel instead.

        Seed policy: ``v2x_seed`` owns all channel randomness. When it is None
        the environment seed captured here is used, so runs stay reproducible
        by default. ``reset()`` replays the same channel seed and never consults
        ``reset(seed=...)``.
        """
        self.candidate_source = candidate_source
        if self.candidate_source not in ("ground_truth", "v2x"):
            raise ValueError(
                f"candidate_source {candidate_source!r} must be 'ground_truth' or 'v2x'"
            )
        self.v2x_broadcasters = v2x_broadcasters
        if self.v2x_broadcasters not in ("cavs_only", "cavs_and_background"):
            raise ValueError(
                f"v2x_broadcasters {v2x_broadcasters!r} must be 'cavs_only' "
                "or 'cavs_and_background'"
            )
        # The V2X channel is CARLA-independent (pure NumPy). It is only created
        # when v2x mode is active, so ground_truth mode has zero V2X overhead.
        self._v2x_channel = None
        self._v2x_stats = None
        self._v2x_seed = seed if v2x_seed is None else v2x_seed
        if self.candidate_source == "v2x":
            if v2x_config is None:
                v2x_config = V2XConfig()
            if not isinstance(v2x_config, V2XConfig):
                raise TypeError("v2x_config must be a V2XConfig instance or None")
            self._v2x_channel = V2XChannel(config=v2x_config, seed=self._v2x_seed)
        # Last action per agent, used as the V2X message "intent" field.
        self._last_action = {agent_id: KEEP for agent_id in self.agents}
        # Last received messages per agent, cached from _get_obs() for metadata.
        self._last_received = {}
        # Per-agent candidate-source metadata from the most recent _get_obs().
        self._last_candidate_meta = {}

    def _reset_v2x_state(self):
        """Clear V2X channel state and cached per-agent V2X metadata.

        Called by ``reset()`` so no in-flight message from a previous episode
        can be delivered, and so a fresh episode starts with an empty inbox.
        The channel replays its own documented seed (``v2x_seed``, or the
        environment seed captured at construction when ``v2x_seed`` is None).
        """
        if getattr(self, "_v2x_channel", None) is not None:
            self._v2x_channel.reset()
            self._v2x_stats = None
        self._last_received = {}
        self._last_candidate_meta = {}
        self._last_action = {agent_id: KEEP for agent_id in self.agents}

    # ---- spawn configuration (experimental) --------------------------------
    def _configure_spawn(self, spawn_mode, spawn_cluster_max_distance):
        """Configure optional experimental spawn selection.

        ``spawn_mode="default"`` preserves the existing shuffled scan over all
        CARLA spawn points. ``spawn_mode="cluster"`` tries groups of spawn
        points whose pairwise 2D distances are all within
        ``spawn_cluster_max_distance`` meters, preferring nearby/compatible
        placements while still subject to the existing route validation.
        """
        if spawn_mode not in ("default", "cluster"):
            raise ValueError(
                f"spawn_mode {spawn_mode!r} must be 'default' or 'cluster'"
            )
        self.spawn_mode = spawn_mode
        self.spawn_cluster_max_distance = float(spawn_cluster_max_distance)
        if self.spawn_mode == "cluster" and self.spawn_cluster_max_distance <= 0.0:
            raise ValueError("spawn_cluster_max_distance must be > 0 in cluster mode")

    @staticmethod
    def _pairwise_spawn_distance(spawn_a, spawn_b):
        a = spawn_a.location
        b = spawn_b.location
        return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))

    def _find_spawn_clusters(self, spawn_points, n_agents, max_distance):
        """Return deterministic clusters of spawn-point indices.

        Each cluster is a tuple of ``n_agents`` indices such that every pair of
        spawn points is within ``max_distance`` meters in the XY plane. Clusters
        are unique (as sorted index sets) and returned in deterministic order.
        """
        n = len(spawn_points)
        # Precompute neighbors within radius for pruning.
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self._pairwise_spawn_distance(spawn_points[i], spawn_points[j]) <= max_distance:
                    neighbors[i].add(j)
                    neighbors[j].add(i)

        clusters = []
        seen = set()
        # Deterministic enumeration by index order.
        for a in range(n):
            if len(neighbors[a]) < n_agents - 1:
                continue
            # Candidate pool: a and its neighbors, in index order.
            pool = [a] + sorted(neighbors[a])
            # Search combinations of n_agents from pool that include a and form a clique.
            for combo in combinations(pool, n_agents):
                if a not in combo:
                    continue
                key = tuple(sorted(combo))
                if key in seen:
                    continue
                ok = True
                for i in range(n_agents):
                    for j in range(i + 1, n_agents):
                        if self._pairwise_spawn_distance(
                            spawn_points[combo[i]], spawn_points[combo[j]]
                        ) > max_distance:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    seen.add(key)
                    clusters.append(combo)
        clusters.sort()
        return clusters

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
                self.stop_reason = {}
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
        path = self._tm_path(route)
        try:
            self.tm.set_path(vehicle, path)
        except Exception as exc:
            logger.warning("Traffic Manager did not accept route for %s: %s", agent_id, exc)
            return False
        return True

    @staticmethod
    def _tm_path(route):
        return [wp.transform.location for wp in route[4:-1:4]] + [
            route[-1].transform.location
        ]

    @staticmethod
    def _goal_before_route_end(path):
        """Return the path location GOAL_MARGIN metres before its endpoint."""
        if not path:
            raise ValueError("route path cannot be empty")
        if len(path) == 1:
            return path[0]
        remaining = GOAL_MARGIN
        for index in range(len(path) - 1, 0, -1):
            start, end = path[index - 1], path[index]
            segment = start.distance(end)
            if segment >= remaining:
                fraction = (segment - remaining) / segment if segment else 0.0
                return carla.Location(
                    x=start.x + (end.x - start.x) * fraction,
                    y=start.y + (end.y - start.y) * fraction,
                    z=start.z + (end.z - start.z) * fraction,
                )
            remaining -= segment
        return path[0]

    @staticmethod
    def _path_length_from_spawn(spawn_location, path):
        points = [spawn_location, *path]
        return sum(
            points[index].distance(points[index + 1])
            for index in range(len(points) - 1)
        )

    @staticmethod
    def _nearest_route_point(location, route):
        """Return distance and nearest waypoint index for a waypoint route."""
        best_distance = float("inf")
        best_index = 0
        for index, waypoint in enumerate(route):
            distance = location.distance(waypoint.transform.location)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_distance, best_index

    @staticmethod
    def _route_start_is_valid(location, route):
        distance, nearest_index = CarlaMultiAgentEnv._nearest_route_point(
            location, route
        )
        return distance <= ROUTE_START_TOLERANCE_M and nearest_index <= 2

    def _configure_traffic_lights(self):
        if not self.reset_traffic_lights:
            logger.info("Traffic lights left unchanged (reset_traffic_lights=False)")
            return
        lights = self.world.get_actors().filter("traffic.traffic_light*")
        configured_groups = set()
        for light in lights:
            if self.light_cycle:
                group = light.get_group_traffic_lights()
                if not group:
                    group = [light]
                group_key = tuple(sorted(
                    getattr(item, "id", id(item)) for item in group
                ))
                if group_key in configured_groups:
                    continue
                for group_light in group:
                    group_light.set_green_time(10.0)
                    group_light.set_yellow_time(2.0)
                    group_light.set_red_time(5.0)
                light.reset_group()
                for group_light in group:
                    group_light.freeze(False)
                    logger.info(
                        "Traffic light id=%s durations green=%.2fs yellow=%.2fs red=%.2fs",
                        getattr(group_light, "id", "unknown"),
                        group_light.get_green_time(),
                        group_light.get_yellow_time(),
                        group_light.get_red_time(),
                    )
                configured_groups.add(group_key)
            else:
                light.freeze(False)
        logger.info(
            "Traffic lights reset: light_cycle=%s lights=%d groups=%d durations=%s",
            self.light_cycle,
            len(lights),
            len(configured_groups),
            "green=10.0s yellow=2.0s red=5.0s" if self.light_cycle else "unchanged",
        )

    # ---- spawn helpers -----------------------------------------------------
    def _try_spawn_at_transform(
        self, agent_id, agent_index, spawn_transform, bp, carla_map
    ):
        """Try to spawn and validate one agent at ``spawn_transform``.

        Returns ``None`` if CARLA cannot spawn the actor at the transform.
        Otherwise returns ``(vehicle, route, goal, accepted, rejection_count)``
        where ``accepted`` is False if the spawn was rejected by route
        validation (the caller is responsible for destroying the vehicle).
        """
        v = self.world.try_spawn_actor(bp, spawn_transform)
        if v is None:
            return None
        v.set_autopilot(True, self.tm_port)
        self.world.tick()
        start_waypoint = carla_map.get_waypoint(v.get_location())
        agent_rng = random.Random(self.seed + agent_index)
        route = self._plan_route(start_waypoint, agent_rng)
        path = self._tm_path(route)
        goal = self._goal_before_route_end(path)
        goal_distance_at_reset = v.get_location().distance(goal)
        route_length = self._path_length_from_spawn(v.get_location(), path)
        logger.info(
            "Reset route for %s: distance_to_goal=%.2f m "
            "route_length=%.2f m goal_radius=%.2f m",
            agent_id, goal_distance_at_reset, route_length, GOAL_RADIUS,
        )
        start_to_goal_distance = v.get_location().distance(path[-1])
        if start_to_goal_distance > route_length:
            logger.warning(
                "Rejecting spawn for %s: Euclidean start-to-goal distance "
                "%.2f m exceeds planned route length %.2f m",
                agent_id,
                start_to_goal_distance,
                route_length,
            )
            return (v, route, goal, False, 1)
        assert start_to_goal_distance <= route_length
        start_distance, start_index = self._nearest_route_point(
            v.get_location(), route
        )
        if not self._route_start_is_valid(v.get_location(), route):
            logger.warning(
                "Rejecting spawn for %s: route start misaligned "
                "distance=%.2f m nearest_waypoint=%d",
                agent_id, start_distance, start_index,
            )
            return (v, route, goal, False, 1)
        return (v, route, goal, True, 0)

    def _spawn_agents_clustered(self, spawn_points, bp, carla_map, rng):
        """Spawn all agents from a nearby cluster of valid CARLA spawn points.

        Tries clusters whose pairwise XY distances are within
        ``spawn_cluster_max_distance`` in a seed-deterministic order. Each
        cluster attempt must pass the existing per-agent route validation for
        every agent; if any agent fails, the attempt is abandoned and the next
        cluster is tried.
        """
        clusters = self._find_spawn_clusters(
            spawn_points, self.n_agents, self.spawn_cluster_max_distance
        )
        if not clusters:
            raise RuntimeError(
                f"no spawn-point clusters found for {self.n_agents} agents "
                f"within {self.spawn_cluster_max_distance} m in {self.town}"
            )
        rng.shuffle(clusters)
        route_rejections = dict.fromkeys(self.agents, 0)

        for cluster in clusters:
            accepted_routes = {}
            cluster_ok = True
            for agent_index, agent_id in enumerate(self.agents):
                spawn_transform = spawn_points[cluster[agent_index]]
                result = self._try_spawn_at_transform(
                    agent_id, agent_index, spawn_transform, bp, carla_map
                )
                if result is None:
                    cluster_ok = False
                    break
                vehicle, route, goal, ok, rejection_count = result
                route_rejections[agent_id] += rejection_count
                if not ok:
                    vehicle.destroy()
                    cluster_ok = False
                    break
                self.vehicles[agent_id] = vehicle
                self.target_speed[agent_id] = CRUISE_SPEED_MPS
                self._set_tm_target_speed(agent_id)
                self.tm.auto_lane_change(vehicle, False)
                self.tm.ignore_lights_percentage(vehicle, 0)
                accepted_routes[agent_id] = (route, goal)
            if cluster_ok:
                return accepted_routes, route_rejections
            # Cluster attempt failed: destroy any vehicles placed in this attempt.
            for vehicle in list(self.vehicles.values()):
                try:
                    vehicle.destroy()
                except RuntimeError:
                    pass
            self.vehicles = {}
            self.target_speed = {}
            self.world.tick()

        self._destroy_actors()
        raise RuntimeError(
            f"could not find a valid clustered spawn+route for all agents "
            f"(tried {len(clusters)} clusters, max_distance="
            f"{self.spawn_cluster_max_distance} m)"
        )

    # ---- API --------------------------------------------------------------
    def reset(self, seed=None, town=None):
        if seed is not None:
            self.seed = seed
        if town is not None and town != self.town:
            self.close()
            self.town = town
            self.world = self._load_town(town)
            self.tm = self.client.get_trafficmanager(self.tm_port)

        # Reward state belongs to an episode and must exist before any route or
        # step path can read it, including failed/partial resets.
        self.previous_longitudinal_acceleration = {
            agent_id: None for agent_id in self.agents
        }
        self.previous_velocity = {agent_id: None for agent_id in self.agents}
        self.previous_route_arc = {agent_id: 0.0 for agent_id in self.agents}
        self.arrival_step = {agent_id: None for agent_id in self.agents}

        self._stop_sync_and_destroy()
        self._set_sync(True)
        self.tm.set_synchronous_mode(True)
        reset_complete = False
        try:
            self.tm.set_random_device_seed(self.seed)
            rng = random.Random(self.seed)

            spawn_points = self.world.get_map().get_spawn_points()
            bp = self.world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
            self._configure_traffic_lights()
            carla_map = self.world.get_map()
            accepted_routes = {}
            route_rejections = dict.fromkeys(self.agents, 0)

            if getattr(self, "spawn_mode", "default") == "cluster":
                accepted_routes, route_rejections = self._spawn_agents_clustered(
                    spawn_points, bp, carla_map, rng
                )
            else:
                rng.shuffle(spawn_points)
                spawn_cursor = 0
                for agent_index, agent_id in enumerate(self.agents):
                    accepted = False
                    while spawn_cursor < len(spawn_points):
                        spawn_transform = spawn_points[spawn_cursor]
                        spawn_cursor += 1
                        result = self._try_spawn_at_transform(
                            agent_id, agent_index, spawn_transform, bp, carla_map
                        )
                        if result is None:
                            # try_spawn_actor failed (occupied/invalid); not a route rejection
                            continue
                        vehicle, route, goal, ok, rejection_count = result
                        route_rejections[agent_id] += rejection_count
                        if not ok:
                            vehicle.destroy()
                            continue
                        self.vehicles[agent_id] = vehicle
                        self.target_speed[agent_id] = CRUISE_SPEED_MPS
                        self._set_tm_target_speed(agent_id)
                        self.tm.auto_lane_change(vehicle, False)
                        self.tm.ignore_lights_percentage(vehicle, 0)
                        accepted_routes[agent_id] = (route, goal)
                        accepted = True
                        break
                    if not accepted:
                        self._destroy_actors()
                        raise RuntimeError(
                            f"could not find a spawn with a valid route for {agent_id}"
                        )
            route_signatures = {
                agent_id: tuple(
                    (round(wp.transform.location.x, 2),
                     round(wp.transform.location.y, 2),
                     round(wp.transform.location.z, 2))
                    for wp in accepted_routes[agent_id][0]
                )
                for agent_id in self.agents
            }
            assert len(set(route_signatures.values())) == len(self.agents), (
                "each agent must have a distinct planned route"
            )
            logger.info("Route-invalid spawn rejections per reset: %s", route_rejections)

            for agent_id, v in self.vehicles.items():
                route, goal = accepted_routes[agent_id]
                self.route[agent_id] = route
                self.goal[agent_id] = goal
                self.reached[agent_id] = False
                self.arrival_step[agent_id] = None
                self.stuck_steps[agent_id] = 0
                self.off_road_steps[agent_id] = 0
                self.finished[agent_id] = False
                self.stop_reason[agent_id] = "none"
                applied = self._follow_route(agent_id, v, route)
                if not applied:
                    applied = self._follow_route(agent_id, v, route)
                self.route_applied[agent_id] = bool(applied)
                if not self.route_applied[agent_id]:
                    logger.warning("Route remains unapplied after retry for %s", agent_id)

            failed_route_agents = [
                agent_id for agent_id, applied in self.route_applied.items() if applied is not True
            ]
            if failed_route_agents:
                failure_info = {
                    "route_applied": False,
                    "route_application_failure": {
                        "reason": "traffic_manager_route_not_applied",
                        "failed_agents": failed_route_agents,
                        "per_agent": {
                            agent_id: {"route_applied": bool(self.route_applied.get(agent_id, False))}
                            for agent_id in self.agents
                        },
                    },
                }
                logger.error("Route application failed during reset: %s", failure_info)
                raise RuntimeError(
                    f"Route application failed for agents: {', '.join(failed_route_agents)}"
                )

            for _ in range(5):                          # let physics settle
                self.world.tick()
            self.monitor = CollisionMonitor(self.world, self.vehicles)   # after settling

            self.step_count = 0
            self.min_ttc = TTC_CAP_SECONDS
            self.jerk_samples = []
            self.previous_longitudinal_acceleration = {agent_id: None for agent_id in self.agents}
            self.previous_velocity = {agent_id: None for agent_id in self.agents}
            self.previous_route_arc = {agent_id: 0.0 for agent_id in self.agents}
            self._last_selection = {}
            # Reset the V2X channel at environment reset so stale in-flight
            # messages from the previous episode are never reused.
            self._reset_v2x_state()
            reset_info = {
                "seed": self.seed,
                "town": self.town,
                "spawn_mode": getattr(self, "spawn_mode", "default"),
                "spawn_cluster_max_distance": getattr(
                    self, "spawn_cluster_max_distance", None
                ),
                "route_applied": bool(all(self.route_applied.values())),
                "per_agent": {
                    agent_id: {"route_applied": applied}
                    for agent_id, applied in self.route_applied.items()
                },
            }
            # No channel step has happened yet at reset: the reset observation is
            # built with an empty V2X inbox so reset never advances the channel.
            observations = self._get_obs(advance_v2x=False)
            if getattr(self, "_v2x_channel", None) is not None:
                reset_info["v2x"] = {
                    "step_index": self._v2x_channel.step_index,
                    **self._v2x_channel.episode_stats(),
                }
            # _last_selection is now populated by _get_obs(); expose selection
            # metadata in reset info so it is available before the first step.
            candidate_source = getattr(self, "candidate_source", "ground_truth")
            if self._selector is not None:
                for agent_id in self.agents:
                    selection = self._last_selection.get(agent_id)
                    if selection is not None:
                        reset_info["per_agent"][agent_id]["selection"] = {
                            "strategy": selection.strategy,
                            "k": selection.k,
                            "candidate_source": candidate_source,
                            "pool_size": selection.pool_size,
                            "selected_count": selection.selected_count,
                            "missing_slots": selection.missing_slots,
                            "selected_ids": [i for i in selection.ids],
                            "selected_scores": [s for s in selection.scores],
                        }
            reset_complete = True
            return observations, reset_info
        finally:
            if not reset_complete:
                self._destroy_actors()

    def step(self, actions):
        """actions: dict agent_id -> int in [0, 5]"""
        if not isinstance(actions, dict) or set(actions) != set(self.agents):
            raise ValueError("actions must contain exactly the configured agent ids")
        for action in actions.values():
            if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= int(action) < N_ACTIONS:
                raise ValueError("each action must be an integer in 0..5")

        was_finished = {agent_id: self.finished.get(agent_id, False) for agent_id in self.agents}
        invalid_action = 0
        masks = self.action_masks()
        effective_actions = dict(actions)
        for agent_id, action in actions.items():
            if (
                action == LANE_LEFT and not masks[agent_id][LANE_LEFT]
            ) or (
                action == LANE_RIGHT and not masks[agent_id][LANE_RIGHT]
            ):
                # addition beyond spec §2, pending agreement
                effective_actions[agent_id] = KEEP
                invalid_action += 1
        # V2X "intent" field: the action actually applied this decision step.
        self._last_action = {
            agent_id: int(effective_actions[agent_id]) for agent_id in self.agents
        }

        for agent_id, a in effective_actions.items():
            if not self.reached[agent_id]:          # agents parked at their goal ignore actions
                self._apply_action(agent_id, a)
        for _ in range(TICKS_PER_STEP):
            self.world.tick()
        self.step_count += 1

        collided = {
            a: self.monitor.is_collided(a) if not self.finished.get(a, False) else False
            for a in self.agents
        }
        self._update_goal_and_stuck()

        step_ttc = self._measure_min_ttc()
        self.min_ttc = min(self.min_ttc, step_ttc)
        step_kinematics = {}
        for agent_id, vehicle in self.vehicles.items():
            if was_finished[agent_id]:
                continue
            velocity = vehicle.get_velocity()
            transform = vehicle.get_transform()
            yaw = math.radians(float(transform.rotation.yaw))
            acceleration_longitudinal, jerk = longitudinal_dynamics(
                (velocity.x, velocity.y),
                self.previous_velocity[agent_id],
                self.previous_longitudinal_acceleration[agent_id],
                yaw,
                DECISION_DT_SECONDS,
            )
            self.jerk_samples.append(jerk)
            self.previous_velocity[agent_id] = (float(velocity.x), float(velocity.y))
            self.previous_longitudinal_acceleration[agent_id] = acceleration_longitudinal
            step_kinematics[agent_id] = (acceleration_longitudinal, jerk)

        off_road_by_agent = {}
        for agent_id in self.agents:
            if self.finished.get(agent_id, False):
                off_road_by_agent[agent_id] = False
                self.off_road_steps[agent_id] = 0
                continue
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
            if result.terminated or result.truncated:
                self.finished[agent_id] = True

        obs = self._get_obs()
        rewards = {}
        reward_terms = {}
        for agent_id in self.agents:
            if was_finished[agent_id]:
                rewards[agent_id], reward_terms[agent_id] = compute_agent_step_reward(
                    self.reward_calculator, agent_id, distance_travelled=0.0,
                    finished=True,
                )
                continue
            route = self.route[agent_id]
            location = self.vehicles[agent_id].get_location()
            _, route_index = self._nearest_route_point(location, route)
            current_arc = sum(
                route[index - 1].transform.location.distance(
                    route[index].transform.location
                ) for index in range(1, route_index + 1)
            )
            previous_arc = self.previous_route_arc.get(agent_id, current_arc)
            distance_travelled = max(0.0, current_arc - previous_arc)
            self.previous_route_arc[agent_id] = current_arc
            acceleration_longitudinal, jerk = step_kinematics.get(agent_id, (0.0, 0.0))
            speed = self.vehicles[agent_id].get_velocity().length()
            at_red_light = (
                self.vehicles[agent_id].is_at_traffic_light()
                and self.vehicles[agent_id].get_traffic_light_state() == carla.TrafficLightState.Red
            )
            rewards[agent_id], terms = compute_agent_step_reward(
                self.reward_calculator,
                agent_id,
                distance_travelled=distance_travelled,
                collision=collided[agent_id],
                ttc=self._measure_agent_ttc(agent_id),
                goal_reached=self.reached[agent_id] and self.arrival_step[agent_id] == self.step_count,
                acceleration=acceleration_longitudinal,
                jerk=jerk,
                speed=speed,
                at_red_light=at_red_light,
            )
            reward_terms[agent_id] = terms
            assert math.isfinite(rewards[agent_id]), f"non-finite reward for {agent_id}"
        team_reward = self.reward_calculator.team_reward(
            rewards, collision=any(collided.values())
        )
        terminated = {
            a: bool(result.terminated or self.reached[a]) for a in self.agents
        }
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
            min_ttc=self.min_ttc,
            jerk=float(np.mean(self.jerk_samples)) if self.jerk_samples else 0.0,
            invalid_action=invalid_action,
            stop_reason=self.stop_reason,
            arrival_step=getattr(self, "arrival_step", dict.fromkeys(self.agents)),
            ending_reason=result.reason,
        )
        info["team_reward"] = float(team_reward)
        candidate_source = getattr(self, "candidate_source", "ground_truth")
        for agent_id in self.agents:
            info["per_agent"][agent_id]["reward_terms"] = reward_terms[agent_id]
            acceleration_longitudinal, jerk = step_kinematics.get(agent_id, (0.0, 0.0))
            info["per_agent"][agent_id]["a_long"] = float(acceleration_longitudinal)
            info["per_agent"][agent_id]["jerk"] = float(jerk)
            info["per_agent"][agent_id]["candidate_source"] = candidate_source
            candidate_meta = getattr(self, "_last_candidate_meta", {}).get(agent_id)
            if candidate_meta is not None:
                info["per_agent"][agent_id]["v2x_received"] = {
                    "count": candidate_meta["received_count"],
                    "sender_ids": list(candidate_meta["received_ids"]),
                    "ages": list(candidate_meta["received_ages"]),
                }
            if getattr(self, "_selector", None) is not None:
                last_sel = getattr(self, "_last_selection", None)
                if last_sel is not None:
                    selection = last_sel.get(agent_id)
                    if selection is not None:
                        info["per_agent"][agent_id]["selection"] = {
                            "strategy": selection.strategy,
                            "k": selection.k,
                            "candidate_source": candidate_source,
                            "pool_size": selection.pool_size,
                            "selected_count": selection.selected_count,
                            "missing_slots": selection.missing_slots,
                            "selected_ids": [i for i in selection.ids],
                            "selected_scores": [s for s in selection.scores],
                        }
        if (
            candidate_source == "v2x"
            and getattr(self, "_v2x_stats", None) is not None
        ):
            # Channel-level diagnostics for this environment decision step.
            info["v2x"] = {
                "step_index": self._v2x_channel.step_index,
                **self._v2x_stats,
                "episode": self._v2x_channel.episode_stats(),
            }
        return obs, rewards, terminated, truncated, info

    def _update_goal_and_stuck(self):
        for a, v in self.vehicles.items():
            if self.reached[a]:
                continue
            if is_goal_reached(v, self.goal[a], GOAL_RADIUS_M):
                self.reached[a] = True
                self.finished[a] = True
                self.stop_reason[a] = "goal"
                if self.arrival_step[a] is None:
                    self.arrival_step[a] = self.step_count
                self._remove_finished_agent(v)
                continue
            at_red_light = (
                v.is_at_traffic_light()
                and v.get_traffic_light_state() == carla.TrafficLightState.Red
            )
            if at_red_light:
                self.stuck_steps[a] = 0
                self.stop_reason[a] = "red_light"
            elif v.get_velocity().length() < STUCK_SPEED_MS:
                self.stuck_steps[a] += 1
                self.stop_reason[a] = (
                    "stuck" if self.stuck_steps[a] > STUCK_STEPS else "blocked"
                )
            else:
                self.stuck_steps[a] = 0
                self.stop_reason[a] = "none"

    def _remove_finished_agent(self, vehicle):
        vehicle.set_autopilot(False, self.tm_port)
        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=True,
            )
        )
        vehicle.set_simulate_physics(False)
        transform = vehicle.get_transform()
        transform.location.z -= 200.0
        vehicle.set_transform(transform)

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
        if self.finished.get(agent_id, False):
            return []
        ego = self.vehicles[agent_id]
        finished_actor_ids = {
            int(self.vehicles[finished_id].id)
            for finished_id, done in self.finished.items()
            if done and finished_id in self.vehicles
        }
        ego_location = ego.get_location()
        candidates = []
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == ego.id or int(actor.id) in finished_actor_ids:
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

    # ---- V2X candidate source -------------------------------------------
    def _v2x_finished_actor_ids(self):
        return {
            int(self.vehicles[agent_id].id)
            for agent_id, done in self.finished.items()
            if done and agent_id in self.vehicles
        }

    def _build_v2x_senders(self):
        """Build one outgoing V2X message per broadcasting vehicle.

        Message layout is the channel contract ``[x, y, vx, vy, heading, intent]``
        in world coordinates. Sender ids are numeric CARLA actor ids so that
        ``build_observation``'s actor-id tie-breaking stays meaningful.
        Background vehicles are included only when ``v2x_broadcasters`` says so,
        which keeps the candidate population equal to the ground-truth one.
        """
        finished_actor_ids = self._v2x_finished_actor_ids()
        cav_actor_ids = {
            int(vehicle.id) for vehicle in self.vehicles.values()
        }
        senders = []
        for agent_id, vehicle in self.vehicles.items():
            if self.finished.get(agent_id, False):
                continue
            location = vehicle.get_location()
            velocity = vehicle.get_velocity()
            transform = vehicle.get_transform()
            senders.append((
                int(vehicle.id),
                np.array(
                    [
                        float(location.x),
                        float(location.y),
                        float(velocity.x),
                        float(velocity.y),
                        math.radians(float(transform.rotation.yaw)),
                        float(self._last_action.get(agent_id, KEEP)),
                    ],
                    dtype=np.float64,
                ),
            ))
        if self.v2x_broadcasters == "cavs_only":
            return senders
        for actor in self.world.get_actors().filter("vehicle.*"):
            actor_id = int(actor.id)
            if actor_id in cav_actor_ids or actor_id in finished_actor_ids:
                continue
            location = actor.get_location()
            velocity = actor.get_velocity()
            transform = actor.get_transform()
            # Background vehicles have no controlled action; intent 0 ("keep")
            # is a documented placeholder. build_observation ignores intent.
            senders.append((
                actor_id,
                np.array(
                    [
                        float(location.x),
                        float(location.y),
                        float(velocity.x),
                        float(velocity.y),
                        math.radians(float(transform.rotation.yaw)),
                        0.0,
                    ],
                    dtype=np.float64,
                ),
            ))
        return senders

    def _v2x_candidates(self, agent_id, received):
        """Rebuild the eligible candidate pool for one agent from V2X messages.

        Mirrors ``get_candidates`` eligibility (self and finished actors
        excluded, 50 m candidate range) while sourcing positions from received
        messages instead of CARLA world state. The channel already rejects
        duplicate senders per step and delivers at most one message per sender
        per step, so duplicates are impossible in practice; the freshest message
        is kept defensively.
        """
        if self.finished.get(agent_id, False):
            self._last_candidate_meta.pop(agent_id, None)
            return []
        vehicle = self.vehicles.get(agent_id)
        if vehicle is None:
            self._last_candidate_meta.pop(agent_id, None)
            return []
        ego_actor_id = int(vehicle.id)
        ego_location = vehicle.get_location()
        inbox = received.get(ego_actor_id, []) if received else []
        self._last_candidate_meta[agent_id] = {
            "received_count": len(inbox),
            "received_ids": [int(message.sender_id) for message in inbox],
            "received_ages": [int(message.age) for message in inbox],
        }
        if not inbox:
            return []

        records = received_to_candidates(
            inbox,
            {"x": float(ego_location.x), "y": float(ego_location.y)},
        )
        finished_actor_ids = self._v2x_finished_actor_ids()
        # actor_id -> (age, distance, record)
        freshest: dict[int, tuple[int, float, dict]] = {}
        for record in records:
            try:
                numeric_id = int(record["actor_id"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"V2X candidate is missing a numeric sender id: {record!r}"
                ) from None
            if numeric_id == ego_actor_id or numeric_id in finished_actor_ids:
                continue
            distance = math.hypot(
                float(record["x"]) - float(ego_location.x),
                float(record["y"]) - float(ego_location.y),
            )
            if distance > CANDIDATE_RANGE:
                continue
            age = int(record.get("age", 0))
            previous = freshest.get(numeric_id)
            if previous is None or age < previous[0]:
                freshest[numeric_id] = (age, distance, record)

        candidates = []
        for numeric_id, (_, _, record) in sorted(
            freshest.items(), key=lambda item: (item[1][1], item[0])
        ):
            candidates.append({**record, "actor_id": numeric_id})
        return candidates

    def _measure_min_ttc(self):
        """Measure nearest same-lane lead TTC, capped when no lead is closing."""
        values = []
        actors = self.world.get_actors().filter("vehicle.*")
        finished_actor_ids = {
            int(self.vehicles[finished_id].id)
            for finished_id, done in self.finished.items()
            if done and finished_id in self.vehicles
        }
        for agent_id, ego in self.vehicles.items():
            if self.finished.get(agent_id, False):
                continue
            ego_location = ego.get_location()
            transform = ego.get_transform()
            yaw = math.radians(float(transform.rotation.yaw))
            forward_x, forward_y = math.cos(yaw), math.sin(yaw)
            ego_velocity = ego.get_velocity()
            ego_waypoint = self.world.get_map().get_waypoint(
                ego_location, project_to_road=True, lane_type=carla.LaneType.Driving,
            )
            leads = []
            for actor in actors:
                if actor.id == ego.id or int(actor.id) in finished_actor_ids:
                    continue
                location = actor.get_location()
                dx, dy = location.x - ego_location.x, location.y - ego_location.y
                gap = dx * forward_x + dy * forward_y
                if gap <= 0:
                    continue
                lead_waypoint = self.world.get_map().get_waypoint(
                    location, project_to_road=True, lane_type=carla.LaneType.Driving,
                )
                if (
                    lead_waypoint is None
                    or ego_waypoint is None
                    or lead_waypoint.road_id != ego_waypoint.road_id
                    or lead_waypoint.lane_id != ego_waypoint.lane_id
                ):
                    continue
                lead_velocity = actor.get_velocity()
                closing = (
                    (ego_velocity.x - lead_velocity.x) * forward_x
                    + (ego_velocity.y - lead_velocity.y) * forward_y
                )
                leads.append((gap, closing))
            gap, closing = min(leads, key=lambda pair: pair[0]) if leads else (None, None)
            values.append(longitudinal_ttc(gap, closing))
        return min(values, default=TTC_CAP_SECONDS)

    def _measure_agent_ttc(self, agent_id):
        """Measure same-lane TTC for one agent using the episode metric rules."""
        ego = self.vehicles[agent_id]
        location = ego.get_location()
        yaw = math.radians(float(ego.get_transform().rotation.yaw))
        forward_x, forward_y = math.cos(yaw), math.sin(yaw)
        ego_velocity = ego.get_velocity()
        ego_waypoint = self.world.get_map().get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving,
        )
        leads = []
        finished_actor_ids = {
            int(self.vehicles[finished_id].id)
            for finished_id, done in self.finished.items()
            if done and finished_id != agent_id and finished_id in self.vehicles
        }
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == ego.id or int(actor.id) in finished_actor_ids:
                continue
            other = actor.get_location()
            dx, dy = other.x - location.x, other.y - location.y
            gap = dx * forward_x + dy * forward_y
            if gap <= 0:
                continue
            waypoint = self.world.get_map().get_waypoint(
                other, project_to_road=True, lane_type=carla.LaneType.Driving,
            )
            if (waypoint is None or ego_waypoint is None
                    or waypoint.road_id != ego_waypoint.road_id
                    or waypoint.lane_id != ego_waypoint.lane_id):
                continue
            velocity = actor.get_velocity()
            closing = ((ego_velocity.x - velocity.x) * forward_x
                       + (ego_velocity.y - velocity.y) * forward_y)
            leads.append((gap, closing))
        gap, closing = min(leads, key=lambda pair: pair[0]) if leads else (None, None)
        return longitudinal_ttc(gap, closing)

    def _extract_observation_inputs(self, agent_id, received=None):
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

        # ---- teammate selection -------------------------------------------
        # Selectors receive the *full eligible pool* and are never given a
        # pre-truncated nearest-six pool. Eligibility rules are unchanged.
        # Selected world-frame records are passed to the observation builder;
        # the untouched pool is not modified.
        if getattr(self, "candidate_source", "ground_truth") == "v2x":
            raw_candidates = self._v2x_candidates(agent_id, received)
        else:
            self._last_candidate_meta.pop(agent_id, None)
            raw_candidates = self.get_candidates(agent_id)
        selector = getattr(self, "_selector", None)
        if selector is not None:
            ego_ref = {"x": float(location.x), "y": float(location.y)}
            selection = selector.select(
                raw_candidates,
                k=getattr(self, "selection_k", 2),
                ego=ego_ref,
                rng=getattr(self, "_selection_rng", None),
            )
            last_sel = getattr(self, "_last_selection", None)
            if last_sel is not None:
                last_sel[agent_id] = selection
            candidates_for_obs = selection.selected_records()
        else:
            last_sel = getattr(self, "_last_selection", None)
            if last_sel is not None:
                last_sel.pop(agent_id, None)
            candidates_for_obs = raw_candidates

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
            "candidates": candidates_for_obs,
            "action_mask": mask,
        }

    def _get_obs(self, advance_v2x=True):
        observations = {}
        # Step the V2X channel exactly once per decision step (not per agent,
        # not per 20 Hz tick). All agents read from the same received mapping.
        # reset() passes advance_v2x=False: no channel step has happened yet,
        # so the reset observation is built with an empty inbox.
        received = {}
        if (
            advance_v2x
            and getattr(self, "candidate_source", "ground_truth") == "v2x"
            and getattr(self, "_v2x_channel", None) is not None
        ):
            senders = self._build_v2x_senders()
            received, self._v2x_stats = self._v2x_channel.step(senders)
            self._last_received = received
        for agent_id in self.agents:
            if self.finished.get(agent_id, False):
                observations[agent_id] = np.zeros(self.obs_dim, dtype=np.float32)
                self._last_candidate_meta.pop(agent_id, None)
                continue
            data = self._extract_observation_inputs(agent_id, received=received)
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
