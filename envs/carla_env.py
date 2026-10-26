"""
CARLA multi-agent env skeleton (env spec v1), PettingZoo-parallel style.

Done here (Person A, steps 1-4):
  - connection, synchronous mode, fixed delta, seeded Traffic Manager
  - seeded spawn of N CAVs, clean teardown
  - 6 discrete meta-actions on top of Traffic Manager control
  - reset(seed) / step(actions) API shell with placeholder obs

Left for later (marked TODO):
  - collision sensor + termination + reward  (Person B)
  - full observation vector, V2X layer, grouping hook
"""
import random
import numpy as np
import carla

# ---- spec constants -------------------------------------------------------
N_AGENTS = 4
M_CANDIDATES = 6
OBS_DIM = 56            # ego 8 + route 6 + candidates 6*6 + action mask 6
N_ACTIONS = 6
SIM_DT = 0.05           # 20 Hz simulation
TICKS_PER_STEP = 4      # 5 Hz decision rate
MAX_STEPS = 300         # 60 s of sim time
SPEED_STEP_KMH = 5.0

# meta-actions
KEEP, ACCEL, DECEL, BRAKE, LANE_LEFT, LANE_RIGHT = range(6)


class CarlaMultiAgentEnv:
    def __init__(self, town="Town03", n_agents=N_AGENTS, host="localhost",
                 port=2000, tm_port=8000, seed=0):
        self.town = town
        self.n_agents = n_agents
        self.agents = [f"cav_{i}" for i in range(n_agents)]
        self.seed = seed
        self.tm_port = tm_port

        self.client = carla.Client(host, port)
        self.client.set_timeout(30.0)
        self.world = self._load_town(town)
        self._set_sync(True)

        self.tm = self.client.get_trafficmanager(tm_port)
        self.tm.set_synchronous_mode(True)

        self.vehicles = {}        # agent_id -> carla.Vehicle
        self.target_speed = {}    # agent_id -> km/h
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
        if self.vehicles:
            self.client.apply_batch(
                [carla.command.DestroyActor(v.id) for v in self.vehicles.values()]
            )
        self.vehicles, self.target_speed = {}, {}
        self.world.tick()

    def close(self):
        self._destroy_actors()
        self._set_sync(False)             # never leave the server stuck in sync
        self.tm.set_synchronous_mode(False)

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
        self.tm.set_random_device_seed(self.seed)
        rng = random.Random(self.seed)

        spawn_points = self.world.get_map().get_spawn_points()
        rng.shuffle(spawn_points)
        bp_lib = self.world.get_blueprint_library().filter("vehicle.tesla.model3")

        for agent_id, sp in zip(self.agents, spawn_points):
            v = self.world.try_spawn_actor(bp_lib[0], sp)
            if v is None:
                continue
            v.set_autopilot(True, self.tm_port)
            self.tm.auto_lane_change(v, False)      # lane changes only by action
            self.tm.ignore_lights_percentage(v, 0)
            self.vehicles[agent_id] = v
            self.target_speed[agent_id] = 30.0
        assert len(self.vehicles) == self.n_agents, "not enough free spawn points"

        for _ in range(5):                          # let physics settle
            self.world.tick()

        self.step_count = 0
        return self._get_obs(), {"seed": self.seed, "town": self.town}

    def step(self, actions):
        """actions: dict agent_id -> int in [0, 5]"""
        for agent_id, a in actions.items():
            self._apply_action(agent_id, a)
        for _ in range(TICKS_PER_STEP):
            self.world.tick()
        self.step_count += 1

        obs = self._get_obs()
        rewards = {a: 0.0 for a in self.agents}                 # TODO (B): reward
        terminated = {a: False for a in self.agents}            # TODO (B): collision/goal
        truncated = {a: self.step_count >= MAX_STEPS for a in self.agents}
        info = {"step": self.step_count}
        return obs, rewards, terminated, truncated, info

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
        # TODO: ego(8) + route(6) + candidates(6x6, via V2X layer) + mask(6)
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
        # TODO: mask lane changes when no adjacent lane
        return {a: np.ones(N_ACTIONS, dtype=bool) for a in self.agents}