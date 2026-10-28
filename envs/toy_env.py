
import numpy as np


class ToyEnv:
    """Simple 2D intersection environment for two vehicles."""

    BRAKE = 0
    MAINTAIN = 1
    ACCELERATE = 2

    def __init__(self, dt=0.1, max_speed=5.0, max_steps=500):
        self.agents = ["vehicle_0", "vehicle_1"]
        self.n_agents = len(self.agents)
        self.obs_dims = 4

        self.dt = dt
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.vehicle_radius = 1.0
        self.accelerations = {
            self.BRAKE: -2.0,
            self.MAINTAIN: 0.0,
            self.ACCELERATE: 2.0,
        }

        self.reset()

    def reset(self, seed=None):
        """Reset both vehicles to their initial states."""
        self.rng = np.random.default_rng(seed)

        # Vehicle 0 approaches from the west.
        # Vehicle 1 approaches from the south.
        self.positions = np.array([
            [-10.0, 0.0],
            [0.0, -10.0],
        ])

        self.speeds = np.array([2.0, 2.0])
        self.steps = 0

        return self._get_observations(), {}

    def _get_observations(self):
        """Return each vehicle's local observation."""
        observations = {}

        for i, agent in enumerate(self.agents):
            distance_to_intersection = 10.0 - (
                self.positions[i, 0] if i == 0
                else self.positions[i, 1]
            )

            observations[agent] = np.array([
                self.positions[i, 0],
                self.positions[i, 1],
                self.speeds[i],
                distance_to_intersection,
            ], dtype=np.float32)

        return observations
    
    def step(self, actions):
        """Advance both vehicles by one simulation step."""
        expected_agents = set(self.agents)

        if set(actions) != expected_agents:
            raise ValueError(
                f"Actions must contain exactly: {self.agents}"
            )

        for action in actions.values():
            if action not in (
                self.BRAKE,
                self.MAINTAIN,
                self.ACCELERATE,
            ):
                raise ValueError(f"Invalid action: {action}")

        old_positions = self.positions.copy()

        # 1. Update both vehicle speeds.
        for i, agent in enumerate(self.agents):
            acceleration = self.accelerations[actions[agent]]

            self.speeds[i] = np.clip(
                self.speeds[i] + acceleration * self.dt,
                0.0,
                self.max_speed,
            )

        # 2. Move vehicle 0 along x and vehicle 1 along y.
        self.positions[0, 0] += self.speeds[0] * self.dt
        self.positions[1, 1] += self.speeds[1] * self.dt

        self.steps += 1

        collision = self._has_collision(old_positions)

        # 4. Check whether each vehicle has crossed the intersection.
        reached_goal = np.array([
            self.positions[0, 0] >= 10.0,
            self.positions[1, 1] >= 10.0,
        ])

        all_reached = bool(np.all(reached_goal))
        terminated_flag = collision or all_reached
        terminated = {agent: bool(terminated_flag) for agent in self.agents}
        truncated_flag = (
            self.steps >= self.max_steps and not terminated_flag
        )
        truncated = {agent: bool(truncated_flag) for agent in self.agents}

        # 5. Temporary reward: progress, with a collision penalty.
        progress = np.linalg.norm(
            self.positions - old_positions, axis=1
        )

        rewards = {
            agent: float(progress[i])
            for i, agent in enumerate(self.agents)
        }

        if collision:
            rewards = {
                agent: rewards[agent] - 100.0
                for agent in self.agents
            }

        info = {
            "collision": collision,
            "reached_goal": reached_goal.copy(),
            "positions": self.positions.copy(),
            "speeds": self.speeds.copy(),
            "steps": self.steps,
        }

        return (
            self._get_observations(),
            rewards,
            terminated,
            truncated,
            info,
        )

    def state(self):
        """Return the global state for centralized training."""
        return np.concatenate([
            self.positions.flatten(),
            self.speeds,
        ]).astype(np.float32)

    def action_masks(self):
        """All three actions are currently available to both vehicles."""
        return {
            agent: np.ones(3, dtype=bool)
            for agent in self.agents
        }
    
    def _has_collision(self, old_positions):
        """Detect a collision, including between-step overlap."""
        old_relative = old_positions[0] - old_positions[1]
        new_relative = self.positions[0] - self.positions[1]

        # Relative movement between the two vehicles.
        relative_motion = new_relative - old_relative
        motion_squared = np.dot(relative_motion, relative_motion)

        if motion_squared == 0:
            closest_t = 0.0
        else:
            closest_t = np.clip(
                -np.dot(old_relative, relative_motion) / motion_squared,
                0.0,
                1.0,
            )

        closest_relative = old_relative + closest_t * relative_motion
        minimum_distance = np.linalg.norm(closest_relative)

        return minimum_distance < 2 * self.vehicle_radius
