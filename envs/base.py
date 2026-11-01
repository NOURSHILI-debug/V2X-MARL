"""Env contract v1, shared by toy_env, highway_adapter and carla_env.

Rules (docs/env_spec.md, section 12):
  * Only adapters import `carla` or `highway_env`. Everything else is env-agnostic.
  * Every env follows the PettingZoo-parallel style API below.
  * Changing anything in this file needs both teammates to agree and env_spec.md updated.
"""
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, Dict, List, Tuple

import numpy as np

# ---- candidate neighbours (V2X / grouping input) ---------------------------
M_CANDIDATES = 6          # nearest vehicles exposed to the grouping module
CAND_FEATURES = 6         # [rel_x, rel_y, vx, vy, heading, valid]


def candidate_block(rows, m: int = M_CANDIDATES) -> np.ndarray:
    """rows: list of [rel_x, rel_y, vx, vy, heading], nearest first.
    Returns (m, 6) float32; missing slots are zero with valid=0."""
    block = np.zeros((m, CAND_FEATURES), np.float32)
    for i, r in enumerate(rows[:m]):
        block[i, :5] = r
        block[i, 5] = 1.0
    return block


# ---- canonical meta-actions (CARLA env and highway adapter) ----------------
class Action(IntEnum):
    KEEP = 0
    ACCEL = 1
    DECEL = 2
    BRAKE = 3
    LANE_LEFT = 4
    LANE_RIGHT = 5


N_ACTIONS = len(Action)   # the toy env may use fewer actions (see its n_actions)

# ---- info keys -------------------------------------------------------------
REQUIRED_STEP_INFO = ("collision", "goal_reached", "steps")   # every step()
REQUIRED_EPISODE_INFO = ("success", "min_ttc", "jerk")        # on the step that ends the episode


class MultiAgentEnv(ABC):
    """Contract every environment implements."""

    agents: List[str]      # agent ids, fixed per env instance
    obs_dim: int           # length of each agent's observation vector
    n_actions: int         # size of each agent's discrete action set

    @abstractmethod
    def reset(self, seed: int = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """-> (obs, info). obs: agent_id -> float vector (obs_dim,)."""

    @abstractmethod
    def step(self, actions: Dict[str, int]):
        """-> (obs, rewards, terminated, truncated, info)
        obs / rewards / terminated / truncated: dicts keyed by agent id.
        Episode is over when all(terminated.values()) or any(truncated.values())."""

    @abstractmethod
    def state(self) -> np.ndarray:
        """Global state, 1-D float vector. Training only (QMIX mixer, critics)."""

    @abstractmethod
    def action_masks(self) -> Dict[str, np.ndarray]:
        """agent_id -> bool vector (n_actions,), at least one True."""

    def close(self) -> None:
        pass