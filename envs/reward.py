"""Reward utilities for per-vehicle driving objectives.

The reward follows Contract v1 (docs/env_spec.md section 12): progress toward a
goal, collision penalties, TTC-based safety penalties, comfort penalties, a goal
bonus, and an idle penalty. All inputs are plain Python/NumPy numeric values, so
the calculator stays independent from CARLA actor APIs.

The caller is responsible for passing one-time event flags only when they occur in
the current step. For example, a collision or goal completion should be passed as
``collision=True`` or ``goal_reached=True`` only for the step on which the event
happens; repeated calls with the same state will continue to apply the same reward.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional


class RewardCalculator:
    """Compute deterministic scalar rewards for a controlled vehicle."""

    DEFAULT_WEIGHTS = {
        "progress": 1.0,
        "collision": -10.0,
        "ttc": -0.5,
        "ttc_threshold": 3.0,
        "comfort": -0.05,
        "goal": 10.0,
        "idle": -0.1,
        "team_collision": -10.0,
    }

    def __init__(
        self,
        agent_ids: Optional[Iterable[str]] = None,
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.agent_ids = tuple(agent_ids) if agent_ids is not None else ()
        self.agent_id_set = set(self.agent_ids)
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights is not None:
            self.weights.update({key: float(value) for key, value in weights.items()})

    def _validate_agent(self, agent_id: str) -> None:
        if self.agent_id_set and agent_id not in self.agent_id_set:
            raise KeyError(f"Unknown agent ID: '{agent_id}'")

    @staticmethod
    def _as_float(value: Any, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a finite numeric value.") from None
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite.")
        return number

    def _ttc_term(self, ttc: Optional[float]) -> float:
        if ttc is None:
            return 0.0
        ttc_value = self._as_float(ttc, "ttc")
        threshold = self.weights["ttc_threshold"]
        if ttc_value >= threshold:
            return 0.0
        fraction = 1.0 - max(ttc_value, 0.0) / max(threshold, 1e-9)
        return self.weights["ttc"] * fraction

    def _comfort_term(
        self,
        acceleration: Optional[float],
        jerk: Optional[float],
    ) -> float:
        magnitude = 0.0
        if acceleration is not None:
            magnitude += abs(self._as_float(acceleration, "acceleration"))
        if jerk is not None:
            magnitude += abs(self._as_float(jerk, "jerk"))
        return self.weights["comfort"] * magnitude

    def compute_terms(
        self,
        agent_id: str,
        *,
        progress: float,
        collision: bool = False,
        ttc: Optional[float] = None,
        goal_reached: bool = False,
        acceleration: Optional[float] = None,
        jerk: Optional[float] = None,
        idle: bool = False,
    ) -> Dict[str, float]:
        """Return the per-term reward contributions for one vehicle."""
        self._validate_agent(agent_id)
        progress_value = self._as_float(progress, "progress")

        terms = {
            "progress": self.weights["progress"] * progress_value,
            "collision": self.weights["collision"] if collision else 0.0,
            "safety": self._ttc_term(ttc),
            "comfort": self._comfort_term(acceleration, jerk),
            "goal": self.weights["goal"] if goal_reached else 0.0,
            "idle": self.weights["idle"] if idle else 0.0,
        }
        for name, value in terms.items():
            if not math.isfinite(value):
                raise ValueError(f"Reward term '{name}' is not finite.")
        return {name: float(value) for name, value in terms.items()}

    def compute(
        self,
        agent_id: str,
        *,
        progress: float,
        collision: bool = False,
        ttc: Optional[float] = None,
        goal_reached: bool = False,
        acceleration: Optional[float] = None,
        jerk: Optional[float] = None,
        idle: bool = False,
    ) -> float:
        """Compute the scalar reward for one vehicle in the current step."""
        terms = self.compute_terms(
            agent_id,
            progress=progress,
            collision=collision,
            ttc=ttc,
            goal_reached=goal_reached,
            acceleration=acceleration,
            jerk=jerk,
            idle=idle,
        )
        reward = float(sum(terms.values()))
        if not math.isfinite(reward):
            raise ValueError("Reward calculation produced a non-finite value.")
        return reward

    def team_reward(
        self,
        rewards: Mapping[str, float],
        *,
        collision: bool = False,
    ) -> float:
        """Return ``mean(r_i)`` plus the team collision penalty."""
        values = [self._as_float(value, "reward") for value in rewards.values()]
        team = sum(values) / len(values) if values else 0.0
        if collision:
            team += self.weights["team_collision"]
        if not math.isfinite(team):
            raise ValueError("Team reward calculation produced a non-finite value.")
        return float(team)

    def compute_batch(
        self,
        rewards: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, float]:
        """Compute rewards for multiple agents from a simple mapping."""
        results: Dict[str, float] = {}
        for agent_id, payload in rewards.items():
            results[agent_id] = self.compute(agent_id, **payload)
        return results


__all__ = ["RewardCalculator"]
