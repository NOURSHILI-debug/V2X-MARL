"""Reward utilities for per-vehicle driving objectives.

The reward is intentionally simple and explicit: it combines progress toward a goal,
collision penalties, TTC-based safety penalties, comfort penalties, and a goal bonus.
All inputs are plain Python/NumPy numeric values, so the calculator stays independent
from CARLA actor APIs.

The caller is responsible for passing one-time event flags only when they occur in the
current step. For example, a collision or goal completion should be passed as
``collision=True`` or ``goal_reached=True`` only for the step on which the event
happens; repeated calls with the same state will continue to apply the same reward.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional


class RewardCalculator:
    """Compute deterministic scalar rewards for a controlled vehicle.

    Default weights are intentionally conservative and can be overridden at
    initialization. Progress and goal completion are positive rewards; collisions,
    low TTC, and excessive comfort signals are negative penalties.
    """

    DEFAULT_WEIGHTS = {
        "progress": 1.0,
        "collision": 100.0,
        "safety": 15.0,
        "safety_threshold": 2.5,
        "comfort_accel": 1.0,
        "comfort_jerk": 1.0,
        "comfort_accel_threshold": 2.0,
        "comfort_jerk_threshold": 2.0,
        "goal": 50.0,
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

    def _safety_penalty(self, ttc: Optional[float]) -> float:
        if ttc is None:
            return 0.0
        ttc_value = self._as_float(ttc, "ttc")
        threshold = self.weights["safety_threshold"]
        if ttc_value >= threshold:
            return 0.0
        if ttc_value <= 0.0:
            return self.weights["safety"]
        normalized = (threshold - ttc_value) / max(threshold, 1e-9)
        return self.weights["safety"] * normalized

    def _comfort_penalty(
        self,
        value: Optional[float],
        *,
        weight_key: str,
        threshold_key: str,
        name: str,
    ) -> float:
        if value is None:
            return 0.0
        numeric = self._as_float(value, name)
        threshold = self.weights[threshold_key]
        magnitude = abs(numeric)
        if magnitude <= threshold:
            return 0.0
        return self.weights[weight_key] * (magnitude - threshold)

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
    ) -> float:
        """Compute the scalar reward for one vehicle in the current step.

        Positive terms:
            - progress contribution: ``weight['progress'] * progress``
            - goal completion bonus: ``weight['goal']`` when ``goal_reached`` is true

        Negative terms:
            - collision penalty: ``weight['collision']`` when ``collision`` is true
            - TTC safety penalty when TTC is below the configured threshold
            - comfort penalties when acceleration or jerk exceed configured limits

        Missing safety or comfort measurements are ignored explicitly; they are not
        treated as zero-valued penalties.
        """
        self._validate_agent(agent_id)
        progress_value = self._as_float(progress, "progress")

        reward = self.weights["progress"] * progress_value

        if collision:
            reward -= self.weights["collision"]

        reward -= self._safety_penalty(ttc)

        reward -= self._comfort_penalty(
            acceleration,
            weight_key="comfort_accel",
            threshold_key="comfort_accel_threshold",
            name="acceleration",
        )
        reward -= self._comfort_penalty(
            jerk,
            weight_key="comfort_jerk",
            threshold_key="comfort_jerk_threshold",
            name="jerk",
        )

        if goal_reached:
            reward += self.weights["goal"]

        if not math.isfinite(reward):
            raise ValueError("Reward calculation produced a non-finite value.")
        return float(reward)

    def compute_batch(
        self,
        rewards: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, float]:
        """Compute rewards for multiple agents from a simple mapping.

        Example:
            {
                "vehicle_0": {"progress": 2.0, "collision": False, "ttc": 1.3},
                "vehicle_1": {"progress": 1.5, "goal_reached": True},
            }
        """
        results: Dict[str, float] = {}
        for agent_id, payload in rewards.items():
            results[agent_id] = self.compute(agent_id, **payload)
        return results


__all__ = ["RewardCalculator"]
