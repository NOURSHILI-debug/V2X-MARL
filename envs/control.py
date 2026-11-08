"""CARLA-independent target-speed control calculations."""

from __future__ import annotations

import math

from envs.obs_builder import ACCEL, DECEL


def update_target_speed(
    target_speed: float,
    action: int,
    *,
    speed_step: float,
    max_speed: float,
) -> float:
    """Apply accelerate/decelerate actions and clamp the target speed."""
    values = (target_speed, speed_step, max_speed)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("speed values must be finite")
    if speed_step < 0.0 or max_speed < 0.0:
        raise ValueError("speed_step and max_speed cannot be negative")

    updated = float(target_speed)
    if action == ACCEL:
        updated += speed_step
    elif action == DECEL:
        updated -= speed_step
    return min(max(updated, 0.0), max_speed)


def speed_limit_difference(target_speed_mps: float, speed_limit_kmh: float) -> float:
    """Convert target m/s to Traffic Manager's percentage-below-limit setting."""
    if not math.isfinite(float(target_speed_mps)) or not math.isfinite(float(speed_limit_kmh)):
        raise ValueError("speed values must be finite")
    speed_limit_mps = max(float(speed_limit_kmh) / 3.6, 1e-6)
    return (1.0 - float(target_speed_mps) / speed_limit_mps) * 100.0


__all__ = ["speed_limit_difference", "update_target_speed"]
