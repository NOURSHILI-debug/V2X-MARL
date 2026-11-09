"""CARLA-independent episode metric calculations."""

from __future__ import annotations

import math

TTC_CAP_SECONDS = 10.0
DECISION_DT_SECONDS = 0.2
MAX_LONGITUDINAL_ACCELERATION = 8.0  # m/s^2, comfort input bound
MAX_LONGITUDINAL_JERK = 20.0  # m/s^3, comfort input bound


def longitudinal_ttc(
    gap_m: float | None,
    closing_speed_mps: float | None,
    cap_seconds: float = TTC_CAP_SECONDS,
) -> float:
    """Return capped TTC for a lead vehicle, or the cap if none is closing."""
    if not math.isfinite(cap_seconds) or cap_seconds <= 0:
        raise ValueError("cap_seconds must be finite and positive")
    if gap_m is None or closing_speed_mps is None:
        return float(cap_seconds)
    gap, closing = float(gap_m), float(closing_speed_mps)
    if not math.isfinite(gap) or not math.isfinite(closing):
        return float(cap_seconds)
    if gap < 0:
        return 0.0
    if closing <= 0:
        return float(cap_seconds)
    return min(float(cap_seconds), gap / closing)


def jerk_magnitude(
    acceleration: float,
    previous_acceleration: float | None,
    decision_dt_seconds: float,
) -> float:
    """Return absolute longitudinal jerk in m/s^3 for one decision interval."""
    accel = float(acceleration)
    dt = float(decision_dt_seconds)
    if not math.isfinite(accel) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("acceleration must be finite and dt positive")
    if previous_acceleration is None:
        return 0.0
    previous = float(previous_acceleration)
    if not math.isfinite(previous):
        raise ValueError("previous_acceleration must be finite")
    return abs(accel - previous) / dt


def longitudinal_dynamics(
    velocity: tuple[float, float],
    previous_velocity: tuple[float, float] | None,
    previous_acceleration: float | None,
    yaw_radians: float,
    dt_seconds: float = DECISION_DT_SECONDS,
) -> tuple[float, float]:
    """Return clipped forward acceleration (m/s^2) and jerk (m/s^3).

    Acceleration comes from the forward projection of the velocity difference
    over one decision interval. The first sample has no history and returns
    zero for both values. Comfort inputs are bounded at 8 m/s^2 and 20 m/s^3.
    """
    vx, vy = map(float, velocity)
    yaw, dt = float(yaw_radians), float(dt_seconds)
    if not all(math.isfinite(value) for value in (vx, vy, yaw, dt)) or dt <= 0:
        raise ValueError("velocity, yaw, and positive dt must be finite")
    if previous_velocity is None:
        return 0.0, 0.0
    previous_vx, previous_vy = map(float, previous_velocity)
    if not math.isfinite(previous_vx) or not math.isfinite(previous_vy):
        raise ValueError("previous_velocity must be finite")
    forward_x, forward_y = math.cos(yaw), math.sin(yaw)
    raw_acceleration = (
        (vx - previous_vx) * forward_x + (vy - previous_vy) * forward_y
    ) / dt
    acceleration = max(
        -MAX_LONGITUDINAL_ACCELERATION,
        min(MAX_LONGITUDINAL_ACCELERATION, raw_acceleration),
    )
    jerk = min(
        MAX_LONGITUDINAL_JERK,
        jerk_magnitude(acceleration, previous_acceleration, dt),
    )
    return float(acceleration), float(jerk)


__all__ = [
    "DECISION_DT_SECONDS",
    "MAX_LONGITUDINAL_ACCELERATION",
    "MAX_LONGITUDINAL_JERK",
    "TTC_CAP_SECONDS",
    "jerk_magnitude",
    "longitudinal_dynamics",
    "longitudinal_ttc",
]
