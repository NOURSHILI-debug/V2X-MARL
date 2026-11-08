"""CARLA-independent episode metric calculations."""

from __future__ import annotations

import math

TTC_CAP_SECONDS = 10.0


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


__all__ = ["TTC_CAP_SECONDS", "jerk_magnitude", "longitudinal_ttc"]
