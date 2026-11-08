"""CARLA independent adapter for per-decision reward inputs."""

from __future__ import annotations

import math

from envs.metrics import MAX_LONGITUDINAL_ACCELERATION, MAX_LONGITUDINAL_JERK


def compute_agent_step_reward(
    calculator,
    agent_id,
    *,
    distance_travelled,
    collision=False,
    ttc=None,
    goal_reached=False,
    acceleration=None,
    jerk=None,
    speed=0.0,
    at_red_light=False,
    finished=False,
    v_max=30.0,
    dt=0.2,
):
    """Return scalar and terms; clamp comfort inputs to 8 m/s^2 and 20 m/s^3."""
    if finished:
        terms = {name: 0.0 for name in
                 ("progress", "collision", "safety", "comfort", "goal", "idle")}
        return 0.0, terms
    distance = float(distance_travelled)
    v_max, dt = float(v_max), float(dt)
    if not math.isfinite(distance) or not math.isfinite(v_max) or not math.isfinite(dt):
        raise ValueError("distance_travelled, v_max, and dt must be finite")
    if v_max <= 0 or dt <= 0:
        raise ValueError("v_max and dt must be positive")
    progress = max(0.0, distance) / (v_max * dt)
    if acceleration is not None:
        acceleration = float(acceleration)
        if not math.isfinite(acceleration):
            raise ValueError("acceleration must be finite")
        acceleration = max(
            -MAX_LONGITUDINAL_ACCELERATION,
            min(MAX_LONGITUDINAL_ACCELERATION, acceleration),
        )
    if jerk is not None:
        jerk = float(jerk)
        if not math.isfinite(jerk):
            raise ValueError("jerk must be finite")
        jerk = min(MAX_LONGITUDINAL_JERK, abs(jerk))
    terms = calculator.compute_terms(
        agent_id,
        progress=progress,
        collision=collision,
        ttc=ttc,
        goal_reached=goal_reached,
        acceleration=acceleration,
        jerk=jerk,
        idle=float(speed) < 1.0 and not at_red_light,
    )
    reward = float(sum(terms.values()))
    if not math.isfinite(reward):
        raise ValueError("Reward calculation produced a non-finite value")
    return reward, terms


__all__ = ["compute_agent_step_reward"]
