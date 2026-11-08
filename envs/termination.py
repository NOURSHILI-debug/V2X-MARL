from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


OFF_ROAD_STEP_LIMIT = 10


@dataclass(frozen=True)
class TerminationResult:
    terminated: bool
    truncated: bool
    success: bool
    reason: str | None
    stuck_by_agent: dict[str, bool]


def update_off_road_steps(previous_steps: int, is_off_road: bool) -> int:
    """Return the consecutive off-road count for the current decision step."""
    if previous_steps < 0:
        raise ValueError("previous_steps cannot be negative")
    return previous_steps + 1 if is_off_road else 0


def is_off_road(waypoint: object | None) -> bool:
    """A CARLA map lookup with project_to_road=False returns None off-road."""
    return waypoint is None


def evaluate_termination(
    agents: Sequence[str],
    collided: Mapping[str, bool],
    reached: Mapping[str, bool],
    off_road_steps: Mapping[str, int],
    stuck_steps: Mapping[str, int],
    *,
    step_count: int,
    max_steps: int,
    off_road_step_limit: int = OFF_ROAD_STEP_LIMIT,
    stuck_step_limit: int = 50,
    stuck_grace_steps: int = 0,
) -> TerminationResult:
    """Evaluate shared episode completion in Contract v1 priority order."""
    if not agents:
        raise ValueError("agents cannot be empty")
    expected = set(agents)
    if any(
        set(values) != expected
        for values in (collided, reached, off_road_steps, stuck_steps)
    ):
        raise ValueError("termination inputs must contain exactly the configured agents")
    if (
        step_count < 0
        or max_steps <= 0
        or off_road_step_limit <= 0
        or stuck_step_limit < 0
        or stuck_grace_steps < 0
    ):
        raise ValueError("step limits must be positive and step_count non-negative")
    if any(count < 0 for counts in (off_road_steps, stuck_steps) for count in counts.values()):
        raise ValueError("consecutive step counts cannot be negative")

    stuck_by_agent = {
        agent_id: step_count > stuck_grace_steps and count > stuck_step_limit
        for agent_id, count in stuck_steps.items()
    }

    if any(collided.values()):
        reason = "collision"
    elif all(reached.values()):
        reason = "goal"
    elif any(count >= off_road_step_limit for count in off_road_steps.values()):
        reason = "off_road"
    elif any(stuck_by_agent.values()):
        reason = "stuck"
    elif step_count >= max_steps:
        reason = "timeout"
    else:
        reason = None

    return TerminationResult(
        terminated=reason in {"collision", "goal", "off_road", "stuck"},
        truncated=reason == "timeout",
        success=reason == "goal",
        reason=reason,
        stuck_by_agent=stuck_by_agent,
    )


def build_step_info(
    agents: Sequence[str],
    collided: Mapping[str, bool],
    reached: Mapping[str, bool],
    stuck: Mapping[str, bool],
    off_road: Mapping[str, bool],
    route_applied: Mapping[str, bool],
    *,
    steps: int,
    episode_ended: bool,
    success: bool,
    town: str,
    seed: int,
) -> dict[str, object]:
    """Build per-step info, adding episode metadata only on the final step."""
    per_agent = {
        agent_id: {
            "collision": bool(collided[agent_id]),
            "goal_reached": bool(reached[agent_id]),
            "stuck": bool(stuck[agent_id]),
            "off_road": bool(off_road[agent_id]),
            "route_applied": bool(route_applied[agent_id]),
        }
        for agent_id in agents
    }
    info: dict[str, object] = {
        "steps": steps,
        "collision": bool(any(collided.values())),
        "goal_reached": bool(all(reached.values())),
        "stuck": bool(any(stuck.values())),
        "off_road": bool(any(off_road.values())),
        "route_applied": bool(all(route_applied.values())),
        "per_agent": per_agent,
    }
    if episode_ended:
        info.update({"success": bool(success), "town": town, "seed": seed})
    return info
