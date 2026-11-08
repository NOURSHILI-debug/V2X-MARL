"""Inspect keep-policy stopping and route progress on a live CARLA server."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.carla_env import CarlaMultiAgentEnv, GOAL_RADIUS_M, KEEP


def _vehicle_within_10m_ahead(env, agent_id):
    ego = env.vehicles[agent_id]
    transform = ego.get_transform()
    forward = transform.get_forward_vector()
    origin = transform.location
    for actor in env.world.get_actors().filter("vehicle.*"):
        if actor.id == ego.id:
            continue
        location = actor.get_location()
        dx, dy = location.x - origin.x, location.y - origin.y
        longitudinal = dx * forward.x + dy * forward.y
        lateral_sq = dx * dx + dy * dy - longitudinal * longitudinal
        if 0.0 < longitudinal <= 10.0 and lateral_sq <= 3.5 * 3.5:
            return True
    return False


def _agent_state(env, agent_id):
    vehicle = env.vehicles[agent_id]
    location = vehicle.get_location()
    speed = vehicle.get_velocity().length()
    distance_to_goal = location.distance(env.goal[agent_id])
    distance_to_last = location.distance(env.route[agent_id][-1].transform.location)
    light_state = vehicle.get_traffic_light_state()
    at_light = vehicle.is_at_traffic_light()
    lead = _vehicle_within_10m_ahead(env, agent_id)
    route_applied = env.route_applied[agent_id]
    return {
        "speed": speed,
        "distance_to_goal": distance_to_goal,
        "distance_to_last_route_waypoint": distance_to_last,
        "traffic_light_state": str(light_state),
        "is_at_traffic_light": bool(at_light),
        "vehicle_within_10m_ahead": lead,
        "route_applied": bool(route_applied),
    }


def main():
    env = CarlaMultiAgentEnv(town="Town03", n_agents=4)
    ever_within_goal_radius = dict.fromkeys(env.agents, False)
    first_slow_step = {}
    try:
        env.reset(seed=0)
        for step in range(1, 301):
            _, _, terminated, truncated, _ = env.step(
                {agent_id: KEEP for agent_id in env.agents}
            )
            for agent_id in env.agents:
                state = _agent_state(env, agent_id)
                if state["distance_to_goal"] < GOAL_RADIUS_M:
                    ever_within_goal_radius[agent_id] = True
                if state["speed"] < 0.5 and agent_id not in first_slow_step:
                    first_slow_step[agent_id] = step
                    print(
                        f"FIRST_SLOW agent={agent_id} step={step} state={state}",
                        flush=True,
                    )
                if step % 10 == 0:
                    print(f"step={step} agent={agent_id} state={state}", flush=True)
            if any(terminated.values()) or any(truncated.values()):
                break

        route_lengths = {}
        for agent_id in env.agents:
            route = env.route[agent_id]
            route_lengths[agent_id] = sum(
                route[index].transform.location.distance(
                    route[index + 1].transform.location
                )
                for index in range(len(route) - 1)
            )
        print(f"goal_radius_m={GOAL_RADIUS_M}")
        for agent_id, route_length in route_lengths.items():
            print(f"agent={agent_id} route_length_m={route_length:.3f}")
        print(
            "agents_ever_within_6m_of_goal="
            f"{sum(ever_within_goal_radius.values())}/{len(env.agents)}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
