"""Inspect keep-policy stopping and route progress on a live CARLA server."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.carla_env import CarlaMultiAgentEnv, GOAL_RADIUS, KEEP

ARRIVAL_COVERAGE = 0.95


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
    route = [wp.transform.location for wp in env.route[agent_id]]
    nearest_distance, nearest_index, covered_distance, route_length = _route_progress(
        location, route
    )
    return {
        "speed": speed,
        "distance_to_goal": distance_to_goal,
        "distance_to_last_route_waypoint": distance_to_last,
        "traffic_light_state": str(light_state),
        "is_at_traffic_light": bool(at_light),
        "vehicle_within_10m_ahead": lead,
        "route_applied": bool(route_applied),
        "distance_to_route_m": nearest_distance,
        "nearest_route_waypoint_index": nearest_index,
        "route_coverage": covered_distance / route_length if route_length else 0.0,
    }


def _route_progress(location, route):
    """Project onto each route segment and return distance, nearest index, progress, total."""
    if len(route) < 2:
        return location.distance(route[0]) if route else float("inf"), 0, 0.0, 0.0
    total_length = sum(route[i].distance(route[i + 1]) for i in range(len(route) - 1))
    best_distance = float("inf")
    best_index = 0
    best_progress = 0.0
    traversed = 0.0
    for index, (start, end) in enumerate(zip(route, route[1:])):
        dx, dy, dz = end.x - start.x, end.y - start.y, end.z - start.z
        segment_sq = dx * dx + dy * dy + dz * dz
        segment = segment_sq ** 0.5
        if segment_sq:
            fraction = max(0.0, min(1.0, (
                (location.x - start.x) * dx
                + (location.y - start.y) * dy
                + (location.z - start.z) * dz
            ) / segment_sq))
        else:
            fraction = 0.0
        projected = type(location)(
            x=start.x + fraction * dx,
            y=start.y + fraction * dy,
            z=start.z + fraction * dz,
        )
        distance = location.distance(projected)
        if distance < best_distance:
            best_distance = distance
            best_index = index if fraction < 0.5 else index + 1
            best_progress = traversed + fraction * segment
        traversed += segment
    return best_distance, best_index, best_progress, total_length


def main():
    env = CarlaMultiAgentEnv(town="Town03", n_agents=4)
    ever_arrived = dict.fromkeys(env.agents, False)
    first_slow_step = {}
    red_start = None
    cav0_red_seconds = 0.0
    try:
        env.reset(seed=0)
        for agent_id in env.agents:
            route = env.route[agent_id]
            route_locations = [wp.transform.location for wp in route]
            route_length = sum(
                route_locations[i].distance(route_locations[i + 1])
                for i in range(len(route_locations) - 1)
            )
            start_goal_distance = route_locations[0].distance(env.goal[agent_id])
            print(
                f"RESET agent={agent_id} route_waypoints={len(route)} "
                f"route_length_m={route_length:.3f} "
                f"euclidean_start_to_goal_m={start_goal_distance:.3f}",
                flush=True,
            )
        for step in range(1, 301):
            _, _, terminated, truncated, _ = env.step(
                {agent_id: KEEP for agent_id in env.agents}
            )
            for agent_id in env.agents:
                state = _agent_state(env, agent_id)
                arrived = (
                    state["route_coverage"] >= ARRIVAL_COVERAGE
                    or state["distance_to_goal"] <= GOAL_RADIUS
                )
                ever_arrived[agent_id] = ever_arrived[agent_id] or arrived
                if agent_id == "cav_0":
                    is_red = state["is_at_traffic_light"] and "Red" in state["traffic_light_state"]
                    if is_red:
                        if red_start is None:
                            red_start = (step - 1) * 0.2
                        cav0_red_seconds = (step * 0.2) - red_start
                    elif red_start is not None:
                        print(
                            f"CAV0_RED_PHASE seconds={cav0_red_seconds:.1f}",
                            flush=True,
                        )
                        red_start = None
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
        if red_start is not None:
            print(f"CAV0_RED_PHASE seconds={cav0_red_seconds:.1f} (ongoing)", flush=True)
        print(f"goal_radius_m={GOAL_RADIUS} diagnostic_arrival_coverage={ARRIVAL_COVERAGE:.0%}")
        for agent_id, route_length in route_lengths.items():
            print(f"agent={agent_id} route_length_m={route_length:.3f}")
        print(
            "agents_arrived_by_coverage_or_radius="
            f"{sum(ever_arrived.values())}/{len(env.agents)}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
