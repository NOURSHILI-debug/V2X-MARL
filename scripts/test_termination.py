"""CARLA-server checks for shared termination and actor cleanup.

Run with CARLA 0.9.16 already running:
    python scripts/test_termination.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.carla_env import BRAKE, ACCEL, CarlaMultiAgentEnv, MAX_STEPS


def _with_env(check):
    env = CarlaMultiAgentEnv(town="Town03", n_agents=2)
    try:
        check(env)
    finally:
        env.close()


def _find_aligned_waypoints(env):
    carla_map = env.world.get_map()
    for transform in carla_map.get_spawn_points():
        front = carla_map.get_waypoint(transform.location)
        if front is None:
            continue
        for rear in front.previous(20.0):
            if rear.road_id == front.road_id and rear.lane_id == front.lane_id:
                return front, rear
    raise RuntimeError("Could not find two waypoints on the same lane.")


def _setup_forced_collision(env, seed=101):
    env.reset(seed=seed)
    front_id, rear_id = env.agents
    front_wp, rear_wp = _find_aligned_waypoints(env)
    front = env.vehicles[front_id]
    rear = env.vehicles[rear_id]
    front.set_transform(front_wp.transform)
    rear.set_transform(rear_wp.transform)

    front.set_autopilot(False, env.tm_port)
    front_control = front.get_control()
    front_control.throttle = 0.0
    front_control.brake = 1.0
    front_control.hand_brake = True
    front.apply_control(front_control)

    rear.set_autopilot(True, env.tm_port)
    env.tm.ignore_vehicles_percentage(rear, 100)
    rear_route = env._plan_route(rear_wp, random.Random(seed))
    env.tm.set_path(rear, [waypoint.transform.location for waypoint in rear_route[1:]])
    return front_id, rear_id


def _forced_collision(env, seed=101):
    front_id, rear_id = _setup_forced_collision(env, seed)

    for _ in range(MAX_STEPS):
        _, _, terminated, truncated, info = env.step(
            {front_id: BRAKE, rear_id: ACCEL}
        )
        if info["collision"]:
            assert all(terminated.values()), "collision did not terminate every agent"
            assert not any(truncated.values()), "collision was reported as truncation"
            return
        if any(terminated.values()) or any(truncated.values()):
            break
    raise AssertionError("Expected a forced collision within 300 decision steps.")


def _stuck_or_timeout(env):
    env.reset(seed=102)
    for agent_id in env.agents:
        location = env.vehicles[agent_id].get_location()
        location.x += 10000.0
        env.goal[agent_id] = location

    for _ in range(MAX_STEPS):
        _, _, terminated, truncated, info = env.step(
            {agent_id: BRAKE for agent_id in env.agents}
        )
        if any(terminated.values()):
            assert all(terminated.values()), "stuck termination was not shared"
            assert info["stuck"], "termination occurred before stuck or timeout"
            return
        if any(truncated.values()):
            assert all(truncated.values()), "timeout truncation was not shared"
            assert info["steps"] == MAX_STEPS
            assert not info["stuck"]
            return
    raise AssertionError("Expected stuck termination or timeout within 300 steps.")


def _short_goal_route(env):
    env.reset(seed=103)
    for agent_id in env.agents:
        vehicle = env.vehicles[agent_id]
        vehicle.set_autopilot(False, env.tm_port)
        control = vehicle.get_control()
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = True
        vehicle.apply_control(control)
        waypoint = env.world.get_map().get_waypoint(vehicle.get_location())
        env.route[agent_id] = [waypoint]
        env.goal[agent_id] = vehicle.get_location()

    _, _, terminated, truncated, info = env.step(
        {agent_id: BRAKE for agent_id in env.agents}
    )
    assert all(terminated.values()), "all-at-goal did not terminate every agent"
    assert not any(truncated.values())
    assert info["goal_reached"] and info["success"]


def _ten_resets_leave_only_controlled_vehicles(env):
    for seed in range(10):
        env.reset(seed=seed)
        vehicle_count = len(env.world.get_actors().filter("vehicle.*"))
        assert vehicle_count == env.n_agents, (
            f"reset {seed}: found {vehicle_count} vehicles, expected {env.n_agents}"
        )


def _run_check(name, check):
    try:
        _with_env(check)
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        return False
    print(f"PASS {name}")
    return True


def main():
    checks = [
        ("forced collision terminates all agents", _forced_collision),
        ("stuck or timeout termination", _stuck_or_timeout),
        ("short route goal termination", _short_goal_route),
        ("ten resets leave exactly two vehicles", _ten_resets_leave_only_controlled_vehicles),
    ]
    results = [_run_check(name, check) for name, check in checks]
    if all(results):
        print("ALL OK")
        return 0
    print("FAIL: one or more termination checks failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
