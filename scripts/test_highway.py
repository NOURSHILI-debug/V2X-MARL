"""Run a visual Highway-Env simulation with four controlled CAVs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.highway_env import HighwayV2XEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--background-vehicles", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    env = HighwayV2XEnv(
        render_mode=None if args.headless else "human",
        background_vehicles=args.background_vehicles,
    )
    try:
        observations, _ = env.reset(seed=args.seed)
        print("Highway-Env visual simulation")
        print("Controlled CAVs:", ", ".join(env.agent_ids))
        print("Background vehicles:", args.background_vehicles)
        print("Simulation: 20 Hz; decisions: 5 Hz")
        print("Seed:", args.seed)
        print(
            "Initial positions:",
            {
                agent: (
                    round(info["x"], 1),
                    round(info["y"], 1),
                )
                for agent, info in _vehicle_info(env).items()
            },
        )
        print("Observation shapes:", {
            agent: observation.shape
            for agent, observation in observations.items()
        })

        for step in range(args.steps):
            actions = {agent_id: 0 for agent_id in env.agent_ids}
            masks = env.action_masks()
            if step % 40 == 10 and masks["cav_0"][4]:
                actions["cav_0"] = 4
            elif step % 40 == 10 and masks["cav_0"][5]:
                actions["cav_0"] = 5
            if step % 40 == 20 and masks["cav_1"][5]:
                actions["cav_1"] = 5
            elif step % 40 == 20 and masks["cav_1"][4]:
                actions["cav_1"] = 4
            if step % 50 == 0:
                actions["cav_2"] = 1

            observations, rewards, terminated, truncated, infos = env.step(actions)
            if step % 10 == 0 or any(terminated.values()) or any(truncated.values()):
                positions = {
                    agent: (
                        round(info["x"], 1),
                        round(info["y"], 1),
                    )
                    for agent, info in infos.items()
                }
                speeds = {
                    agent: round(info["speed_mps"], 1)
                    for agent, info in infos.items()
                }
                print(
                    f"step={step + 1:03d} "
                    f"positions={positions} speeds_mps={speeds}"
                )
            if any(terminated.values()) or any(truncated.values()):
                print("Episode ended.")
                break
            time.sleep(0.03)
    finally:
        env.close()


def _vehicle_info(env: HighwayV2XEnv) -> dict[str, dict[str, float]]:
    return {
        agent_id: {
            "x": float(vehicle.position[0]),
            "y": float(vehicle.position[1]),
        }
        for agent_id, vehicle in zip(
            env.agent_ids,
            env._vehicles(),
            strict=True,
        )
    }


if __name__ == "__main__":
    main()
