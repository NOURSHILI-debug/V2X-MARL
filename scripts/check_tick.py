"""Small synchronous CARLA tick timing check."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.carla_env import CarlaMultiAgentEnv, KEEP


def main():
    env = CarlaMultiAgentEnv(town="Town03", n_agents=2)
    try:
        env.reset(seed=0)
        started = time.perf_counter()
        for _ in range(20):
            env.step({agent_id: KEEP for agent_id in env.agents})
        elapsed = time.perf_counter() - started
        print(f"20 steps OK, sec/step: {elapsed / 20:.4f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
