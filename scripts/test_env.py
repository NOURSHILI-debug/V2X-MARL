"""
Sanity tests for the env skeleton. Run with the CARLA server already started.

  1. 100 ticks without errors
  2. 10 consecutive resets leave no leftover actors
  3. random-action loop: 5 episodes x 50 steps
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
from envs.carla_env import CarlaMultiAgentEnv, N_ACTIONS

env = CarlaMultiAgentEnv(town="Town03", n_agents=2)   # start with 2, then 4

# 1) ticking
env.reset(seed=0)
for _ in range(100):
    env.world.tick()
print("[1] 100 ticks OK")

# 2) clean teardown
for i in range(10):
    env.reset(seed=i)
n_vehicles = len(env.world.get_actors().filter("vehicle.*"))
print(f"[2] vehicles after 10 resets: {n_vehicles} (expected {env.n_agents})")
assert n_vehicles == env.n_agents

# 3) random-action loop
for ep in range(5):
    obs, info = env.reset(seed=100 + ep)
    for t in range(50):
        actions = {a: random.randrange(N_ACTIONS) for a in env.agents}
        obs, rew, term, trunc, info = env.step(actions)
    print(f"[3] episode {ep} done, steps={info['step']}")

env.close()
print("ALL OK")