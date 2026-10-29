import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def check(env, name):
    out = env.reset(seed=0)
    print(name, "reset returns:", type(out).__name__, "len", len(out) if isinstance(out, tuple) else "-")
    obs = out[0] if isinstance(out, tuple) else out
    print(" agents:", env.agents, "| obs shape:", np.asarray(obs[env.agents[0]]).shape)
    print(" mask shape:", np.asarray(env.action_masks()[env.agents[0]]).shape)
    out = env.step({a: 0 for a in env.agents})
    print(" step returns", len(out), "items | terminated:", type(out[2]).__name__,
          "| truncated:", type(out[3]).__name__)
    print(" state shape:", np.asarray(env.state()).shape)


if "carla" in sys.argv:
    from envs.carla_env import CarlaMultiAgentEnv
    env = CarlaMultiAgentEnv(town="Town03", n_agents=2)
    check(env, "CARLA")
    env.close()
else:
    from envs.toy_env import ToyEnv
    check(ToyEnv(), "TOY")