import sys, time
sys.path.insert(0, '.')
import numpy as np
from marl.qmix import QMIX
from envs.carla_env import CarlaMultiAgentEnv

N_STEPS = 100
env = CarlaMultiAgentEnv(town='Town03', n_agents=4)
ok = True
try:
    model = QMIX(n_agents=4, obs_dim=56, state_dim=24, n_actions=6, seed=0)
    observations, info = env.reset(seed=0)
    agents = list(env.agents)
    assert len(agents) == 4, agents
    t0 = time.time()
    steps = 0
    for _ in range(N_STEPS):
        obs = np.stack([np.asarray(observations[a], np.float32) for a in agents])
        mb = env.action_masks()
        masks = np.stack([np.asarray(mb[a], bool) for a in agents])
        state = np.asarray(env.state(), np.float32)
        assert obs.shape == (4, 56), obs.shape
        assert masks.shape == (4, 6), masks.shape
        assert state.shape == (24,), state.shape
        assert np.isfinite(obs).all() and np.isfinite(state).all()
        assert masks.any(axis=1).all(), "an agent has no valid action"
        actions = model.act(obs, masks, epsilon=0.0 if steps % 2 else 1.0)
        for i in range(4):
            assert masks[i, int(actions[i])], f"masked action chosen: agent {i}"
        observations, rewards, term, trunc, info = env.step(
            {a: int(actions[i]) for i, a in enumerate(agents)})
        steps += 1
        if all(term.values()) or any(trunc.values()):
            break
    dt = (time.time() - t0) / max(steps, 1)
    print(f"QMIX CARLA interface OK: steps={steps} sec/step={dt:.4f}")
    print("final info:", {k: v for k, v in info.items() if k != 'per_agent'})
except Exception as e:
    ok = False
    print("INTERFACE TEST FAILED:", repr(e))
    raise
finally:
    env.close()
    print("STATUS:", "COMPLETE" if ok else "INCOMPLETE")