import sys, time
sys.path.insert(0, '.')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from envs.carla_env import CarlaMultiAgentEnv as E
from envs.obs_builder import V_MAX

e = E(town='Town03', n_agents=4)
speeds, t = {}, []
try:
    obs, info = e.reset(seed=0)
    for k in e.agents:
        speeds[k] = []
    cars = [a for a in e.world.get_actors().filter('vehicle.*')]
    spec = e.world.get_spectator()
    for step in range(300):
        obs, r, term, trunc, info = e.step({k: 0 for k in e.agents})
        for k in e.agents:
            speeds[k].append(float(obs[k][0]) * V_MAX)
        t.append(step * 0.2)
        if cars:
            tr = cars[0].get_transform()
            fwd = tr.get_forward_vector()
            tr.location.x -= 10 * fwd.x
            tr.location.y -= 10 * fwd.y
            tr.location.z += 5
            tr.rotation.pitch = -15
            spec.set_transform(tr)
        time.sleep(0.05)
        if all(term.values()) or all(trunc.values()):
            break
    print("final info:", {k: v for k, v in info.items() if k != 'per_agent'})
finally:
    e.close()

plt.figure(figsize=(9, 4))
for k, v in speeds.items():
    plt.plot(t[:len(v)], v, label=str(k))
plt.axhline(9, ls='--', c='grey', label='target 9 m/s')
plt.xlabel('time (s)'); plt.ylabel('speed (m/s)')
plt.title('Keep policy, Town03, seed 0'); plt.legend()
plt.tight_layout(); plt.savefig('results/keep_speed.png', dpi=150)
print("saved results/keep_speed.png")