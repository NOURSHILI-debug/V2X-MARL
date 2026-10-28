"""
CARLA demo for the supervisor meeting (record it with Win+Alt+R or OBS).

Start the server WITH rendering (no -RenderOffScreen), then:
    python scripts\\demo_carla.py
    python scripts\\demo_carla.py --town Town05 --agents 4 --seconds 90
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import carla
from envs.carla_env import (CarlaMultiAgentEnv, KEEP, ACCEL, DECEL, BRAKE,
                            LANE_LEFT, LANE_RIGHT)

p = argparse.ArgumentParser()
p.add_argument("--town", default="Town03")
p.add_argument("--agents", type=int, default=4)
p.add_argument("--seconds", type=int, default=90)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

env = CarlaMultiAgentEnv(town=args.town, n_agents=args.agents)
spectator = env.world.get_spectator()
STEP_REAL_TIME = 0.05 * 4          # one decision step = 0.2 s, so the video looks natural


def scripted(agent_idx, t):
    """Simple scripted behaviour so the video shows all six actions."""
    if t < 15:
        return ACCEL
    if t % 60 == 30 + 5 * agent_idx:
        return LANE_LEFT if agent_idx % 2 == 0 else LANE_RIGHT
    if t % 90 == 70 and agent_idx == 0:
        return BRAKE
    if t % 90 in (75, 76) and agent_idx == 0:
        return ACCEL
    return KEEP


obs, info = env.reset(seed=args.seed)
print(f"Demo: {args.town}, {args.agents} agents, seed {args.seed}. Start recording now.")
time.sleep(3)

n_steps = int(args.seconds / STEP_REAL_TIME)
for t in range(n_steps):
    t0 = time.time()
    actions = {a: scripted(i, t) for i, a in enumerate(env.agents)}
    env.step(actions)

    # chase camera on agent 0 + name tags above each car
    tr = env.vehicles["cav_0"].get_transform()
    fwd = tr.get_forward_vector()
    cam = carla.Location(tr.location.x - 10 * fwd.x, tr.location.y - 10 * fwd.y,
                         tr.location.z + 5)
    spectator.set_transform(carla.Transform(cam, carla.Rotation(pitch=-18, yaw=tr.rotation.yaw)))
    for a, v in env.vehicles.items():
        loc = v.get_location() + carla.Location(z=2.5)
        env.world.debug.draw_string(loc, a, life_time=STEP_REAL_TIME * 1.5,
                                    color=carla.Color(255, 255, 0))

    time.sleep(max(0.0, STEP_REAL_TIME - (time.time() - t0)))

env.close()
print("Demo finished.")