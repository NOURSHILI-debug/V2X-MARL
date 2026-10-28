"""Train VDN on the toy env. Run from repo root:
    python scripts/train_toy.py --episodes 1000
Outputs: results/vdn_toy_log.csv and results/vdn_toy_curve.png
"""
import sys, os, csv, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from envs.toy_env import ToyEnv
from marl.vdn import VDN
from marl.buffers import ReplayBuffer


def env_reset(env, seed):
    """Accept both reset() -> obs_dict and reset() -> (obs_dict, info)."""
    out = env.reset(seed=seed)
    return out[0] if isinstance(out, tuple) else out


def env_step(env, actions):
    """Accept step() -> (obs, rew, term, trunc, info) or (obs, rew, done, info)."""
    out = env.step(actions)
    if len(out) == 5:
        obs, rew, term, trunc, _ = out
    elif len(out) == 4:
        obs, rew, term, _ = out
        trunc = False
    else:
        raise ValueError(f"unexpected step() output of length {len(out)}")
    return obs, rew, term, trunc


def all_true(x):
    return all(x.values()) if isinstance(x, dict) else bool(x)


def team_reward(r):
    return float(np.mean(list(r.values()))) if isinstance(r, dict) else float(r)


def stack(d, agents):
    return np.stack([np.asarray(d[a], dtype=np.float32) for a in agents])


def masks_of(env, agents):
    m = env.action_masks()
    return np.stack([np.asarray(m[a], dtype=bool) for a in agents])


def run_episode(env, agents, policy, seed):
    """policy(obs, masks) -> actions array. Returns list of transitions + return."""
    obs_d = env_reset(env, seed)
    obs, masks = stack(obs_d, agents), masks_of(env, agents)
    transitions, ep_ret = [], 0.0
    while True:
        act = policy(obs, masks)
        obs_d, rew, term, trunc = env_step(env, {a: int(act[i]) for i, a in enumerate(agents)})
        next_obs, next_masks = stack(obs_d, agents), masks_of(env, agents)
        team_r = team_reward(rew)
        terminal = all_true(term)
        ended = terminal or all_true(trunc)
        transitions.append((obs, act, team_r, next_obs, terminal, next_masks))
        ep_ret += team_r
        obs, masks = next_obs, next_masks
        if ended:
            return transitions, ep_ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--reward_scale", type=float, default=0.1)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    env = ToyEnv()
    agents = list(env.agents)
    N = len(agents)
    obs_d = env_reset(env, args.seed)
    obs_dim = len(np.asarray(obs_d[agents[0]]))
    n_actions = len(env.action_masks()[agents[0]])
    print(f"agents={N} obs_dim={obs_dim} n_actions={n_actions}")

    # random baseline (for the demo slide)
    def rand_policy(obs, masks):
        return np.array([np.random.choice(np.flatnonzero(m)) for m in masks])
    rand_ret = np.mean([run_episode(env, agents, rand_policy, 10_000 + i)[1] for i in range(50)])
    print(f"random baseline return: {rand_ret:.2f}")

    agent = VDN(N, obs_dim, n_actions, reward_scale=args.reward_scale)
    buf = ReplayBuffer(50_000, N, obs_dim, n_actions)
    log, evals = [], []
    for ep in range(args.episodes):
        eps = max(0.05, 1.0 - ep / (0.7 * args.episodes))
        trans, ret = run_episode(env, agents, lambda o, m: agent.act(o, m, eps), args.seed + ep)
        for t in trans:
            buf.add(*t)
            if len(buf) >= args.warmup:
                agent.update(buf.sample(args.batch))
        ev = ""
        if ep % args.eval_every == 0:      # greedy evaluation on fixed seeds
            ev = float(np.mean([run_episode(env, agents, lambda o, m: agent.act(o, m, 0.0),
                                            20_000 + i)[1] for i in range(20)]))
            evals.append((ep, ev))
            print(f"ep {ep:5d} | train {ret:8.2f} | greedy eval {ev:8.2f} | eps {eps:.2f}")
        log.append((ep, ret, eps, ev))

    with open(os.path.join(args.out, "vdn_toy_log.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "return", "epsilon", "greedy_eval"]); w.writerows(log)
    agent.save(os.path.join(args.out, "vdn_toy.pt"))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = np.array([x[1] for x in log]); k = 50
    plt.plot(r, alpha=0.25, label="train return (with exploration)")
    plt.plot(np.convolve(r, np.ones(k) / k, mode="valid"), label=f"train moving avg ({k})")
    plt.plot([e[0] for e in evals], [e[1] for e in evals], "g-o", ms=3, label="greedy eval (20 fixed seeds)")
    plt.axhline(rand_ret, color="r", ls="--", label="random baseline")
    plt.xlabel("episode"); plt.ylabel("team return"); plt.title("VDN on toy env")
    plt.legend(); plt.savefig(os.path.join(args.out, "vdn_toy_curve.png"), dpi=150)
    print("saved results/vdn_toy_log.csv, vdn_toy_curve.png")


if __name__ == "__main__":
    main()