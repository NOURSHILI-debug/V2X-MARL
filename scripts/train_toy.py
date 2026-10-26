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


def stack(d, agents):
    return np.stack([np.asarray(d[a], dtype=np.float32) for a in agents])


def masks_of(env, agents):
    m = env.action_masks()
    return np.stack([np.asarray(m[a], dtype=bool) for a in agents])


def run_episode(env, agents, policy, seed):
    """policy(obs, masks) -> actions array. Returns list of transitions + return."""
    obs_d, _ = env.reset(seed=seed)
    obs, masks = stack(obs_d, agents), masks_of(env, agents)
    transitions, ep_ret = [], 0.0
    while True:
        act = policy(obs, masks)
        obs_d, rew, term, trunc, _ = env.step({a: int(act[i]) for i, a in enumerate(agents)})
        next_obs, next_masks = stack(obs_d, agents), masks_of(env, agents)
        team_r = float(np.mean([rew[a] for a in agents]))
        terminal = all(term[a] for a in agents)
        ended = terminal or all(trunc[a] for a in agents)
        transitions.append((obs, act, team_r, next_obs, terminal, next_masks))
        ep_ret += team_r
        obs, masks = next_obs, next_masks
        if ended:
            return transitions, ep_ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    env = ToyEnv()
    agents = list(env.agents)
    N = len(agents)
    obs_d, _ = env.reset(seed=args.seed)
    obs_dim = len(np.asarray(obs_d[agents[0]]))
    n_actions = len(env.action_masks()[agents[0]])
    print(f"agents={N} obs_dim={obs_dim} n_actions={n_actions}")

    # random baseline (for the demo slide)
    def rand_policy(obs, masks):
        return np.array([np.random.choice(np.flatnonzero(m)) for m in masks])
    rand_ret = np.mean([run_episode(env, agents, rand_policy, 10_000 + i)[1] for i in range(50)])
    print(f"random baseline return: {rand_ret:.2f}")

    agent = VDN(N, obs_dim, n_actions)
    buf = ReplayBuffer(50_000, N, obs_dim, n_actions)
    log = []
    for ep in range(args.episodes):
        eps = max(0.05, 1.0 - ep / (0.5 * args.episodes))
        trans, ret = run_episode(env, agents, lambda o, m: agent.act(o, m, eps), args.seed + ep)
        for t in trans:
            buf.add(*t)
            if len(buf) >= args.warmup:
                agent.update(buf.sample(args.batch))
        log.append((ep, ret, eps))
        if ep % 50 == 0:
            print(f"ep {ep:5d} | return {ret:8.2f} | eps {eps:.2f}")

    with open(os.path.join(args.out, "vdn_toy_log.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["episode", "return", "epsilon"]); w.writerows(log)
    agent.save(os.path.join(args.out, "vdn_toy.pt"))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = np.array([x[1] for x in log]); k = 50
    plt.plot(r, alpha=0.3, label="episode return")
    plt.plot(np.convolve(r, np.ones(k) / k, mode="valid"), label=f"moving avg ({k})")
    plt.axhline(rand_ret, color="r", ls="--", label="random baseline")
    plt.xlabel("episode"); plt.ylabel("team return"); plt.title("VDN on toy env")
    plt.legend(); plt.savefig(os.path.join(args.out, "vdn_toy_curve.png"), dpi=150)
    print("saved results/vdn_toy_log.csv, vdn_toy_curve.png")


if __name__ == "__main__":
    main()