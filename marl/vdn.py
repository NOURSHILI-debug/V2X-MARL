"""VDN: Q_tot = sum_i Q_i(o_i, a_i). Trained end to end on the team reward.
v2: reward scaling, Double-Q target, soft target update."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from marl.networks import AgentQNet


class VDN:
    def __init__(self, n_agents, obs_dim, n_actions, lr=3e-4, gamma=0.99,
                 hidden=128, tau=0.01, reward_scale=0.1, device="cpu"):
        self.n_agents, self.n_actions = n_agents, n_actions
        self.gamma, self.tau, self.reward_scale, self.device = gamma, tau, reward_scale, device
        self.q = AgentQNet(obs_dim, n_actions, n_agents, hidden).to(device)
        self.target = AgentQNet(obs_dim, n_actions, n_agents, hidden).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, masks, epsilon=0.0):
        """obs [N, obs_dim] np, masks [N, n_actions] bool np -> actions [N] np"""
        q = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None])[0]
        m = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        actions = q.masked_fill(~m, -1e9).argmax(dim=-1).cpu().numpy()
        for i in range(self.n_agents):
            if np.random.rand() < epsilon:
                actions[i] = np.random.choice(np.flatnonzero(masks[i]))
        return actions

    def update(self, batch):
        obs, act, rew, next_obs, done, next_mask = batch
        q_tot = self.q(obs).gather(2, act.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        with torch.no_grad():
            # Double-Q: online net picks the action, target net evaluates it
            next_a = self.q(next_obs).masked_fill(next_mask == 0, -1e9).argmax(2, keepdim=True)
            nq_tot = self.target(next_obs).gather(2, next_a).squeeze(-1).sum(dim=1)
            target = rew * self.reward_scale + self.gamma * (1.0 - done) * nq_tot
        loss = F.smooth_l1_loss(q_tot, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():                      # soft target update
            for p, tp in zip(self.q.parameters(), self.target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return loss.item()

    def save(self, path):
        torch.save(self.q.state_dict(), path)