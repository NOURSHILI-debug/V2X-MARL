"""VDN: Q_tot = sum_i Q_i(o_i, a_i). Trained end to end on the team reward."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from marl.networks import AgentQNet


class VDN:
    def __init__(self, n_agents, obs_dim, n_actions, lr=5e-4, gamma=0.99,
                 hidden=128, target_update=200, device="cpu"):
        self.n_agents, self.n_actions = n_agents, n_actions
        self.gamma, self.target_update, self.device = gamma, target_update, device
        self.q = AgentQNet(obs_dim, n_actions, n_agents, hidden).to(device)
        self.target = AgentQNet(obs_dim, n_actions, n_agents, hidden).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.updates = 0

    @torch.no_grad()
    def act(self, obs, masks, epsilon=0.0):
        """obs [N, obs_dim] np, masks [N, n_actions] bool np -> actions [N] np"""
        q = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None])[0]
        m = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        q = q.masked_fill(~m, -1e9)
        greedy = q.argmax(dim=-1).cpu().numpy()
        actions = greedy.copy()
        for i in range(self.n_agents):
            if np.random.rand() < epsilon:
                actions[i] = np.random.choice(np.flatnonzero(masks[i]))
        return actions

    def update(self, batch):
        obs, act, rew, next_obs, done, next_mask = batch
        q_i = self.q(obs).gather(2, act.unsqueeze(-1)).squeeze(-1)      # [B, N]
        q_tot = q_i.sum(dim=1)                                          # [B]
        with torch.no_grad():
            nq = self.target(next_obs).masked_fill(next_mask == 0, -1e9)
            nq_tot = nq.max(dim=2)[0].sum(dim=1)
            target = rew + self.gamma * (1.0 - done) * nq_tot
        loss = F.smooth_l1_loss(q_tot, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())
        return loss.item()

    def save(self, path):
        torch.save(self.q.state_dict(), path)