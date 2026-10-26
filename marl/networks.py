"""Shared-weight per-agent Q network (used by VDN now, QMIX later)."""
import torch
import torch.nn as nn


class AgentQNet(nn.Module):
    """Input: obs [B, N, obs_dim]  ->  Q-values [B, N, n_actions].
    One network shared by all agents; a one-hot agent id is appended to obs."""

    def __init__(self, obs_dim, n_actions, n_agents, hidden=128):
        super().__init__()
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_agents, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        B = obs.shape[0]
        ids = torch.eye(self.n_agents, device=obs.device).unsqueeze(0).expand(B, -1, -1)
        return self.net(torch.cat([obs, ids], dim=-1))