"""Simple circular replay buffer for cooperative multi-agent transitions."""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, n_agents, obs_dim, n_actions):
        self.cap, self.size, self.ptr = capacity, 0, 0
        self.obs = np.zeros((capacity, n_agents, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), np.float32)
        self.act = np.zeros((capacity, n_agents), np.int64)
        self.rew = np.zeros(capacity, np.float32)          # team reward
        self.done = np.zeros(capacity, np.float32)         # true terminal only
        self.next_mask = np.ones((capacity, n_agents, n_actions), np.float32)

    def add(self, obs, act, rew, next_obs, done, next_mask):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.done[i], self.next_mask[i] = next_obs, float(done), next_mask
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size, device="cpu"):
        idx = np.random.randint(0, self.size, size=batch_size)
        f = lambda x, dt=torch.float32: torch.as_tensor(x[idx], dtype=dt, device=device)
        return (f(self.obs), f(self.act, torch.int64), f(self.rew),
                f(self.next_obs), f(self.done), f(self.next_mask))

    def __len__(self):
        return self.size