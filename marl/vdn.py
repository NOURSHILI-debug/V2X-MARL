"""VDN: Q_tot = sum_i Q_i(o_i, a_i). Trained end to end on the team reward."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.networks import AgentQNet


class VDN:
    def __init__(
        self,
        n_agents,
        obs_dim,
        n_actions,
        lr=3e-4,
        gamma=0.99,
        hidden=128,
        tau=0.01,
        reward_scale=1.0,
        double_q=False,
        device="cpu",
        seed=None,
    ):
        if min(n_agents, obs_dim, n_actions, hidden) <= 0:
            raise ValueError("Network dimensions must be positive.")
        self.n_agents, self.n_actions = n_agents, n_actions
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.lr = lr
        self.gamma, self.tau, self.reward_scale = gamma, tau, reward_scale
        self.double_q = bool(double_q)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.q = AgentQNet(obs_dim, n_actions, n_agents, hidden).to(device)
        self.target = AgentQNet(obs_dim, n_actions, n_agents, hidden).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, masks, epsilon=0.0):
        """obs [N, obs_dim] np, masks [N, n_actions] bool np -> actions [N] np"""
        observations = np.asarray(obs, dtype=np.float32)
        action_masks = np.asarray(masks, dtype=bool)
        if observations.shape != (self.n_agents, self.obs_dim):
            raise ValueError(
                f"observations must have shape {(self.n_agents, self.obs_dim)}."
            )
        if action_masks.shape != (self.n_agents, self.n_actions):
            raise ValueError(
                f"action_masks must have shape {(self.n_agents, self.n_actions)}."
            )
        if np.any(~action_masks.any(axis=1)):
            raise ValueError("Every agent must have at least one valid action.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")
        q = self.q(
            torch.as_tensor(observations[None], dtype=torch.float32, device=self.device)
        )[0].cpu().numpy()
        actions = np.empty(self.n_agents, dtype=np.int64)
        for i in range(self.n_agents):
            valid_actions = np.flatnonzero(action_masks[i])
            if self.rng.random() < epsilon:
                actions[i] = self.rng.choice(valid_actions)
            else:
                masked_q = np.where(action_masks[i], q[i], -np.inf)
                actions[i] = int(np.argmax(masked_q))
        return actions

    def q_tot(self, observations, actions):
        """Return sum_i Q_i(o_i, a_i) for a [B, N, obs_dim] batch."""
        return self.q(observations).gather(
            dim=2,
            index=actions.unsqueeze(-1),
        ).squeeze(-1).sum(dim=1)

    @torch.no_grad()
    def target_actions(self, next_observations, next_action_masks):
        """Return the next actions used by the target computation."""
        target_q = self.target(next_observations)
        if self.double_q:
            selector_q = self.q(next_observations)
        else:
            selector_q = target_q
        selector_q = selector_q.masked_fill(
            ~next_action_masks.bool(),
            -torch.inf,
        )
        return selector_q.argmax(dim=-1)

    @torch.no_grad()
    def td_target(self, batch):
        """Compute the TD target; exposed for correctness tests."""
        _, _, rew, next_obs, done, next_mask = batch
        if torch.any(~next_mask.bool().any(dim=-1)):
            raise ValueError("Every agent must have at least one valid next action.")
        next_a = self.target_actions(next_obs, next_mask)
        nq_tot = self.target(next_obs).gather(
            dim=2,
            index=next_a.unsqueeze(-1),
        ).squeeze(-1).sum(dim=1)
        return rew * self.reward_scale + self.gamma * (1.0 - done) * nq_tot

    def update(self, batch):
        obs, act, rew, next_obs, done, next_mask = batch
        q_tot = self.q_tot(obs, act)
        target = self.td_target(batch)
        loss = F.mse_loss(q_tot, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():  # soft target update
            for p, tp in zip(self.q.parameters(), self.target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return float(loss.detach().cpu())

    def _checkpoint(self) -> dict[str, Any]:
        return {
            "algorithm": "vdn",
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "lr": self.lr,
            "gamma": self.gamma,
            "tau": self.tau,
            "reward_scale": self.reward_scale,
            "double_q": self.double_q,
            "q": self.q.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.opt.state_dict(),
        }

    def save(self, path):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint(), checkpoint_path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm") != "vdn":
            raise ValueError("Checkpoint is not a VDN checkpoint.")
        for key in ("n_agents", "obs_dim", "n_actions"):
            if checkpoint.get(key) != getattr(self, key):
                raise ValueError(
                    f"Checkpoint {key}={checkpoint.get(key)} does not match "
                    f"learner {key}={getattr(self, key)}."
                )
        if "double_q" in checkpoint and bool(checkpoint["double_q"]) != self.double_q:
            raise ValueError(
                f"Checkpoint double_q={checkpoint['double_q']} does not match "
                f"learner double_q={self.double_q}."
            )
        self.q.load_state_dict(checkpoint["q"])
        self.target.load_state_dict(checkpoint["target"])
        if "optimizer" in checkpoint:
            self.opt.load_state_dict(checkpoint["optimizer"])

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("algorithm") != "vdn":
            raise ValueError("Checkpoint is not a VDN checkpoint.")
        learner = cls(
            checkpoint["n_agents"],
            checkpoint["obs_dim"],
            checkpoint["n_actions"],
            lr=checkpoint["lr"],
            gamma=checkpoint["gamma"],
            hidden=checkpoint["hidden"],
            tau=checkpoint["tau"],
            reward_scale=checkpoint["reward_scale"],
            double_q=bool(checkpoint.get("double_q", False)),
            device=device,
        )
        learner.load(path)
        return learner


__all__ = ["VDN"]
