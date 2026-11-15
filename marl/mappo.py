"""MAPPO: shared masked actor with a centralized team-value critic.

The actor is fully decentralized at execution time: it consumes only the local
observation and the agent one-hot identity, and invalid actions are masked
before the softmax. The critic consumes the global state and produces one team
value per timestep; it is used only during training.

This module is env-agnostic: it never imports ``highway_env`` or ``carla``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.normalization import StateNormalizer

NEGATIVE_INF = -1e9


class SharedActor(nn.Module):
    """Shared masked policy: local obs plus an agent one-hot -> action logits."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_agents: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.network = nn.Sequential(
            nn.Linear(obs_dim + n_agents, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        agent_ids = (
            torch.eye(
                self.n_agents,
                dtype=observations.dtype,
                device=observations.device,
            )
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        return self.network(torch.cat((observations, agent_ids), dim=-1))


class CentralCritic(nn.Module):
    """Centralized team-value network used only for training."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation with correct episode boundaries.

    A true termination never bootstraps from ``next_values``. A truncation does
    bootstrap, but the recursion stops at the boundary either way, so separate
    episodes concatenated in one buffer never leak advantages across episodes.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    if not (rewards.shape == values.shape == next_values.shape == terminated.shape == truncated.shape):
        raise ValueError("GAE inputs must all share the same shape.")
    length = rewards.shape[0]
    advantages = np.zeros(length, dtype=np.float64)
    running = 0.0
    for index in reversed(range(length)):
        done = bool(terminated[index] or truncated[index])
        bootstrap = 0.0 if terminated[index] else float(next_values[index])
        delta = (
            float(rewards[index]) + gamma * bootstrap - float(values[index])
        )
        running = delta + gamma * gae_lambda * (0.0 if done else 1.0) * running
        advantages[index] = running
    return advantages, advantages + values


class RolloutBuffer:
    """Fixed-length on-policy rollout storage with episode flags."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
    ) -> None:
        if min(capacity, n_agents, obs_dim, state_dim, n_actions) <= 0:
            raise ValueError("Rollout dimensions and capacity must be positive.")
        self.capacity = int(capacity)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.observations = np.zeros(
            (capacity, n_agents, obs_dim), dtype=np.float32
        )
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.log_probs = np.zeros((capacity, n_agents), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.action_masks = np.ones(
            (capacity, n_agents, n_actions), dtype=bool
        )
        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.size = 0

    def add(
        self,
        observations: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        value: float,
        reward: float,
        action_masks: np.ndarray,
        terminated: bool,
        truncated: bool,
        next_state: np.ndarray,
    ) -> None:
        if self.size >= self.capacity:
            raise ValueError("Rollout buffer is full; call reset() or update().")
        expected = (
            (observations, (self.n_agents, self.obs_dim), "observations"),
            (state, (self.state_dim,), "state"),
            (actions, (self.n_agents,), "actions"),
            (log_probs, (self.n_agents,), "log_probs"),
            (action_masks, (self.n_agents, self.n_actions), "action_masks"),
            (next_state, (self.state_dim,), "next_state"),
        )
        for value_array, shape, name in expected:
            if np.asarray(value_array).shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")
        if not np.isfinite(value) or not np.isfinite(reward):
            raise ValueError("value and reward must be finite.")
        masks = np.asarray(action_masks, dtype=bool)
        if np.any(~masks.any(axis=1)):
            raise ValueError("Every agent must have at least one valid action.")
        if np.any(np.asarray(actions) < 0) or np.any(
            np.asarray(actions) >= self.n_actions
        ):
            raise ValueError("actions contain an out-of-range action.")
        index = self.size
        self.observations[index] = observations
        self.states[index] = state
        self.actions[index] = actions
        self.log_probs[index] = log_probs
        self.values[index] = float(value)
        self.rewards[index] = float(reward)
        self.action_masks[index] = masks
        self.terminated[index] = bool(terminated)
        self.truncated[index] = bool(truncated)
        self.next_states[index] = next_state
        self.size += 1

    def reset(self) -> None:
        self.size = 0

    def compute_gae(
        self,
        gamma: float,
        gae_lambda: float,
        next_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot compute GAE for an empty rollout.")
        next_values = np.asarray(next_values, dtype=np.float64)
        if next_values.shape != (self.size,):
            raise ValueError(f"next_values must have shape ({self.size},).")
        return compute_gae(
            self.rewards[: self.size],
            self.values[: self.size],
            next_values,
            self.terminated[: self.size],
            self.truncated[: self.size],
            gamma,
            gae_lambda,
        )

    def __len__(self) -> int:
        return self.size

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            checkpoint_path,
            observations=self.observations[: self.size],
            states=self.states[: self.size],
            actions=self.actions[: self.size],
            log_probs=self.log_probs[: self.size],
            values=self.values[: self.size],
            rewards=self.rewards[: self.size],
            action_masks=self.action_masks[: self.size],
            terminated=self.terminated[: self.size],
            truncated=self.truncated[: self.size],
            next_states=self.next_states[: self.size],
            __size=np.asarray(self.size),
        )

    def load(self, path: str | Path) -> None:
        with np.load(path) as arrays:
            size = int(arrays["__size"])
            if size > self.capacity:
                raise ValueError("Saved rollout is larger than the buffer capacity.")
            self.observations[:size] = arrays["observations"]
            self.states[:size] = arrays["states"]
            self.actions[:size] = arrays["actions"]
            self.log_probs[:size] = arrays["log_probs"]
            self.values[:size] = arrays["values"]
            self.rewards[:size] = arrays["rewards"]
            self.action_masks[:size] = arrays["action_masks"]
            self.terminated[:size] = arrays["terminated"]
            self.truncated[:size] = arrays["truncated"]
            self.next_states[:size] = arrays["next_states"]
            self.size = size


class MAPPO:
    """Masked shared-actor PPO with a centralized team-value critic."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int = 6,
        *,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        value_clip: float | None = None,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 256,
        reward_scale: float = 1.0,
        state_norm: bool = False,
        state_mean: Any = None,
        state_std: Any = None,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if min(n_agents, obs_dim, state_dim, n_actions, hidden_dim) <= 0:
            raise ValueError("Network dimensions must be positive.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be between 0 and 1.")
        if not 0.0 < clip_ratio < 1.0:
            raise ValueError("clip_ratio must be in (0, 1).")
        if entropy_coef < 0.0 or value_coef < 0.0:
            raise ValueError("entropy_coef and value_coef must be non-negative.")
        if value_clip is not None and value_clip <= 0.0:
            raise ValueError("value_clip must be positive when set.")
        if max_grad_norm <= 0.0 or ppo_epochs <= 0 or minibatch_size <= 0:
            raise ValueError("grad clip, epochs, and minibatch must be positive.")
        if reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.value_clip = None if value_clip is None else float(value_clip)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.reward_scale = float(reward_scale)
        self.state_norm = bool(state_norm)
        self.device = torch.device(device)
        self.seed = seed
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))
        if self.state_norm:
            if state_mean is None or state_std is None:
                raise ValueError(
                    "state_norm=True requires explicit state_mean and state_std."
                )
            self.state_normalizer: StateNormalizer | None = StateNormalizer(
                state_dim,
                mean=state_mean,
                std=state_std,
            ).to(self.device)
        else:
            self.state_normalizer = None
        self.actor = SharedActor(
            obs_dim, n_actions, n_agents, hidden_dim
        ).to(self.device)
        self.critic = CentralCritic(state_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate,
        )

    # -- execution ---------------------------------------------------------
    def _masked_logits(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.actor(observations)
        masks = action_masks.to(torch.bool)
        return logits.masked_fill(~masks, NEGATIVE_INF)

    @torch.no_grad()
    def action_probs(
        self,
        observations: np.ndarray,
        action_masks: np.ndarray,
    ) -> torch.Tensor:
        """Return masked action probabilities; invalid actions are exactly zero."""
        observations = np.asarray(observations, dtype=np.float32)
        action_masks = np.asarray(action_masks, dtype=bool)
        if observations.shape[-1] != self.obs_dim:
            raise ValueError(f"observations last dim must be {self.obs_dim}.")
        if action_masks.shape[-1] != self.n_actions:
            raise ValueError(f"action_masks last dim must be {self.n_actions}.")
        squeeze = observations.ndim == 2
        obs_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
        if squeeze:
            obs_tensor = obs_tensor.unsqueeze(0)
            mask_tensor = mask_tensor.unsqueeze(0)
        probs = F.softmax(self._masked_logits(obs_tensor, mask_tensor), dim=-1)
        return probs[0] if squeeze else probs

    @torch.no_grad()
    def act(
        self,
        observations: np.ndarray,
        action_masks: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Choose masked actions from local observations only."""
        observations = np.asarray(observations, dtype=np.float32)
        action_masks = np.asarray(action_masks, dtype=bool)
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
        obs_tensor = torch.as_tensor(
            observations[None], dtype=torch.float32, device=self.device
        )
        mask_tensor = torch.as_tensor(
            action_masks[None], dtype=torch.bool, device=self.device
        )
        probs = F.softmax(self._masked_logits(obs_tensor, mask_tensor), dim=-1)[0]
        actions = np.empty(self.n_agents, dtype=np.int64)
        for agent_index in range(self.n_agents):
            if deterministic:
                actions[agent_index] = int(torch.argmax(probs[agent_index]).item())
            else:
                sample = torch.multinomial(
                    probs[agent_index], 1, generator=self.generator
                )
                actions[agent_index] = int(sample.item())
        return actions

    @torch.no_grad()
    def act_with_info(
        self,
        observations: np.ndarray,
        action_masks: np.ndarray,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return actions, per-agent log-probs, and the team value estimate."""
        observations = np.asarray(observations, dtype=np.float32)
        action_masks = np.asarray(action_masks, dtype=bool)
        actions = self.act(observations, action_masks, deterministic=deterministic)
        obs_tensor = torch.as_tensor(
            observations[None], dtype=torch.float32, device=self.device
        )
        mask_tensor = torch.as_tensor(
            action_masks[None], dtype=torch.bool, device=self.device
        )
        logits = self._masked_logits(obs_tensor, mask_tensor)
        log_probs = F.log_softmax(logits, dim=-1)[0].gather(
            -1,
            torch.as_tensor(actions, device=self.device).unsqueeze(-1),
        ).squeeze(-1)
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},).")
        value = float(self.value(state_array)[0])
        return (
            actions,
            log_probs.cpu().numpy().astype(np.float32),
            value,
        )

    # -- value / diagnostics ----------------------------------------------
    def _normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        if self.state_normalizer is None:
            return states
        return self.state_normalizer(states)

    @torch.no_grad()
    def value(self, states: np.ndarray | torch.Tensor) -> np.ndarray:
        tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        values = self.critic(self._normalize_states(tensor))
        return values.cpu().numpy()

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return log-probs and entropies for a ``[B, N, ...]`` batch."""
        logits = self._masked_logits(observations, action_masks)
        log_probs = F.log_softmax(logits, dim=-1).gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        return log_probs, entropy

    @torch.no_grad()
    def rollout_diagnostics(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Ratio / clip statistics for the rollout before any update."""
        if buffer.size == 0:
            raise ValueError("Cannot diagnose an empty rollout.")
        observations = torch.as_tensor(
            buffer.observations[: buffer.size], dtype=torch.float32, device=self.device
        )
        action_masks = torch.as_tensor(
            buffer.action_masks[: buffer.size], dtype=torch.bool, device=self.device
        )
        actions = torch.as_tensor(
            buffer.actions[: buffer.size], dtype=torch.int64, device=self.device
        )
        old_log_probs = torch.as_tensor(
            buffer.log_probs[: buffer.size], dtype=torch.float32, device=self.device
        )
        log_probs, _ = self.evaluate_actions(observations, action_masks, actions)
        log_ratio = log_probs - old_log_probs
        ratio = log_ratio.exp()
        clip_fraction = (
            (log_ratio.abs() > float(np.log(1.0 + self.clip_ratio)))
            .float()
            .mean()
        )
        return {
            "ratio_mean": float(ratio.mean().item()),
            "ratio_max": float(ratio.max().item()),
            "clip_fraction": float(clip_fraction.item()),
            "approx_kl": float(((ratio - 1.0) - log_ratio).mean().item()),
        }

    # -- learning ----------------------------------------------------------
    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run PPO epochs over one rollout and return training statistics."""
        if buffer.size == 0:
            raise ValueError("Cannot update from an empty rollout.")
        size = buffer.size
        with torch.no_grad():
            next_states = torch.as_tensor(
                buffer.next_states[:size], dtype=torch.float32, device=self.device
            )
            next_values = (
                self.critic(self._normalize_states(next_states)).cpu().numpy()
            )
        scaled_rewards = buffer.rewards[:size].astype(np.float64) * self.reward_scale
        advantages, returns = buffer.compute_gae(
            self.gamma, self.gae_lambda, next_values
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        observations = torch.as_tensor(
            buffer.observations[:size], dtype=torch.float32, device=self.device
        )
        action_masks = torch.as_tensor(
            buffer.action_masks[:size], dtype=torch.bool, device=self.device
        )
        actions = torch.as_tensor(
            buffer.actions[:size], dtype=torch.int64, device=self.device
        )
        old_log_probs = torch.as_tensor(
            buffer.log_probs[:size], dtype=torch.float32, device=self.device
        )
        old_values = torch.as_tensor(
            buffer.values[:size], dtype=torch.float32, device=self.device
        )
        states = torch.as_tensor(
            buffer.states[:size], dtype=torch.float32, device=self.device
        )
        advantage_tensor = torch.as_tensor(
            advantages, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        return_tensor = torch.as_tensor(
            returns, dtype=torch.float32, device=self.device
        )

        batch_size = min(self.minibatch_size, size)
        stats = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "clip_fraction": [],
            "approx_kl": [],
        }
        for _epoch in range(self.ppo_epochs):
            permutation = torch.randperm(size, device=self.device)
            for start in range(0, size, batch_size):
                index = permutation[start : start + batch_size]
                log_probs, entropy = self.evaluate_actions(
                    observations[index],
                    action_masks[index],
                    actions[index],
                )
                ratio = (log_probs - old_log_probs[index]).exp()
                surrogate = torch.min(
                    ratio * advantage_tensor[index],
                    torch.clamp(
                        ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                    )
                    * advantage_tensor[index],
                )
                policy_loss = -surrogate.mean()
                values = self.critic(self._normalize_states(states[index]))
                value_loss = F.mse_loss(values, return_tensor[index])
                if self.value_clip is not None:
                    clipped = old_values[index] + torch.clamp(
                        values - old_values[index],
                        -self.value_clip,
                        self.value_clip,
                    )
                    value_loss = torch.max(
                        value_loss, F.mse_loss(clipped, return_tensor[index])
                    )
                entropy_loss = -self.entropy_coef * entropy.mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + entropy_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=self.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs[index]
                    clip_fraction = (
                        (log_ratio.abs() > float(np.log(1.0 + self.clip_ratio)))
                        .float()
                        .mean()
                    )
                stats["loss"].append(float(loss.detach().cpu()))
                stats["policy_loss"].append(float(policy_loss.detach().cpu()))
                stats["value_loss"].append(float(value_loss.detach().cpu()))
                stats["entropy"].append(float(entropy.mean().detach().cpu()))
                stats["clip_fraction"].append(float(clip_fraction.cpu()))
                stats["approx_kl"].append(
                    float(((ratio - 1.0) - log_ratio).mean().detach().cpu())
                )
        return {key: float(np.mean(values)) for key, values in stats.items()}

    # -- persistence -------------------------------------------------------
    def _checkpoint(self) -> dict[str, Any]:
        return {
            "algorithm": "mappo",
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_ratio": self.clip_ratio,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "value_clip": self.value_clip,
            "max_grad_norm": self.max_grad_norm,
            "ppo_epochs": self.ppo_epochs,
            "minibatch_size": self.minibatch_size,
            "reward_scale": self.reward_scale,
            "state_norm": self.state_norm,
            "state_mean": (
                None
                if self.state_normalizer is None
                else self.state_normalizer.mean.detach().cpu().tolist()
            ),
            "state_std": (
                None
                if self.state_normalizer is None
                else self.state_normalizer.std.detach().cpu().tolist()
            ),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
        }

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint(), checkpoint_path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm") != "mappo":
            raise ValueError("Checkpoint is not a MAPPO checkpoint.")
        for key in ("n_agents", "obs_dim", "state_dim", "n_actions"):
            if checkpoint.get(key) != getattr(self, key):
                raise ValueError(
                    f"Checkpoint {key}={checkpoint.get(key)} does not match "
                    f"learner {key}={getattr(self, key)}."
                )
        if bool(checkpoint.get("state_norm", False)) != self.state_norm:
            raise ValueError("Checkpoint state_norm does not match the learner.")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("generator") is not None:
            self.generator.set_state(checkpoint["generator"])
        serialized_normalizer = checkpoint.get("state_mean")
        if serialized_normalizer is not None:
            if self.state_normalizer is None:
                raise ValueError(
                    "Checkpoint contains state scales but the learner has "
                    "state_norm=False."
                )
            self.state_normalizer.mean.copy_(
                torch.as_tensor(
                    checkpoint["state_mean"], dtype=torch.float32, device=self.device
                )
            )
            self.state_normalizer.std.copy_(
                torch.as_tensor(
                    checkpoint["state_std"], dtype=torch.float32, device=self.device
                )
            )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "MAPPO":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("algorithm") != "mappo":
            raise ValueError("Checkpoint is not a MAPPO checkpoint.")
        learner = cls(
            n_agents=checkpoint["n_agents"],
            obs_dim=checkpoint["obs_dim"],
            state_dim=checkpoint["state_dim"],
            n_actions=checkpoint["n_actions"],
            hidden_dim=checkpoint["hidden_dim"],
            learning_rate=checkpoint["learning_rate"],
            gamma=checkpoint["gamma"],
            gae_lambda=checkpoint["gae_lambda"],
            clip_ratio=checkpoint["clip_ratio"],
            entropy_coef=checkpoint["entropy_coef"],
            value_coef=checkpoint["value_coef"],
            value_clip=checkpoint.get("value_clip"),
            max_grad_norm=checkpoint["max_grad_norm"],
            ppo_epochs=checkpoint["ppo_epochs"],
            minibatch_size=checkpoint["minibatch_size"],
            reward_scale=checkpoint["reward_scale"],
            state_norm=bool(checkpoint.get("state_norm", False)),
            state_mean=checkpoint.get("state_mean"),
            state_std=checkpoint.get("state_std"),
            device=device,
            seed=None,
        )
        learner.load(path)
        return learner


__all__ = [
    "CentralCritic",
    "MAPPO",
    "RolloutBuffer",
    "SharedActor",
    "compute_gae",
]
