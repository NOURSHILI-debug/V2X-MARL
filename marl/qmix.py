"""QMIX with shared per-agent utilities and a state-conditioned monotonic mixer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedAgentQNetwork(nn.Module):
    """Map one observation and an agent identity to that agent's action values."""

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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        agent_ids = torch.eye(
            self.n_agents,
            dtype=observations.dtype,
            device=observations.device,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        return self.network(torch.cat((observations, agent_ids), dim=-1))


class QMixer(nn.Module):
    """Monotonic state-conditioned mixer for per-agent selected Q-values."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_dim: int = 32,
        hypernet_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_dim = mixing_dim
        self.state_norm = nn.LayerNorm(state_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_dim),
            nn.ReLU(),
            nn.Linear(hypernet_dim, n_agents * mixing_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_dim),
            nn.ReLU(),
            nn.Linear(hypernet_dim, mixing_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, 1),
        )

    def forward(
        self,
        agent_q_values: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        states = self.state_norm(states)
        batch_size = agent_q_values.shape[0]
        w1 = F.softplus(self.hyper_w1(states)).view(
            batch_size,
            self.n_agents,
            self.mixing_dim,
        )
        b1 = self.hyper_b1(states).view(batch_size, 1, self.mixing_dim)
        hidden = F.elu(
            torch.bmm(agent_q_values.view(batch_size, 1, self.n_agents), w1) + b1
        )
        w2 = F.softplus(self.hyper_w2(states)).view(
            batch_size,
            self.mixing_dim,
            1,
        )
        baseline = self.value(states).view(batch_size, 1, 1)
        return (torch.bmm(hidden, w2) + baseline).view(batch_size)


class QMIXReplayBuffer:
    """Circular replay buffer storing local observations and global states."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
    ) -> None:
        if min(capacity, n_agents, obs_dim, state_dim, n_actions) <= 0:
            raise ValueError("Replay buffer dimensions and capacity must be positive.")
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.size = 0
        self.pointer = 0
        self.observations = np.zeros(
            (capacity, n_agents, obs_dim), dtype=np.float32
        )
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.next_states = np.zeros_like(self.states)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.next_action_masks = np.ones(
            (capacity, n_agents, n_actions), dtype=np.float32
        )

    def add(
        self,
        observations: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_observations: np.ndarray,
        next_state: np.ndarray,
        terminated: bool,
        next_action_masks: np.ndarray,
    ) -> None:
        expected_shapes = (
            (observations, (self.n_agents, self.obs_dim), "observations"),
            (state, (self.state_dim,), "state"),
            (actions, (self.n_agents,), "actions"),
            (next_observations, (self.n_agents, self.obs_dim), "next_observations"),
            (next_state, (self.state_dim,), "next_state"),
            (
                next_action_masks,
                (self.n_agents, self.n_actions),
                "next_action_masks",
            ),
        )
        for value, expected, name in expected_shapes:
            if np.asarray(value).shape != expected:
                raise ValueError(f"{name} must have shape {expected}.")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite.")
        if np.any(np.asarray(actions) < 0) or np.any(
            np.asarray(actions) >= self.n_actions
        ):
            raise ValueError("actions contain an out-of-range action.")
        if np.any(~np.asarray(next_action_masks, dtype=bool).any(axis=1)):
            raise ValueError("Every agent must have at least one valid next action.")

        index = self.pointer
        self.observations[index] = observations
        self.states[index] = state
        self.actions[index] = actions
        self.rewards[index] = reward
        self.next_observations[index] = next_observations
        self.next_states[index] = next_state
        self.terminated[index] = float(terminated)
        self.next_action_masks[index] = next_action_masks
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, ...]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        indices = np.random.randint(0, self.size, size=batch_size)

        def tensor(values: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.as_tensor(values[indices], dtype=dtype, device=device)

        return (
            tensor(self.observations),
            tensor(self.states),
            tensor(self.actions, torch.int64),
            tensor(self.rewards),
            tensor(self.next_observations),
            tensor(self.next_states),
            tensor(self.terminated),
            tensor(self.next_action_masks),
        )

    def __len__(self) -> int:
        return self.size


class QMIX:
    """QMIX learner with parameter-shared decentralized agent networks.

    ``act`` only uses local observations and action masks. Global state is used
    exclusively by the mixer during training and target computation.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int = 6,
        *,
        hidden_dim: int = 128,
        mixing_dim: int = 32,
        hypernet_dim: int = 64,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        reward_scale: float = 1.0,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        if min(n_agents, obs_dim, state_dim, n_actions, hidden_dim) <= 0:
            raise ValueError("Network dimensions must be positive.")
        if n_actions != 6:
            raise ValueError("The V2X specification requires exactly six actions.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1].")
        if reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.mixing_dim = mixing_dim
        self.hypernet_dim = hypernet_dim
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        self.agent_q = SharedAgentQNetwork(
            obs_dim, n_actions, n_agents, hidden_dim
        ).to(self.device)
        self.mixer = QMixer(
            n_agents, state_dim, mixing_dim, hypernet_dim
        ).to(self.device)
        self.target_agent_q = SharedAgentQNetwork(
            obs_dim, n_actions, n_agents, hidden_dim
        ).to(self.device)
        self.target_mixer = QMixer(
            n_agents, state_dim, mixing_dim, hypernet_dim
        ).to(self.device)
        self.target_agent_q.load_state_dict(self.agent_q.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_agent_q.eval()
        self.target_mixer.eval()
        self.optimizer = torch.optim.Adam(
            list(self.agent_q.parameters()) + list(self.mixer.parameters()),
            lr=learning_rate,
        )

    @torch.no_grad()
    def act(
        self,
        observations: np.ndarray,
        action_masks: np.ndarray,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """Choose masked epsilon-greedy actions from local observations only."""
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
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")

        obs_tensor = torch.as_tensor(
            observations[None],
            dtype=torch.float32,
            device=self.device,
        )
        q_values = self.agent_q(obs_tensor)[0].cpu().numpy()
        actions = np.empty(self.n_agents, dtype=np.int64)
        for agent_index in range(self.n_agents):
            valid_actions = np.flatnonzero(action_masks[agent_index])
            if self.rng.random() < epsilon:
                actions[agent_index] = self.rng.choice(valid_actions)
            else:
                masked_q = np.where(action_masks[agent_index], q_values[agent_index], -np.inf)
                actions[agent_index] = int(np.argmax(masked_q))
        return actions

    def update(self, batch: tuple[torch.Tensor, ...]) -> float:
        (
            observations,
            states,
            actions,
            rewards,
            next_observations,
            next_states,
            terminated,
            next_action_masks,
        ) = batch
        if torch.any(~next_action_masks.bool().any(dim=-1)):
            raise ValueError("Every agent must have at least one valid next action.")

        chosen_agent_q = self.agent_q(observations).gather(
            dim=2,
            index=actions.unsqueeze(-1),
        ).squeeze(-1)
        total_q = self.mixer(chosen_agent_q, states)

        with torch.no_grad():
            next_online_q = self.agent_q(next_observations)
            next_online_q = next_online_q.masked_fill(
                ~next_action_masks.bool(),
                torch.finfo(next_online_q.dtype).min,
            )
            next_actions = next_online_q.argmax(dim=-1, keepdim=True)
            next_target_q = self.target_agent_q(next_observations).gather(
                dim=2,
                index=next_actions,
            ).squeeze(-1)
            next_total_q = self.target_mixer(next_target_q, next_states)
            targets = (
                rewards * self.reward_scale
                + self.gamma * (1.0 - terminated) * next_total_q
            )

        loss = F.smooth_l1_loss(total_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent_q.parameters()) + list(self.mixer.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()
        self._soft_update(self.agent_q, self.target_agent_q)
        self._soft_update(self.mixer, self.target_mixer)
        return float(loss.detach().cpu())

    @torch.no_grad()
    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        for online_parameter, target_parameter in zip(
            online.parameters(),
            target.parameters(),
            strict=True,
        ):
            target_parameter.lerp_(online_parameter, self.tau)

    def parameter_summary(self) -> dict[str, float]:
        """Return selected online parameter norms for training diagnostics."""
        return {
            "agent_first_layer_norm": float(
                self.agent_q.network[0].weight.detach().norm().cpu()
            ),
            "mixer_hyper_w1_norm": float(
                self.mixer.hyper_w1[-1].weight.detach().norm().cpu()
            ),
            "mixer_hyper_w2_norm": float(
                self.mixer.hyper_w2[-1].weight.detach().norm().cpu()
            ),
        }

    def _checkpoint(self) -> dict[str, Any]:
        return {
            "algorithm": "qmix",
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "hidden_dim": self.hidden_dim,
            "mixing_dim": self.mixing_dim,
            "hypernet_dim": self.hypernet_dim,
            "gamma": self.gamma,
            "tau": self.tau,
            "reward_scale": self.reward_scale,
            "agent_q": self.agent_q.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_agent_q": self.target_agent_q.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint(), checkpoint_path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("algorithm") != "qmix":
            raise ValueError("Checkpoint is not a QMIX checkpoint.")
        for key in ("n_agents", "obs_dim", "state_dim", "n_actions"):
            if checkpoint.get(key) != getattr(self, key):
                raise ValueError(
                    f"Checkpoint {key}={checkpoint.get(key)} does not match "
                    f"learner {key}={getattr(self, key)}."
                )
        self.agent_q.load_state_dict(checkpoint["agent_q"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_agent_q.load_state_dict(checkpoint["target_agent_q"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "QMIX":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("algorithm") != "qmix":
            raise ValueError("Checkpoint is not a QMIX checkpoint.")
        learner = cls(
            n_agents=checkpoint["n_agents"],
            obs_dim=checkpoint["obs_dim"],
            state_dim=checkpoint["state_dim"],
            n_actions=checkpoint["n_actions"],
            hidden_dim=checkpoint["hidden_dim"],
            mixing_dim=checkpoint["mixing_dim"],
            hypernet_dim=checkpoint["hypernet_dim"],
            gamma=checkpoint["gamma"],
            tau=checkpoint["tau"],
            reward_scale=checkpoint["reward_scale"],
            device=device,
        )
        learner.load(path)
        return learner


__all__ = ["QMIX", "QMIXReplayBuffer", "QMixer", "SharedAgentQNetwork"]
