import itertools

import numpy as np
import pytest
import torch
import torch.nn as nn

from marl.qmix import QMIX, QMixer
from marl.vdn import VDN


class _ConstantQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))

    def forward(self, observations):
        return self.values.unsqueeze(0).expand(observations.shape[0], -1, -1)


class _ConstantMixer(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, agent_q_values, states):
        return torch.full(
            (agent_q_values.shape[0],),
            self.value,
            dtype=agent_q_values.dtype,
            device=agent_q_values.device,
        )


def _qmix(seed=0, obs_dim=5, state_dim=8, n_actions=6, double_q=False):
    return QMIX(
        n_agents=4,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=16,
        mixing_dim=8,
        hypernet_dim=16,
        double_q=double_q,
        seed=seed,
    )


def _vdn(seed=0, obs_dim=5, n_actions=6, double_q=False):
    return VDN(4, obs_dim, n_actions, hidden=16, double_q=double_q, seed=seed)


def test_mixer_monotonic_when_one_agent_value_increases():
    torch.manual_seed(0)
    mixer = QMixer(n_agents=4, state_dim=8, mixing_dim=8, hypernet_dim=16)

    for _ in range(50):
        q_values = torch.randn(10, 4)
        states = torch.randn(10, 8)
        baseline = mixer(q_values, states)
        agent = int(torch.randint(0, 4, (1,)))
        raised = q_values.clone()
        raised[:, agent] += torch.rand(10) + 1e-3

        torch.testing.assert_close(
            mixer(raised, states) >= baseline - 1e-6,
            torch.ones(10, dtype=torch.bool),
        )


def test_vdn_q_tot_is_sum_of_selected_values():
    learner = _vdn()
    observations = torch.rand(6, 4, 5)
    actions = torch.randint(0, 6, (6, 4))

    selected = learner.q(observations).gather(2, actions.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(learner.q_tot(observations, actions), selected.sum(1))


@pytest.mark.parametrize("kind", ["qmix", "vdn"])
def test_act_never_returns_invalid_action_with_or_without_epsilon(kind):
    rng = np.random.default_rng(0)
    learner = _qmix(seed=0) if kind == "qmix" else _vdn(seed=0)

    for _ in range(100):
        masks = rng.random((4, 6)) > 0.4
        masks[:, 0] = True
        observations = rng.normal(size=(4, 5)).astype(np.float32)
        for epsilon in (0.0, 0.25, 1.0):
            actions = learner.act(observations, masks, epsilon=epsilon)
            assert actions.shape == (4,)
            assert all(masks[index, action] for index, action in enumerate(actions))


@pytest.mark.parametrize("double_q", [False, True])
def test_target_max_ignores_masked_action_with_largest_raw_q(double_q):
    masks = torch.tensor(
        [
            [True, False, True],
            [True, False, True],
        ]
    )
    raw_q = [[0.0, 100.0, 0.0], [0.0, 100.0, 0.0]]
    observations = torch.zeros(1, 2, 4)

    qmix = QMIX(
        n_agents=2,
        obs_dim=4,
        state_dim=8,
        n_actions=3,
        hidden_dim=8,
        mixing_dim=4,
        hypernet_dim=8,
        double_q=double_q,
    )
    qmix.target_agent_q = _ConstantQ(raw_q)
    qmix.agent_q = _ConstantQ(raw_q)
    qmix_actions = qmix.target_actions(observations, masks)

    vdn = VDN(2, 4, 3, hidden=8, double_q=double_q)
    vdn.target = _ConstantQ(raw_q)
    vdn.q = _ConstantQ(raw_q)
    vdn_actions = vdn.target_actions(observations, masks)

    for actions in (qmix_actions, vdn_actions):
        assert actions.tolist() == [[0, 0]]


def test_truncated_transition_bootstraps_but_terminated_does_not():
    next_value = 2.0
    observations = torch.zeros(2, 4, 5)
    states = torch.zeros(2, 8)
    actions = torch.zeros(2, 4, dtype=torch.int64)
    rewards = torch.zeros(2)
    next_observations = torch.zeros(2, 4, 5)
    next_states = torch.zeros(2, 8)
    next_masks = torch.ones(2, 4, 6, dtype=torch.bool)

    qmix = _qmix()
    qmix.target_agent_q = _ConstantQ(np.ones((4, 6)))
    qmix.target_mixer = _ConstantMixer(next_value)
    qmix_batch = (
        observations,
        states,
        actions,
        rewards,
        next_observations,
        next_states,
        torch.tensor([0.0, 1.0]),
        next_masks,
    )
    qmix_targets = qmix.td_target(qmix_batch)
    torch.testing.assert_close(
        qmix_targets,
        torch.tensor([qmix.gamma * next_value, 0.0]),
    )

    vdn = _vdn()
    vdn.target = _ConstantQ(np.ones((4, 6)))
    vdn_batch = (
        observations,
        actions,
        rewards,
        next_observations,
        torch.tensor([0.0, 1.0]),
        next_masks,
    )
    vdn_targets = vdn.td_target(vdn_batch)
    torch.testing.assert_close(
        vdn_targets,
        torch.tensor([vdn.gamma * 4 * 1.0, 0.0]),
    )


def test_checkpoint_round_trip_preserves_q_values(tmp_path):
    torch.manual_seed(0)
    observations = torch.rand(3, 4, 5)
    masks = np.ones((4, 6), dtype=bool)

    qmix = _qmix(seed=1)
    qmix_checkpoint = tmp_path / "qmix.pt"
    qmix.save(qmix_checkpoint)
    loaded_qmix = QMIX.from_checkpoint(qmix_checkpoint)
    torch.testing.assert_close(
        qmix.agent_q(observations),
        loaded_qmix.agent_q(observations),
    )
    np.testing.assert_array_equal(
        qmix.act(observations[0].numpy(), masks),
        loaded_qmix.act(observations[0].numpy(), masks),
    )

    vdn = _vdn(seed=1)
    vdn_checkpoint = tmp_path / "vdn.pt"
    vdn.save(vdn_checkpoint)
    loaded_vdn = VDN.from_checkpoint(vdn_checkpoint)
    torch.testing.assert_close(vdn.q(observations), loaded_vdn.q(observations))
    np.testing.assert_array_equal(
        vdn.act(observations[0].numpy(), masks),
        loaded_vdn.act(observations[0].numpy(), masks),
    )


_JOINT_ACTIONS = np.asarray(list(itertools.product(range(2), repeat=2)))


def _matrix_batch(batch_size, state_dim=2):
    indices = np.random.randint(0, len(_JOINT_ACTIONS), size=batch_size)
    actions = _JOINT_ACTIONS[indices]
    rewards = ((actions[:, 0] == 1) & (actions[:, 1] == 1)).astype(np.float32)
    observations = np.zeros((batch_size, 2, 1), dtype=np.float32)
    states = np.zeros((batch_size, state_dim), dtype=np.float32)
    next_masks = np.ones((batch_size, 2, 2), dtype=bool)
    return actions, rewards, observations, states, next_masks


def _train_matrix_game(algorithm, seed, steps=1200, batch_size=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if algorithm == "qmix":
        learner = QMIX(
            n_agents=2,
            obs_dim=1,
            state_dim=2,
            n_actions=2,
            hidden_dim=32,
            mixing_dim=8,
            hypernet_dim=16,
            learning_rate=5e-3,
            gamma=0.9,
            reward_scale=1.0,
            seed=seed,
        )
    else:
        learner = VDN(
            2,
            1,
            2,
            hidden=32,
            lr=5e-3,
            gamma=0.9,
            reward_scale=1.0,
            seed=seed,
        )

    for _ in range(steps):
        actions, rewards, observations, states, next_masks = _matrix_batch(
            batch_size
        )
        actions_t = torch.as_tensor(actions, dtype=torch.int64)
        rewards_t = torch.as_tensor(rewards)
        obs_t = torch.as_tensor(observations)
        states_t = torch.as_tensor(states)
        next_masks_t = torch.as_tensor(next_masks)
        if algorithm == "qmix":
            batch = (
                obs_t,
                states_t,
                actions_t,
                rewards_t,
                obs_t.clone(),
                states_t.clone(),
                torch.ones(batch_size),
                next_masks_t,
            )
        else:
            batch = (
                obs_t,
                actions_t,
                rewards_t,
                obs_t.clone(),
                torch.ones(batch_size),
                next_masks_t,
            )
        learner.update(batch)
    return learner


@pytest.mark.parametrize("algorithm", ["qmix", "vdn"])
def test_matrix_game_reaches_optimal_joint_action(algorithm):
    learner = _train_matrix_game(algorithm, seed=0)
    observations = np.zeros((2, 1), dtype=np.float32)
    masks = np.ones((2, 2), dtype=bool)

    actions = learner.act(observations, masks, epsilon=0.0)

    assert actions.tolist() == [1, 1]


def test_one_update_with_contract56_obs_dim():
    torch.manual_seed(0)
    batch_size = 8
    n_agents = 4
    obs_dim = 56
    state_dim = 20
    observations = torch.rand(batch_size, n_agents, obs_dim)
    states = torch.rand(batch_size, state_dim)
    actions = torch.randint(0, 6, (batch_size, n_agents))
    rewards = torch.rand(batch_size)
    next_masks = torch.ones(batch_size, n_agents, 6, dtype=torch.bool)

    qmix = QMIX(
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=6,
        hidden_dim=16,
        mixing_dim=8,
        hypernet_dim=16,
        seed=0,
    )
    qmix_loss = qmix.update(
        (
            observations,
            states,
            actions,
            rewards,
            observations.clone(),
            states.clone(),
            torch.zeros(batch_size),
            next_masks,
        )
    )
    assert np.isfinite(qmix_loss)

    vdn = VDN(n_agents, obs_dim, 6, hidden=16, seed=0)
    vdn_loss = vdn.update(
        (
            observations,
            actions,
            rewards,
            observations.clone(),
            torch.zeros(batch_size),
            next_masks,
        )
    )
    assert np.isfinite(vdn_loss)
