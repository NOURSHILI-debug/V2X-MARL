import numpy as np
import pytest
import torch

from marl.qmix import QMIX, QMIXReplayBuffer, QMixer


def _learner(seed=0):
    return QMIX(
        n_agents=4,
        obs_dim=5,
        state_dim=8,
        hidden_dim=16,
        mixing_dim=8,
        hypernet_dim=16,
        seed=seed,
    )


def test_shared_agent_network_returns_six_values_for_each_agent():
    learner = _learner()

    q_values = learner.agent_q(torch.zeros(3, 4, 5))

    assert q_values.shape == (3, 4, 6)
    assert len({id(parameter) for parameter in learner.agent_q.parameters()}) == len(
        list(learner.agent_q.parameters())
    )


def test_mixer_is_monotonic_in_each_agent_value():
    mixer = QMixer(n_agents=4, state_dim=8, mixing_dim=8, hypernet_dim=16)
    q_values = torch.randn(10, 4, requires_grad=True)
    states = torch.randn(10, 8)

    total_q = mixer(q_values, states)
    gradients = torch.autograd.grad(total_q.sum(), q_values)[0]

    assert total_q.shape == (10,)
    assert torch.all(gradients >= 0)


def test_action_selection_respects_masks_for_greedy_and_random_actions():
    learner = _learner(seed=1)
    observations = np.zeros((4, 5), dtype=np.float32)
    masks = np.array(
        [
            [False, True, False, False, False, False],
            [False, False, False, False, True, False],
            [True, False, True, False, False, False],
            [False, False, False, False, False, True],
        ]
    )

    for epsilon in (0.0, 1.0):
        actions = learner.act(observations, masks, epsilon=epsilon)
        assert actions.shape == (4,)
        assert all(masks[index, action] for index, action in enumerate(actions))


def test_action_selection_rejects_all_invalid_mask():
    learner = _learner()

    with pytest.raises(ValueError, match="at least one valid action"):
        learner.act(np.zeros((4, 5)), np.zeros((4, 6), dtype=bool))


def test_replay_buffer_samples_global_states_and_masked_actions():
    replay = QMIXReplayBuffer(8, 4, 5, 8, 6)
    replay.add(
        np.zeros((4, 5), dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        np.array([0, 1, 2, 3]),
        1.0,
        np.ones((4, 5), dtype=np.float32),
        np.ones(8, dtype=np.float32),
        False,
        np.ones((4, 6), dtype=bool),
    )

    batch = replay.sample(2)

    assert len(batch) == 8
    assert batch[0].shape == (2, 4, 5)
    assert batch[1].shape == (2, 8)
    assert batch[2].shape == (2, 4)
    assert batch[7].shape == (2, 4, 6)


def test_update_changes_online_parameters_and_keeps_loss_finite():
    learner = _learner()
    replay = QMIXReplayBuffer(16, 4, 5, 8, 6)
    for index in range(8):
        replay.add(
            np.full((4, 5), index / 8, dtype=np.float32),
            np.full(8, index / 8, dtype=np.float32),
            np.array([0, 1, 2, 3]),
            float(index),
            np.full((4, 5), (index + 1) / 8, dtype=np.float32),
            np.full(8, (index + 1) / 8, dtype=np.float32),
            index == 7,
            np.ones((4, 6), dtype=bool),
        )
    before = learner.agent_q.network[0].weight.detach().clone()

    loss = learner.update(replay.sample(4))

    assert np.isfinite(loss)
    assert not torch.equal(before, learner.agent_q.network[0].weight)


def test_checkpoint_round_trip_preserves_greedy_policy(tmp_path):
    learner = _learner(seed=4)
    observations = np.random.default_rng(3).normal(size=(4, 5)).astype(np.float32)
    masks = np.ones((4, 6), dtype=bool)
    checkpoint = tmp_path / "qmix.pt"
    learner.save(checkpoint)

    loaded = QMIX.from_checkpoint(checkpoint)

    np.testing.assert_array_equal(
        learner.act(observations, masks),
        loaded.act(observations, masks),
    )


def test_checkpoint_rejects_mismatched_network_dimensions(tmp_path):
    checkpoint = tmp_path / "qmix.pt"
    _learner().save(checkpoint)
    incompatible = QMIX(
        n_agents=4,
        obs_dim=6,
        state_dim=8,
        hidden_dim=16,
        mixing_dim=8,
        hypernet_dim=16,
    )

    with pytest.raises(ValueError, match="obs_dim"):
        incompatible.load(checkpoint)


def test_highway_environment_interaction():
    pytest.importorskip("gymnasium")
    pytest.importorskip("highway_env")
    from envs.highway_env import HighwayV2XEnv

    env = HighwayV2XEnv(render_mode=None, duration=1)
    try:
        observations, _ = env.reset(seed=7)
        agents = env.agent_ids
        local_observations = np.stack(
            [observations[agent].reshape(-1) for agent in agents]
        )
        masks_by_agent = env.action_masks()
        masks = np.stack([masks_by_agent[agent] for agent in agents])
        learner = QMIX(
            n_agents=len(agents),
            obs_dim=local_observations.shape[1],
            state_dim=env.state().size,
            n_actions=masks.shape[1],
            hidden_dim=16,
            mixing_dim=8,
            hypernet_dim=16,
            seed=7,
        )
        actions = learner.act(local_observations, masks)
        next_observations, rewards, terminated, truncated, info = env.step(
            {agent: int(actions[index]) for index, agent in enumerate(agents)}
        )

        assert set(next_observations) == set(rewards) == set(agents)
        assert set(terminated) == set(truncated) == set(info) == set(agents)
        assert all(next_observations[agent].shape == (5, 5) for agent in agents)
    finally:
        env.close()
