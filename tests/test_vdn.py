import numpy as np
import pytest
import torch

from marl.vdn import VDN


def _learner(seed=0, double_q=False, reward_scale=1.0):
    return VDN(
        4,
        5,
        6,
        hidden=16,
        seed=seed,
        double_q=double_q,
        reward_scale=reward_scale,
    )


def _batch(batch_size=4):
    generator = torch.Generator().manual_seed(0)
    observations = torch.rand(batch_size, 4, 5, generator=generator)
    actions = torch.randint(0, 6, (batch_size, 4), generator=generator)
    rewards = torch.rand(batch_size, generator=generator)
    next_observations = torch.rand(batch_size, 4, 5, generator=generator)
    terminated = torch.zeros(batch_size)
    next_masks = torch.ones(batch_size, 4, 6, dtype=torch.bool)
    return (
        observations,
        actions,
        rewards,
        next_observations,
        terminated,
        next_masks,
    )


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


def test_vdn_q_tot_equals_sum_of_selected_values():
    learner = _learner()
    batch = _batch()
    observations, actions = batch[0], batch[1]

    selected = learner.q(observations).gather(
        2, actions.unsqueeze(-1)
    ).squeeze(-1)

    torch.testing.assert_close(
        learner.q_tot(observations, actions),
        selected.sum(dim=1),
    )


def test_update_changes_parameters_and_keeps_loss_finite():
    learner = _learner()
    before = learner.q.net[0].weight.detach().clone()

    loss = learner.update(_batch())

    assert np.isfinite(loss)
    assert not torch.equal(before, learner.q.net[0].weight)


def test_checkpoint_round_trip_preserves_greedy_policy(tmp_path):
    learner = _learner(seed=4)
    observations = np.random.default_rng(3).normal(size=(4, 5)).astype(np.float32)
    masks = np.ones((4, 6), dtype=bool)
    checkpoint = tmp_path / "vdn.pt"
    learner.save(checkpoint)

    loaded = VDN.from_checkpoint(checkpoint)

    np.testing.assert_array_equal(
        learner.act(observations, masks),
        loaded.act(observations, masks),
    )


def test_checkpoint_rejects_mismatched_obs_dim(tmp_path):
    checkpoint = tmp_path / "vdn.pt"
    _learner().save(checkpoint)
    incompatible = VDN(4, 6, 6, hidden=16)

    with pytest.raises(ValueError, match="obs_dim"):
        incompatible.load(checkpoint)


def test_from_checkpoint_restores_double_q(tmp_path):
    checkpoint = tmp_path / "vdn_double.pt"
    _learner(double_q=True).save(checkpoint)

    loaded = VDN.from_checkpoint(checkpoint)

    assert loaded.double_q is True
