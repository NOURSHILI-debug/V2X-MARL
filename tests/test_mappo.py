import numpy as np
import pytest
import torch

from marl.mappo import MAPPO, RolloutBuffer, compute_gae


def _mappo(seed=0, obs_dim=5, state_dim=8, n_actions=6, **kwargs):
    return MAPPO(
        n_agents=4,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=16,
        ppo_epochs=2,
        minibatch_size=8,
        **kwargs,
    )


def test_gae_hand_computed_terminal_case():
    advantages, returns = compute_gae(
        rewards=np.array([1.0, 1.0]),
        values=np.array([0.5, 0.5]),
        next_values=np.array([10.0, 10.0]),
        terminated=np.array([False, True]),
        truncated=np.array([False, False]),
        gamma=0.9,
        gae_lambda=0.8,
    )

    np.testing.assert_allclose(advantages, [9.86, 0.5], atol=1e-9)
    np.testing.assert_allclose(returns, [10.36, 1.0], atol=1e-9)


def test_gae_hand_computed_truncation_case_bootstraps():
    advantages, returns = compute_gae(
        rewards=np.array([1.0, 1.0]),
        values=np.array([0.5, 0.5]),
        next_values=np.array([10.0, 10.0]),
        terminated=np.array([False, False]),
        truncated=np.array([False, True]),
        gamma=0.9,
        gae_lambda=0.8,
    )

    np.testing.assert_allclose(advantages, [16.34, 9.5], atol=1e-9)
    np.testing.assert_allclose(returns, [16.84, 10.0], atol=1e-9)


def test_gae_termination_does_not_bootstrap_but_truncation_does():
    common = dict(
        rewards=np.array([0.0]),
        values=np.array([0.0]),
        next_values=np.array([100.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )
    terminated_adv, _ = compute_gae(
        terminated=np.array([True]), truncated=np.array([False]), **common
    )
    truncated_adv, _ = compute_gae(
        terminated=np.array([False]), truncated=np.array([True]), **common
    )

    assert terminated_adv[0] == 0.0
    assert truncated_adv[0] == 100.0


def _filled_buffer(mappo, size=8):
    buffer = RolloutBuffer(
        size,
        mappo.n_agents,
        mappo.obs_dim,
        mappo.state_dim,
        mappo.n_actions,
    )
    rng = np.random.default_rng(0)
    for index in range(size):
        observations = rng.normal(size=(mappo.n_agents, mappo.obs_dim)).astype(
            np.float32
        )
        masks = rng.random((mappo.n_agents, mappo.n_actions)) > 0.3
        masks[:, 0] = True
        state = rng.normal(size=(mappo.state_dim,)).astype(np.float32)
        actions, log_probs, value = mappo.act_with_info(
            observations, masks, state, deterministic=False
        )
        buffer.add(
            observations,
            state,
            actions,
            log_probs,
            value,
            0.0,
            masks,
            index == size - 1,
            False,
            state,
        )
    return buffer


def test_ratio_is_one_and_clip_fraction_zero_before_first_update():
    torch.manual_seed(0)
    mappo = _mappo(seed=0)
    buffer = _filled_buffer(mappo)

    diagnostics = mappo.rollout_diagnostics(buffer)

    assert abs(diagnostics["ratio_mean"] - 1.0) < 1e-5
    assert abs(diagnostics["ratio_max"] - 1.0) < 1e-5
    assert diagnostics["clip_fraction"] == 0.0


def test_masked_actions_have_zero_probability():
    torch.manual_seed(0)
    mappo = _mappo(seed=0)
    observations = np.zeros((4, 5), dtype=np.float32)
    masks = np.array(
        [
            [True, False, False, False, False, False],
            [False, True, False, False, False, False],
            [True, False, True, False, False, False],
            [False, False, False, False, True, False],
        ]
    )

    probs = mappo.action_probs(observations, masks).numpy()

    for agent_index in range(4):
        assert np.all(probs[agent_index][~masks[agent_index]] == 0.0)
        np.testing.assert_allclose(probs[agent_index].sum(), 1.0, atol=1e-6)


def test_act_never_returns_invalid_action():
    torch.manual_seed(0)
    mappo = _mappo(seed=0)
    rng = np.random.default_rng(0)

    for _ in range(100):
        masks = rng.random((4, 6)) > 0.4
        masks[:, 0] = True
        observations = rng.normal(size=(4, 5)).astype(np.float32)
        for deterministic in (False, True):
            actions = mappo.act(observations, masks, deterministic=deterministic)
            assert actions.shape == (4,)
            assert all(masks[index, action] for index, action in enumerate(actions))


def test_checkpoint_round_trip_preserves_greedy_policy(tmp_path):
    torch.manual_seed(0)
    mappo = _mappo(seed=1)
    observations = np.random.default_rng(3).normal(size=(4, 5)).astype(np.float32)
    masks = np.ones((4, 6), dtype=bool)
    checkpoint = tmp_path / "mappo.pt"

    mappo.save(checkpoint)
    loaded = MAPPO.from_checkpoint(checkpoint)

    np.testing.assert_array_equal(
        mappo.act(observations, masks, deterministic=True),
        loaded.act(observations, masks, deterministic=True),
    )
    np.testing.assert_allclose(
        mappo.value(np.zeros(8, dtype=np.float32)),
        loaded.value(np.zeros(8, dtype=np.float32)),
    )


def test_seeded_determinism():
    torch.manual_seed(0)
    first = _mappo(seed=7)
    torch.manual_seed(0)
    second = _mappo(seed=7)
    observations = np.random.default_rng(1).normal(size=(4, 5)).astype(np.float32)
    masks = np.ones((4, 6), dtype=bool)

    np.testing.assert_array_equal(
        first.act(observations, masks, deterministic=True),
        second.act(observations, masks, deterministic=True),
    )
    np.testing.assert_array_equal(
        first.act(observations, masks, deterministic=False),
        second.act(observations, masks, deterministic=False),
    )


def test_matrix_game_reaches_optimal_joint_action():
    torch.manual_seed(0)
    np.random.seed(0)
    mappo = MAPPO(
        n_agents=2,
        obs_dim=1,
        state_dim=2,
        n_actions=2,
        hidden_dim=32,
        learning_rate=5e-3,
        gamma=0.9,
        gae_lambda=0.9,
        clip_ratio=0.2,
        entropy_coef=0.0,
        value_coef=0.5,
        ppo_epochs=2,
        minibatch_size=64,
        reward_scale=1.0,
        seed=0,
    )
    observations = np.zeros((2, 1), dtype=np.float32)
    masks = np.ones((2, 2), dtype=bool)
    state = np.zeros(2, dtype=np.float32)

    for _ in range(120):
        buffer = RolloutBuffer(64, 2, 1, 2, 2)
        for _step in range(64):
            actions, log_probs, value = mappo.act_with_info(
                observations, masks, state, deterministic=False
            )
            reward = 1.0 if (actions[0] == 1 and actions[1] == 1) else 0.0
            buffer.add(
                observations,
                state,
                actions,
                log_probs,
                value,
                reward,
                masks,
                True,
                False,
                state,
            )
        mappo.update(buffer)

    actions = mappo.act(observations, masks, deterministic=True)
    assert actions.tolist() == [1, 1]


def test_one_update_with_contract56_obs_dim():
    torch.manual_seed(0)
    mappo = _mappo(obs_dim=56, state_dim=20, seed=0)
    buffer = RolloutBuffer(8, 4, 56, 20, 6)
    rng = np.random.default_rng(0)
    for index in range(8):
        observations = rng.normal(size=(4, 56)).astype(np.float32)
        masks = np.ones((4, 6), dtype=bool)
        state = rng.normal(size=(20,)).astype(np.float32)
        actions, log_probs, value = mappo.act_with_info(observations, masks, state)
        buffer.add(
            observations,
            state,
            actions,
            log_probs,
            value,
            float(index) / 8.0,
            masks,
            index == 7,
            False,
            state,
        )

    stats = mappo.update(buffer)

    assert np.isfinite(stats["loss"])
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])


def test_rollout_buffer_round_trip(tmp_path):
    torch.manual_seed(0)
    mappo = _mappo(seed=0)
    buffer = _filled_buffer(mappo)
    path = tmp_path / "rollout.npz"

    buffer.save(path)
    restored = RolloutBuffer(8, 4, 5, 8, 6)
    restored.load(path)

    assert len(restored) == len(buffer)
    np.testing.assert_array_equal(restored.actions, buffer.actions)
    np.testing.assert_allclose(restored.log_probs, buffer.log_probs)
