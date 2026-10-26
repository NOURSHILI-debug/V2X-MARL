
import numpy as np
import pytest

from envs.toy_env import ToyEnv


def test_reset_initializes_two_vehicles():
    env = ToyEnv()

    observations = env.reset(seed=42)

    assert env.n_agents == 2
    assert set(observations.keys()) == {
        "vehicle_0",
        "vehicle_1",
    }
    assert env.positions.shape == (2, 2)
    assert env.speeds.shape == (2,)


def test_reset_is_reproducible():
    env = ToyEnv()

    first = env.reset(seed=42)
    second = env.reset(seed=42)

    for agent in env.agents:
        np.testing.assert_array_equal(first[agent], second[agent])


def test_maintain_action_moves_vehicles():
    env = ToyEnv(dt=0.1)

    env.reset(seed=42)
    initial_positions = env.positions.copy()

    actions = {
        "vehicle_0": ToyEnv.MAINTAIN,
        "vehicle_1": ToyEnv.MAINTAIN,
    }

    observations, rewards, terminated, truncated, info = (
        env.step(actions)
    )

    assert env.positions[0, 0] > initial_positions[0, 0]
    assert env.positions[1, 1] > initial_positions[1, 1]
    assert not info["collision"]


def test_invalid_action_is_rejected():
    env = ToyEnv()

    env.reset(seed=42)

    actions = {
        "vehicle_0": 99,
        "vehicle_1": ToyEnv.MAINTAIN,
    }

    with pytest.raises(ValueError):
        env.step(actions)


def test_state_has_expected_dimensions():
    env = ToyEnv()

    env.reset(seed=42)

    assert env.state().shape == (6,)

def test_collision_detected_between_positions():
    env = ToyEnv()

    old_positions = np.array([
        [-2.0, 0.0],
        [0.0, -2.0],
    ])

    # The vehicles cross the intersection during this step.
    env.positions = np.array([
        [2.0, 0.0],
        [0.0, 2.0],
    ])

    assert env._has_collision(old_positions)


def test_collision_not_detected_when_far_apart():
    env = ToyEnv()

    old_positions = np.array([
        [-10.0, 0.0],
        [0.0, -10.0],
    ])

    env.positions = np.array([
        [-9.0, 0.0],
        [0.0, -9.0],
    ])

    assert not env._has_collision(old_positions)
