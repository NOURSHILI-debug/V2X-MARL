import math

import pytest

from envs.reward import RewardCalculator


@pytest.fixture
def calc():
    return RewardCalculator(agent_ids=["vehicle_0", "vehicle_1"])


def test_default_weights_match_contract_v1(calc):
    assert calc.weights == {
        "progress": 1.0,
        "collision": -10.0,
        "ttc": -0.5,
        "ttc_threshold": 3.0,
        "comfort": -0.05,
        "goal": 10.0,
        "idle": -0.1,
        "team_collision": -10.0,
    }


def test_progress_gives_positive_contribution(calc):
    reward = calc.compute("vehicle_0", progress=2.5)

    assert reward > 0.0
    assert reward == pytest.approx(2.5, rel=1e-6)


def test_collision_gives_negative_penalty(calc):
    reward = calc.compute("vehicle_0", progress=0.0, collision=True)

    assert reward < 0.0
    assert reward == pytest.approx(-10.0, rel=1e-6)


def test_dangerous_ttc_gives_safety_penalty_and_missing_ttc_is_ignored(calc):
    danger = calc.compute("vehicle_0", progress=0.0, ttc=1.0)
    safe = calc.compute("vehicle_0", progress=0.0, ttc=None)
    at_threshold = calc.compute("vehicle_0", progress=0.0, ttc=3.0)

    assert danger < 0.0
    assert danger == pytest.approx(-0.5 * (1.0 - 1.0 / 3.0))
    assert safe == pytest.approx(0.0, rel=1e-6)
    assert at_threshold == pytest.approx(0.0, rel=1e-6)
    assert danger < safe


def test_goal_completion_adds_bonus(calc):
    reward = calc.compute("vehicle_0", progress=0.0, goal_reached=True)

    assert reward == pytest.approx(10.0, rel=1e-6)


def test_comfort_penalties_behave_correctly(calc):
    reward = calc.compute("vehicle_0", progress=0.0, acceleration=3.0)
    jerk_reward = calc.compute("vehicle_0", progress=0.0, jerk=3.0)
    both = calc.compute(
        "vehicle_0",
        progress=0.0,
        acceleration=3.0,
        jerk=3.0,
    )

    assert reward < 0.0
    assert jerk_reward < 0.0
    assert reward == pytest.approx(-0.15, rel=1e-6)
    assert jerk_reward == pytest.approx(-0.15, rel=1e-6)
    assert both == pytest.approx(-0.3, rel=1e-6)


def test_idle_gives_negative_penalty(calc):
    reward = calc.compute("vehicle_0", progress=0.0, idle=True)

    assert reward == pytest.approx(-0.1, rel=1e-6)


def test_identical_inputs_produce_identical_finite_rewards(calc):
    first = calc.compute("vehicle_0", progress=1.2, collision=False, ttc=3.0, goal_reached=False, acceleration=1.5, jerk=0.8)
    second = calc.compute("vehicle_0", progress=1.2, collision=False, ttc=3.0, goal_reached=False, acceleration=1.5, jerk=0.8)

    assert first == second
    assert math.isfinite(first)


def test_invalid_inputs_and_unknown_agent_ids_are_handled_consistently(calc):
    with pytest.raises(ValueError):
        calc.compute("vehicle_0", progress=float("nan"))

    with pytest.raises(ValueError):
        calc.compute("vehicle_0", progress=1.0, ttc=float("inf"))

    with pytest.raises(KeyError):
        calc.compute("vehicle_99", progress=1.0)


def test_progress_and_goal_bonus_combine_correctly(calc):
    reward = calc.compute("vehicle_0", progress=2.0, goal_reached=True)

    assert reward == pytest.approx(12.0)


def test_collision_penalty_dominates_progress(calc):
    reward = calc.compute("vehicle_0", progress=5.0, collision=True)

    assert reward < 0.0


def test_missing_optional_measurements_are_handled_safely(calc):
    reward = calc.compute(
        "vehicle_0",
        progress=0.0,
        ttc=None,
        acceleration=None,
        jerk=None,
    )

    assert math.isfinite(reward)
    assert reward == pytest.approx(0.0)


def test_compute_terms_sum_to_scalar_reward(calc):
    terms = calc.compute_terms(
        "vehicle_0",
        progress=1.5,
        ttc=1.5,
        jerk=2.0,
        idle=True,
    )
    reward = calc.compute(
        "vehicle_0",
        progress=1.5,
        ttc=1.5,
        jerk=2.0,
        idle=True,
    )

    assert set(terms) == {"progress", "collision", "safety", "comfort", "goal", "idle"}
    assert reward == pytest.approx(sum(terms.values()))


def test_team_reward_is_mean_plus_team_collision_penalty(calc):
    rewards = {"vehicle_0": 1.0, "vehicle_1": 2.0}

    assert calc.team_reward(rewards) == pytest.approx(1.5)
    assert calc.team_reward(rewards, collision=True) == pytest.approx(-8.5)
    assert calc.team_reward({}) == pytest.approx(0.0)
