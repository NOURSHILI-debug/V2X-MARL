import pytest

from envs.termination import evaluate_termination, update_off_road_steps


AGENTS = ("cav_0", "cav_1")


def _decision(
    *,
    collided=None,
    reached=None,
    off_road_steps=None,
    stuck_steps=None,
    step_count=1,
    max_steps=300,
):
    return evaluate_termination(
        AGENTS,
        collided or dict.fromkeys(AGENTS, False),
        reached or dict.fromkeys(AGENTS, False),
        off_road_steps or dict.fromkeys(AGENTS, 0),
        stuck_steps or dict.fromkeys(AGENTS, 0),
        step_count=step_count,
        max_steps=max_steps,
    )


def test_off_road_counter_requires_consecutive_steps():
    assert update_off_road_steps(0, True) == 1
    assert update_off_road_steps(9, True) == 10
    assert update_off_road_steps(9, False) == 0


@pytest.mark.parametrize(
    ("inputs", "reason"),
    [
        (
            {
                "collided": {"cav_0": True, "cav_1": False},
                "reached": {"cav_0": True, "cav_1": True},
                "off_road_steps": {"cav_0": 10, "cav_1": 0},
                "stuck_steps": {"cav_0": 51, "cav_1": 0},
                "step_count": 300,
            },
            "collision",
        ),
        (
            {
                "reached": {"cav_0": True, "cav_1": True},
                "off_road_steps": {"cav_0": 10, "cav_1": 0},
                "stuck_steps": {"cav_0": 51, "cav_1": 0},
                "step_count": 300,
            },
            "goal",
        ),
        (
            {
                "off_road_steps": {"cav_0": 10, "cav_1": 0},
                "stuck_steps": {"cav_0": 51, "cav_1": 0},
                "step_count": 300,
            },
            "off_road",
        ),
        ({"stuck_steps": {"cav_0": 51, "cav_1": 0}, "step_count": 300}, "stuck"),
        ({"step_count": 300}, "timeout"),
    ],
)
def test_termination_priority(inputs, reason):
    result = _decision(**inputs)
    assert result.reason == reason
    assert result.terminated is (reason != "timeout")
    assert result.truncated is (reason == "timeout")
    assert result.success is (reason == "goal")


def test_termination_continues_before_any_condition():
    result = _decision()

    assert result.reason is None
    assert not result.terminated
    assert not result.truncated
    assert not result.success


def test_stuck_requires_more_than_fifty_slow_steps():
    result = _decision(stuck_steps={"cav_0": 50, "cav_1": 0})

    assert result.reason is None
    assert result.stuck_by_agent == {"cav_0": False, "cav_1": False}


def test_termination_requires_matching_agent_keys():
    with pytest.raises(ValueError, match="exactly the configured agents"):
        evaluate_termination(
            AGENTS,
            {"cav_0": False},
            dict.fromkeys(AGENTS, 0),
            dict.fromkeys(AGENTS, 0),
            dict.fromkeys(AGENTS, False),
            step_count=1,
            max_steps=300,
        )
