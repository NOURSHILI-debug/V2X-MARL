import pytest

from envs.control import speed_limit_difference, update_target_speed
from envs.obs_builder import ACCEL, DECEL, EMERGENCY_BRAKE, KEEP


def test_target_speed_accelerates_and_decelerates_with_clamps():
    assert update_target_speed(9.0, ACCEL, speed_step=2.0, max_speed=30.0) == 11.0
    assert update_target_speed(1.0, DECEL, speed_step=2.0, max_speed=30.0) == 0.0
    assert update_target_speed(29.0, ACCEL, speed_step=2.0, max_speed=30.0) == 30.0


def test_keep_and_emergency_brake_do_not_mutate_cruise_target():
    for action in (KEEP, EMERGENCY_BRAKE):
        assert update_target_speed(
            9.0,
            action,
            speed_step=2.0,
            max_speed=30.0,
        ) == 9.0


def test_speed_target_converts_to_traffic_manager_percentage():
    assert speed_limit_difference(9.0, 50.0) == pytest.approx(35.2)


def test_speed_helpers_reject_nonfinite_inputs():
    with pytest.raises(ValueError, match="finite"):
        update_target_speed(float("nan"), KEEP, speed_step=2.0, max_speed=30.0)
    with pytest.raises(ValueError, match="finite"):
        speed_limit_difference(9.0, float("inf"))
