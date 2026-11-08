import pytest

from envs.metrics import jerk_magnitude, longitudinal_ttc


def test_ttc_from_longitudinal_gap_and_closing_speed():
    assert longitudinal_ttc(10.0, 5.0) == pytest.approx(2.0)


def test_constant_speed_has_zero_jerk():
    assert jerk_magnitude(3.0, 3.0, 0.2) == 0.0


def test_missing_or_nonclosing_lead_uses_ten_second_cap():
    assert longitudinal_ttc(None, None) == 10.0
    assert longitudinal_ttc(10.0, 0.0) == 10.0
