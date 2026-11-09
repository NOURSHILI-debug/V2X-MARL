from scripts.check_obs import _reason
from scripts.check_policies import _termination_reason


def test_policy_reason_returns_unknown_for_unrecognized_terminal_state():
    assert _termination_reason({}, terminated=True, truncated=False) == "unknown"


def test_observation_reason_returns_unknown_for_unrecognized_terminal_state():
    assert _reason({}, terminated=True, truncated=False) == "unknown"
