from __future__ import annotations

import numpy as np

from controllers.observation_heuristic import ObservationHeuristicController
from scripts.milestone9.config import default_config
from scripts.milestone9.run_paired import build_controller, choose_strategy_order


def test_controller_integration_and_identity_match_config():
    cfg = default_config()
    ctrl = build_controller(cfg)
    assert isinstance(ctrl, ObservationHeuristicController)

    obs = np.zeros(56, dtype=np.float32)
    obs[0] = 0.2  # normalized speed
    obs[6] = 0.5  # distance to goal
    obs[7] = 0.1  # progress
    obs[50:] = 1.0  # mask allow all
    action = ctrl.act(obs)
    assert isinstance(action, int)
    assert 0 <= action < 6


def test_strategy_order_counterbalanced_and_deterministic():
    import random

    strategies = ["nearest_k", "random_k"]
    assert choose_strategy_order(strategies, random.Random(0), counterbalanced=True, pair_index=0) == ["nearest_k", "random_k"]
    assert choose_strategy_order(strategies, random.Random(0), counterbalanced=True, pair_index=1) == ["random_k", "nearest_k"]
    r1 = choose_strategy_order(strategies, random.Random(123), counterbalanced=False, pair_index=0)
    r2 = choose_strategy_order(strategies, random.Random(123), counterbalanced=False, pair_index=0)
    assert r1 == r2
    assert sorted(r1) == strategies
