"""Integration tests for candidate selection + observation builder.

CARLA-free: these tests use synthetic world-frame candidate dictionaries, the
real selectors from ``grouping``, and the real ``envs.obs_builder.build_observation``.
No CARLA server, Gymnasium, PyTorch or V2X channel is required, and
``envs/v2x_channel.py`` is neither imported nor modified.

They mirror the environment wiring::

    pool = get_candidates(agent_id)           # full eligible pool, not truncated
    sel  = selector.select(pool, k=k, ego=ego_ref, rng=rng)
    obs  = build_observation(ego, route, sel.selected_records(), action_mask)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from envs.obs_builder import (
    CAND,
    EGO,
    MASK,
    OBS_DIM,
    ROUTE,
    build_observation,
)
from grouping import (
    LearnedSelector,
    NearestSelector,
    RandomSelector,
    Selection,
    make_selector,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cand(actor_id, x, y, vx=0.0, vy=0.0, yaw=0.0, valid=True):
    return {
        "actor_id": actor_id,
        "x": float(x),
        "y": float(y),
        "vx": float(vx),
        "vy": float(vy),
        "yaw": float(yaw),
        "valid": valid,
    }


def _ego():
    return {
        "x": 0.0,
        "y": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "yaw": 0.0,
        "speed": 5.0,
        "longitudinal_accel": 0.0,
        "heading_error": 0.0,
        "lateral_offset": 0.0,
        "lane_width": 4.0,
        "lane_index": 0.0,
        "lane_count": 2.0,
        "distance_to_goal": 50.0,
        "progress_fraction": 0.5,
    }


def _route():
    return {
        "next_waypoint_x": 10.0,
        "next_waypoint_y": 0.0,
        "speed_limit": 15.0,
        "at_intersection": False,
        "traffic_light_state": "none",
        "distance_to_next": 10.0,
    }


def _mask():
    return np.ones(MASK, dtype=bool)


def _full_pool(n=8):
    # distances from origin: 10, 9, ..., 3  (actor_id == distance ranking here)
    return [_cand(100 + i, float(10 - i), 0.0) for i in range(n)]


def _check_obs(obs, label=""):
    assert obs.shape == (OBS_DIM,), f"{label}: shape {obs.shape}"
    assert obs.dtype == np.float32, f"{label}: dtype {obs.dtype}"
    assert np.isfinite(obs).all(), f"{label}: non-finite"


# --------------------------------------------------------------------------- #
# 1. legacy behaviour unchanged when selection is disabled
# --------------------------------------------------------------------------- #
def test_legacy_passes_full_pool_unchanged():
    pool = _full_pool(8)
    obs = build_observation(_ego(), _route(), list(pool), _mask())
    _check_obs(obs, "legacy")
    candidate_block = obs[EGO + ROUTE : EGO + ROUTE + CAND].reshape(6, 6)
    # build_observation keeps the 6 nearest; actor 103 (d=3) ... actor 108 (d=10)
    assert np.all(candidate_block[:6, 5] == 1.0)


# --------------------------------------------------------------------------- #
# 2. nearest_k is actually invoked and chooses the closest
# --------------------------------------------------------------------------- #
def test_nearest_k_invoked_and_chooses_closest():
    pool = _full_pool(8)
    sel = NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    assert isinstance(sel, Selection)
    assert sel.strategy == "nearest_k"
    assert tuple(sel.ids) == (107, 106)  # d=3, d=4 closest two of pool(8)
    obs = build_observation(
        _ego(), _route(), sel.selected_records(), _mask()
    )
    _check_obs(obs, "nearest_k")
    candidate_block = obs[EGO + ROUTE : EGO + ROUTE + CAND].reshape(6, 6)
    assert np.count_nonzero(candidate_block[:, 5] == 1.0) == 2


# --------------------------------------------------------------------------- #
# 3. random_k is actually invoked and selects from the pool
# --------------------------------------------------------------------------- #
def test_random_k_invoked_and_selects_from_pool():
    pool = _full_pool(8)
    rng = np.random.default_rng(11)
    sel = RandomSelector().select(pool, k=2, rng=rng)
    assert sel.strategy == "random_k"
    assert sel.selected_count == 2
    pool_ids = {c["actor_id"] for c in pool}
    assert all(i in pool_ids for i in sel.ids if i is not None)
    obs = build_observation(_ego(), _route(), sel.selected_records(), _mask())
    _check_obs(obs, "random_k")


# --------------------------------------------------------------------------- #
# 4. both selectors receive the same full candidate pool (no pre-truncation)
# --------------------------------------------------------------------------- #
def test_both_selectors_receive_same_full_pool():
    pool = _full_pool(10)  # more than six -> must NOT be truncated before selection
    sel_nn = NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    sel_rd = RandomSelector().select(pool, k=2, rng=np.random.default_rng(0))
    assert sel_nn.pool_size == 10
    assert sel_rd.pool_size == 10
    assert sel_nn.pool_size == len(pool)
    assert sel_rd.pool_size == len(pool)
    # neither selector may only see six
    assert sel_nn.selected_count == 2
    assert sel_rd.selected_count == 2


# --------------------------------------------------------------------------- #
# 5 + 6. selected records build a valid (56,) float32 finite observation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "selector_factory, kwargs",
    [
        (lambda: NearestSelector(), {"ego": {"x": 0.0, "y": 0.0}}),
        (lambda: RandomSelector(), {"rng": np.random.default_rng(7)}),
    ],
)
def test_selected_records_build_valid_56d_observation(selector_factory, kwargs):
    pool = _full_pool(8)
    sel = selector_factory().select(pool, k=2, **kwargs)
    obs = build_observation(_ego(), _route(), sel.selected_records(), _mask())
    _check_obs(obs, f"{sel.strategy}")
    candidate_block = obs[EGO + ROUTE : EGO + ROUTE + CAND].reshape(6, 6)
    assert candidate_block.shape == (6, 6)
    assert candidate_block.dtype == np.float32
    assert np.count_nonzero(candidate_block[:, 5] == 1.0) == sel.selected_count


# --------------------------------------------------------------------------- #
# 7. empty candidate pool
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("selector", [NearestSelector(), RandomSelector()])
def test_empty_pool_integration(selector):
    if isinstance(selector, RandomSelector):
        sel = selector.select([], k=2, rng=np.random.default_rng(0))
    else:
        sel = selector.select([], k=2, ego={"x": 0.0, "y": 0.0})
    obs = build_observation(_ego(), _route(), sel.selected_records(), _mask())
    _check_obs(obs, f"{sel.strategy}-empty")
    candidate_block = obs[EGO + ROUTE : EGO + ROUTE + CAND].reshape(6, 6)
    assert np.all(candidate_block == 0.0)
    assert sel.selected_count == 0
    assert sel.missing_slots == 2


# --------------------------------------------------------------------------- #
# 8. pool with fewer than k candidates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("selector", [NearestSelector(), RandomSelector()])
def test_pool_fewer_than_k_integration(selector):
    pool = [_full_pool(1)[0]]
    if isinstance(selector, RandomSelector):
        sel = selector.select(pool, k=2, rng=np.random.default_rng(0))
    else:
        sel = selector.select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    obs = build_observation(_ego(), _route(), sel.selected_records(), _mask())
    _check_obs(obs, f"{sel.strategy}-short")
    candidate_block = obs[EGO + ROUTE : EGO + ROUTE + CAND].reshape(6, 6)
    assert np.count_nonzero(candidate_block[:, 5] == 1.0) == 1
    assert sel.selected_count == 1
    assert sel.missing_slots == 1


# --------------------------------------------------------------------------- #
# 9. selection metadata is correct
# --------------------------------------------------------------------------- #
def test_selection_metadata_correct():
    pool = _full_pool(6)
    sel = NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    meta = {
        "strategy": sel.strategy,
        "k": sel.k,
        "pool_size": sel.pool_size,
        "selected_count": sel.selected_count,
        "missing_slots": sel.missing_slots,
        "selected_ids": [i for i in sel.ids],
        "selected_scores": [s for s in sel.scores],
    }
    assert meta["strategy"] == "nearest_k"
    assert meta["k"] == 2
    assert meta["pool_size"] == 6
    assert meta["selected_count"] == 2
    assert meta["missing_slots"] == 0
    assert meta["selected_ids"] == [105, 104]
    assert meta["selected_scores"] == [pytest.approx(5.0), pytest.approx(6.0)]
    # metadata matches what was actually fed to build_observation
    obs = build_observation(_ego(), _route(), sel.selected_records(), _mask())
    candidate_block = obs[EGO + ROUTE : EGO + ROUTE + CAND].reshape(6, 6)
    assert np.count_nonzero(candidate_block[:, 5] == 1.0) == meta["selected_count"]


def test_random_metadata_shape():
    pool = _full_pool(6)
    sel = RandomSelector().select(pool, k=2, rng=np.random.default_rng(3))
    assert len(sel.ids) == sel.k
    assert len(sel.scores) == sel.k
    assert all(s is None for s in sel.scores)  # random has no scores


# --------------------------------------------------------------------------- #
# 10. random selection reproducible with the same seed
# --------------------------------------------------------------------------- #
def test_random_reproducible_same_seed():
    pool = _full_pool(8)
    sel_a = RandomSelector().select(pool, k=2, rng=np.random.default_rng(99))
    sel_b = RandomSelector().select(pool, k=2, rng=np.random.default_rng(99))
    assert tuple(sel_a.ids) == tuple(sel_b.ids)
    obs_a = build_observation(_ego(), _route(), sel_a.selected_records(), _mask())
    obs_b = build_observation(_ego(), _route(), sel_b.selected_records(), _mask())
    assert np.array_equal(obs_a, obs_b)


# --------------------------------------------------------------------------- #
# 11. invalid strategy/configuration fails clearly
# --------------------------------------------------------------------------- #
def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        make_selector("bogus")
    with pytest.raises(ValueError):
        NearestSelector().select(_full_pool(3), k=0, ego={"x": 0.0, "y": 0.0})
    with pytest.raises(ValueError):
        RandomSelector().select(_full_pool(3), k=2, rng=None)


# --------------------------------------------------------------------------- #
# 12. selected ids correspond to actual candidates used in the observation
# --------------------------------------------------------------------------- #
def test_selected_ids_match_observation_rows():
    pool = _full_pool(8)
    sel = NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    records = sel.selected_records()
    obs = build_observation(_ego(), _route(), records, _mask())
    candidate_block = obs[EGO + ROUTE : EGO + ROUTE + CAND].reshape(6, 6)
    # the two valid rows must correspond to the selected records (nearest first)
    for row, record in zip(candidate_block[:2], records):
        rel_x, rel_y = row[0], row[1]
        # distance preserved in the first two columns (scale 50)
        assert pytest.approx(math.hypot(rel_x * 50.0, rel_y * 50.0)) == math.hypot(
            record["x"], record["y"]
        )
    assert np.all(candidate_block[2:, 5] == 0.0)


# --------------------------------------------------------------------------- #
# factory wiring: legacy / nearest_k / random_k resolve without carla
# --------------------------------------------------------------------------- #
def test_factory_returns_correct_selector():
    assert make_selector("legacy") is None
    assert make_selector(None) is None
    assert isinstance(make_selector("nearest_k"), NearestSelector)
    assert isinstance(make_selector("random_k"), RandomSelector)
    assert isinstance(make_selector("learned_k"), LearnedSelector)
