"""Unit tests for the candidate-selection layer (grouping/).

These tests are CARLA-free: they import only NumPy, the observation builder
(pure NumPy) and the grouping module. No CARLA server, Gymnasium or PyTorch is
required, and `envs/v2x_channel.py` is neither imported nor modified.
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
)


# --------------------------------------------------------------------------- #
# fixtures / helpers
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


def _observation_for(records):
    obs = build_observation(_ego(), _route(), list(records), _mask())
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    return obs


# --------------------------------------------------------------------------- #
# 1. nearest-k chooses the correct closest candidates
# --------------------------------------------------------------------------- #
def test_nearest_k_chooses_closest():
    pool = [
        _cand(1, 10.0, 0.0),   # d = 10
        _cand(2, 3.0, 0.0),    # d = 3   <- closest
        _cand(3, 5.0, 0.0),    # d = 5   <- 2nd closest
        _cand(4, 20.0, 0.0),   # d = 20
    ]
    sel = NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    assert sel.strategy == "nearest_k"
    assert sel.k == 2
    assert sel.pool_size == 4
    assert sel.selected_count == 2
    assert sel.missing_slots == 0
    assert tuple(sel.ids) == (2, 3)
    assert sel.records[0] is pool[1]
    assert sel.records[1] is pool[2]
    # scores are the distances
    assert pytest.approx(sel.scores[0]) == 3.0
    assert pytest.approx(sel.scores[1]) == 5.0


# --------------------------------------------------------------------------- #
# 2. nearest-k resolves equal-distance ties deterministically (distance, id)
# --------------------------------------------------------------------------- #
def test_nearest_k_equal_distance_tie_break_by_actor_id():
    pool = [
        _cand(9, 5.0, 0.0),   # d = 5
        _cand(3, 5.0, 0.0),   # d = 5  (tie; lower id first)
        _cand(1, 5.0, 0.0),   # d = 5  (tie; lowest id first)
    ]
    sel = NearestSelector().select(pool, k=3, ego={"x": 0.0, "y": 0.0})
    assert tuple(sel.ids) == (1, 3, 9)


def test_nearest_k_tie_break_independent_of_input_order():
    pool_a = [_cand(9, 5.0, 0.0), _cand(3, 5.0, 0.0)]
    pool_b = [_cand(3, 5.0, 0.0), _cand(9, 5.0, 0.0)]
    sel_a = NearestSelector().select(pool_a, k=2, ego={"x": 0.0, "y": 0.0})
    sel_b = NearestSelector().select(pool_b, k=2, ego={"x": 0.0, "y": 0.0})
    assert tuple(sel_a.ids) == tuple(sel_b.ids) == (3, 9)


# --------------------------------------------------------------------------- #
# 3. random-k never selects duplicate candidates
# --------------------------------------------------------------------------- #
def test_random_k_never_selects_duplicates():
    pool = [_cand(i, float(i), 0.0) for i in range(8)]
    rng = np.random.default_rng(0)
    for _ in range(200):
        sel = RandomSelector().select(pool, k=3, rng=rng)
        ids = [i for i in sel.ids if i is not None]
        assert len(ids) == len(set(ids)), "duplicate ids selected: %r" % (ids,)
        assert sel.selected_count == 3
        assert sel.missing_slots == 0


# --------------------------------------------------------------------------- #
# 4. random-k is reproducible with the same seed
# --------------------------------------------------------------------------- #
def test_random_k_reproducible_with_same_seed():
    pool = [_cand(i, float(i), 0.0) for i in range(8)]
    sel_a = RandomSelector().select(pool, k=2, rng=np.random.default_rng(42))
    sel_b = RandomSelector().select(pool, k=2, rng=np.random.default_rng(42))
    assert tuple(sel_a.ids) == tuple(sel_b.ids)
    assert tuple(sel_a.indices) == tuple(sel_b.indices)


def test_random_k_different_seed_may_differ():
    pool = [_cand(i, float(i), 0.0) for i in range(8)]
    sel_a = RandomSelector().select(pool, k=2, rng=np.random.default_rng(1))
    sel_b = RandomSelector().select(pool, k=2, rng=np.random.default_rng(2))
    assert sel_a.selected_count == sel_b.selected_count == 2


# --------------------------------------------------------------------------- #
# 5. random-k does not mutate the input pool
# --------------------------------------------------------------------------- #
def test_random_k_does_not_mutate_input_pool():
    pool = [_cand(i, float(i), 0.0) for i in range(6)]
    snapshot = [dict(c) for c in pool]
    rng = np.random.default_rng(7)
    RandomSelector().select(pool, k=3, rng=rng)
    assert pool == snapshot
    for before, after in zip(snapshot, pool):
        assert before == after


# --------------------------------------------------------------------------- #
# 6. empty candidate pool
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("selector", [NearestSelector(), RandomSelector()])
def test_empty_candidate_pool(selector):
    if isinstance(selector, RandomSelector):
        sel = selector.select([], k=2, rng=np.random.default_rng(0))
    else:
        sel = selector.select([], k=2, ego={"x": 0.0, "y": 0.0})
    assert isinstance(sel, Selection)
    assert sel.strategy == selector.name
    assert sel.k == 2
    assert sel.pool_size == 0
    assert sel.selected_count == 0
    assert sel.missing_slots == 2
    assert tuple(sel.indices) == (-1, -1)
    assert tuple(sel.ids) == (None, None)
    assert tuple(sel.records) == (None, None)
    assert sel.selected_records() == []


# --------------------------------------------------------------------------- #
# 7. pool containing fewer than k candidates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("selector", [NearestSelector(), RandomSelector()])
def test_pool_fewer_than_k(selector):
    pool = [_cand(5, 1.0, 0.0)]
    if isinstance(selector, RandomSelector):
        sel = selector.select(pool, k=2, rng=np.random.default_rng(0))
    else:
        sel = selector.select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    assert sel.pool_size == 1
    assert sel.selected_count == 1
    assert sel.missing_slots == 1
    assert tuple(sel.ids) == (5, None)
    assert tuple(sel.indices) == (0, -1)
    assert len(sel.selected_records()) == 1


# --------------------------------------------------------------------------- #
# 8. pool containing exactly k candidates
# --------------------------------------------------------------------------- #
def test_pool_exactly_k():
    pool = [_cand(1, 1.0, 0.0), _cand(2, 2.0, 0.0)]
    sel = NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    assert sel.selected_count == 2
    assert sel.missing_slots == 0
    assert tuple(sel.ids) == (1, 2)
    assert sel.selected_records() == [pool[0], pool[1]]


# --------------------------------------------------------------------------- #
# 9. pool containing more than k candidates
# --------------------------------------------------------------------------- #
def test_pool_more_than_k():
    pool = [_cand(i, float(i * 10), 0.0) for i in range(5)]
    sel = NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    assert sel.pool_size == 5
    assert sel.selected_count == 2
    assert sel.missing_slots == 0
    assert tuple(sel.ids) == (0, 1)  # closest two


# --------------------------------------------------------------------------- #
# 10. duplicate actor ids / malformed records handled explicitly
# --------------------------------------------------------------------------- #
def test_duplicate_actor_id_raises():
    pool = [_cand(1, 1.0, 0.0), _cand(1, 2.0, 0.0)]
    with pytest.raises(ValueError):
        NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    with pytest.raises(ValueError):
        RandomSelector().select(pool, k=2, rng=np.random.default_rng(0))


def test_malformed_record_missing_xy_raises():
    pool = [_cand(1, 1.0, 0.0), {"actor_id": 2, "vx": 0.0, "vy": 0.0}]
    with pytest.raises(ValueError):
        NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})


def test_non_finite_coordinate_raises():
    pool = [_cand(1, 1.0, 0.0), _cand(2, float("nan"), 0.0)]
    with pytest.raises(ValueError):
        NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})


def test_non_mapping_record_raises():
    pool = [_cand(1, 1.0, 0.0), [1, 2, 3]]
    with pytest.raises(ValueError):
        NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})


def test_random_requires_explicit_generator():
    pool = [_cand(1, 1.0, 0.0), _cand(2, 2.0, 0.0)]
    with pytest.raises(ValueError):
        RandomSelector().select(pool, k=2, rng=None)
    with pytest.raises(TypeError):
        RandomSelector().select(pool, k=2, rng=np.random.RandomState(0))


# --------------------------------------------------------------------------- #
# 11. selection metadata is internally consistent
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("selector", [NearestSelector(), RandomSelector()])
def test_selection_metadata_consistency(selector):
    pool = [_cand(i, float(i), 0.0) for i in range(6)]
    if isinstance(selector, RandomSelector):
        sel = selector.select(pool, k=2, rng=np.random.default_rng(3))
    else:
        sel = selector.select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    assert len(sel.indices) == sel.k
    assert len(sel.ids) == sel.k
    assert len(sel.records) == sel.k
    assert len(sel.scores) == sel.k
    assert sel.selected_count + sel.missing_slots == sel.k
    assert sel.selected_count <= sel.pool_size
    real_indices = [i for i in sel.indices if i != -1]
    assert len(real_indices) == len(set(real_indices))
    real_ids = [i for i in sel.ids if i is not None]
    assert len(real_ids) == len(set(real_ids))
    for index in real_indices:
        assert 0 <= index < sel.pool_size
    # nearest-k scores are non-negative distances
    if isinstance(selector, NearestSelector):
        for score in sel.scores:
            if score is not None:
                assert score >= 0.0


def test_nearest_k_scores_match_distances():
    pool = [_cand(1, 3.0, 4.0), _cand(2, 0.0, 0.0)]  # d = 5, d = 0
    sel = NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    assert pytest.approx(sel.scores[0]) == 0.0   # actor 2 first
    assert pytest.approx(sel.scores[1]) == 5.0   # actor 1 second
    assert tuple(sel.ids) == (2, 1)


# --------------------------------------------------------------------------- #
# 12. records compatible with the existing observation builder (56-D)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("selector", [NearestSelector(), RandomSelector()])
def test_selected_records_build_valid_56d_observation(selector):
    pool = [_cand(i, float(i * 5), 0.0) for i in range(4)]
    if isinstance(selector, RandomSelector):
        sel = selector.select(pool, k=2, rng=np.random.default_rng(0))
    else:
        sel = selector.select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    obs = _observation_for(sel.selected_records())
    # candidate block occupies [EGO+ROUTE : EGO+ROUTE+CAND]
    candidate_block = obs[EGO + ROUTE : EGO + ROUTE + CAND].reshape(6, 6)
    assert candidate_block.shape == (6, 6)
    assert candidate_block.dtype == np.float32
    # selected rows have valid == 1, padding rows valid == 0
    valid_col = candidate_block[:, 5]
    assert np.count_nonzero(valid_col == 1.0) == sel.selected_count
    assert np.count_nonzero(valid_col == 0.0) == 6 - sel.selected_count


def test_observation_action_mask_preserved():
    """Selection must not alter the six-action validity mask (distinct concept)."""
    pool = [_cand(i, float(i * 5), 0.0) for i in range(4)]
    sel = NearestSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})
    input_mask = _mask()
    obs = build_observation(_ego(), _route(), sel.selected_records(), input_mask)
    out_mask = obs[OBS_DIM - MASK:]
    # emergency brake is forced True by the builder; other bits preserved
    assert bool(out_mask[3]) is True
    for i in range(MASK):
        if i == 3:
            continue
        assert bool(out_mask[i]) == bool(input_mask[i])


# --------------------------------------------------------------------------- #
# learned selector placeholder
# --------------------------------------------------------------------------- #
def test_learned_selector_placeholder_raises():
    pool = [_cand(1, 1.0, 0.0), _cand(2, 2.0, 0.0)]
    with pytest.raises(NotImplementedError):
        LearnedSelector().select(pool, k=2, ego={"x": 0.0, "y": 0.0})
