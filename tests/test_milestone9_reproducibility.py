from __future__ import annotations

import numpy as np
import pytest

from grouping import NearestSelector, RandomSelector, make_selector


def _cand(actor_id, x, y):
    return {"actor_id": actor_id, "x": float(x), "y": float(y), "vx": 0.0, "vy": 0.0, "yaw": 0.0, "valid": True}


def _selected_ids(selection):
    return [i for i in selection.ids if i is not None]


def test_random_k_same_seed_reproduces_selections():
    pool = [_cand(i, float(i), 0.0) for i in range(8)]
    sel_a = RandomSelector().select(pool, k=3, rng=np.random.default_rng(123))
    sel_b = RandomSelector().select(pool, k=3, rng=np.random.default_rng(123))
    assert _selected_ids(sel_a) == _selected_ids(sel_b)
    assert sel_a.indices == sel_b.indices
    assert sel_a.selected_count == 3
    assert sel_a.missing_slots == 0


def test_random_k_different_seeds_can_differ_when_candidates_permit():
    pool = [_cand(i, float(i), 0.0) for i in range(20)]
    # Try a few different seeds to avoid accidental equality.
    results = {
        tuple(_selected_ids(RandomSelector().select(pool, k=3, rng=np.random.default_rng(s))))
        for s in range(1, 50)
    }
    assert len(results) > 1


def test_per_episode_stream_initialization_via_fresh_seeded_rng():
    pool = [_cand(i, float(i), 0.0) for i in range(6)]
    episode_seed = 777
    # "Resetting an experimental episode correctly initializes its selection stream":
    # for the Milestone 9 runner design, each episode uses a fresh per-episode
    # numpy.random.Generator seeded explicitly (fresh env/selector stream), so the
    # same episode seed reproduces the same initial random-k selections.
    first_run = _selected_ids(RandomSelector().select(pool, k=2, rng=np.random.default_rng(episode_seed)))
    second_run = _selected_ids(RandomSelector().select(pool, k=2, rng=np.random.default_rng(episode_seed)))
    assert first_run == second_run
    # different episode seed initializes a different stream (may or may not differ
    # for a single draw, but must be a valid, deterministic selection)
    other = _selected_ids(RandomSelector().select(pool, k=2, rng=np.random.default_rng(episode_seed + 1)))
    assert len(other) == 2
    assert len(set(other)) == 2


def test_nearest_k_behavior_unchanged_and_deterministic():
    pool = [_cand(3, 30.0, 0.0), _cand(1, 10.0, 0.0), _cand(2, 20.0, 0.0)]
    ego = {"x": 0.0, "y": 0.0}
    sel = NearestSelector().select(pool, k=2, ego=ego)
    assert sel.strategy == "nearest_k"
    assert _selected_ids(sel) == [1, 2]
    assert make_selector("nearest_k").name == "nearest_k"


def test_selector_backward_compatibility_factory_and_legacy_modes():
    assert make_selector("nearest_k").name == "nearest_k"
    assert make_selector("random_k").name == "random_k"
    for legacy in (None, "legacy", "disabled", "none", "off"):
        assert make_selector(legacy) is None
    # unknown modes must still fail explicitly (no silent fallback)
    with pytest.raises(ValueError):
        make_selector("")


def test_global_numpy_rng_not_consumed_by_explicit_random_selector():
    pool = [_cand(i, float(i), 0.0) for i in range(10)]
    np.random.seed(424242)
    state_before = np.random.get_state()[1].copy()
    _ = RandomSelector().select(pool, k=3, rng=np.random.default_rng(7))
    state_after = np.random.get_state()[1].copy()
    assert np.array_equal(state_before, state_after)
    # and global RNG stream remains deterministic
    np.random.seed(424242)
    expected = np.random.rand(5)
    np.random.seed(424242)
    _ = RandomSelector().select(pool, k=3, rng=np.random.default_rng(99))
    got = np.random.rand(5)
    assert np.allclose(expected, got)
