"""Synthetic V2X-to-observation integration tests.

Validates the full pipeline without a CARLA server::

    synthetic vehicle states -> V2XChannel -> received messages
    -> v2x_adapter.received_to_candidates() -> build_observation()
    -> 56-D observation

Uses the *actual* V2X channel and observation builder. No mocks of their
behaviour. Nour's ``envs/v2x_channel.py`` and ``tests/test_v2x_channel.py``
are neither imported-for-modification nor altered.

Communication range (channel) and candidate observation range (50 m, builder)
are kept independent: a vehicle at 70 m can be received but never enters the
candidate block.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from envs.obs_builder import (
    CAND,
    CANDIDATE_RANGE,
    EGO,
    MASK,
    OBS_DIM,
    ROUTE,
    V_MAX,
    build_observation,
)
from envs.v2x_channel import ReceivedMessage, V2XChannel, V2XConfig
from envs.v2x_adapter import received_to_candidates


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _msg(x=0.0, y=0.0, vx=0.0, vy=0.0, heading=0.0, intent=0.0):
    return np.array([x, y, vx, vy, heading, intent], dtype=np.float64)


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


def _observation_for(candidates):
    obs = build_observation(_ego(), _route(), list(candidates), _mask())
    assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape}"
    assert obs.dtype == np.float32, f"obs dtype {obs.dtype}"
    assert np.isfinite(obs).all(), "obs not finite"
    return obs


def _candidate_block(obs):
    return obs[EGO + ROUTE : EGO + ROUTE + CAND].reshape(6, 6)


# --------------------------------------------------------------------------- #
# 1. perfect communication, zero delay, zero loss
# --------------------------------------------------------------------------- #
def test_perfect_communication_all_received():
    channel = V2XChannel(V2XConfig(range_m=100.0, delay_steps=0, loss_prob=0.0), seed=1)
    ego_id = "ego"
    senders = [
        (ego_id, _msg(x=0.0, y=0.0)),
        ("A", _msg(x=10.0, y=0.0, vx=1.0, vy=0.0, heading=0.0, intent=1.0)),
        ("B", _msg(x=5.0, y=0.0, vx=0.0, vy=0.0, heading=0.0, intent=2.0)),
    ]
    received, stats = channel.step(senders)
    candidates = received_to_candidates(received[ego_id], _ego())

    # 3 senders -> 3*2 = 6 directed transmissions, all within 100 m range, no loss
    assert stats["sent"] == 6
    assert stats["dropped_by_range"] == 0
    assert stats["dropped_by_loss"] == 0
    assert stats["delivered"] == 6
    # ego receives 2 messages (from A and B)
    assert len(received[ego_id]) == 2
    assert len(candidates) == 2
    ids = {c["actor_id"] for c in candidates}
    assert ids == {"A", "B"}

    obs = _observation_for(candidates)
    block = _candidate_block(obs)
    assert np.count_nonzero(block[:, 5] == 1.0) == 2


# --------------------------------------------------------------------------- #
# 2. vehicle outside communication range
# --------------------------------------------------------------------------- #
def test_outside_communication_range_not_received():
    channel = V2XChannel(V2XConfig(range_m=10.0, delay_steps=0, loss_prob=0.0), seed=1)
    ego_id = "ego"
    senders = [
        (ego_id, _msg(x=0.0, y=0.0)),
        ("far", _msg(x=100.0, y=0.0)),  # 100 m > 10 m comm range
    ]
    received, stats = channel.step(senders)
    candidates = received_to_candidates(received[ego_id], _ego())

    # 2 senders -> 2 directed transmissions, both > 10 m range -> both dropped
    assert stats["sent"] == 2
    assert stats["dropped_by_range"] == 2
    assert stats["delivered"] == 0
    assert len(candidates) == 0

    obs = _observation_for(candidates)
    block = _candidate_block(obs)
    assert np.all(block == 0.0)


# --------------------------------------------------------------------------- #
# 3. received at 70 m but excluded by 50 m candidate observation radius
# --------------------------------------------------------------------------- #
def test_received_at_70m_excluded_by_50m_candidate_range():
    channel = V2XChannel(V2XConfig(range_m=100.0, delay_steps=0, loss_prob=0.0), seed=1)
    ego_id = "ego"
    senders = [
        (ego_id, _msg(x=0.0, y=0.0)),
        ("near", _msg(x=30.0, y=0.0)),   # 30 m: received AND in candidate range
        ("far", _msg(x=70.0, y=0.0)),    # 70 m: received (comm range 100) but NOT candidate
    ]
    received, stats = channel.step(senders)
    candidates = received_to_candidates(received[ego_id], _ego())

    # 3 senders -> 6 directed transmissions, all within 100 m comm range
    assert stats["dropped_by_range"] == 0
    assert stats["delivered"] == 6
    # ego receives 2 messages (from near and far)
    assert len(received[ego_id]) == 2
    assert len(candidates) == 2

    # build_observation applies the 50 m candidate range: only "near" survives
    obs = _observation_for(candidates)
    block = _candidate_block(obs)
    assert np.count_nonzero(block[:, 5] == 1.0) == 1
    # The surviving candidate is the one at 30 m (rel_x = 30/50 = 0.6)
    assert pytest.approx(block[0, 0]) == 30.0 / CANDIDATE_RANGE


# --------------------------------------------------------------------------- #
# 4. packet loss prevents a message from becoming a candidate
# --------------------------------------------------------------------------- #
def test_packet_loss_prevents_candidate():
    channel = V2XChannel(V2XConfig(range_m=100.0, delay_steps=0, loss_prob=1.0), seed=1)
    ego_id = "ego"
    senders = [
        (ego_id, _msg(x=0.0, y=0.0)),
        ("lost", _msg(x=10.0, y=0.0)),
    ]
    received, stats = channel.step(senders)
    candidates = received_to_candidates(received[ego_id], _ego())

    # 2 senders -> 2 directed transmissions, all dropped by loss (prob=1.0)
    assert stats["sent"] == 2
    assert stats["dropped_by_loss"] == 2
    assert stats["delivered"] == 0
    assert len(candidates) == 0

    obs = _observation_for(candidates)
    block = _candidate_block(obs)
    assert np.all(block == 0.0)


# --------------------------------------------------------------------------- #
# 5. delayed message arrives only per channel semantics
# --------------------------------------------------------------------------- #
def test_delayed_message_arrives_per_channel_semantics():
    channel = V2XChannel(V2XConfig(range_m=100.0, delay_steps=2, loss_prob=0.0), seed=1)
    ego_id = "ego"
    senders = [
        (ego_id, _msg(x=0.0, y=0.0)),
        ("late", _msg(x=10.0, y=0.0, intent=float(0 % 6))),
    ]

    # Steps 0 and 1: nothing delivered yet (delay = 2)
    for step in range(2):
        received, stats = channel.step(senders)
        assert stats["delivered"] == 0, f"step {step}: nothing should be delivered yet"
        assert len(received[ego_id]) == 0

    # Step 2: messages sent at step 0 are due (age = 2) - one from ego, one from late
    received, stats = channel.step(senders)
    assert stats["delivered"] == 2  # ego->ego skipped, but ego<->late both directions
    # ego receives from late (and late receives from ego, but we only look at ego's inbox)
    assert len(received[ego_id]) == 1
    assert received[ego_id][0].age == 2
    candidates = received_to_candidates(received[ego_id], _ego())
    assert len(candidates) == 1
    assert candidates[0]["actor_id"] == "late"


# --------------------------------------------------------------------------- #
# 6. message age preserved as metadata
# --------------------------------------------------------------------------- #
def test_message_age_preserved_as_metadata():
    channel = V2XChannel(V2XConfig(range_m=100.0, delay_steps=3, loss_prob=0.0), seed=1)
    ego_id = "ego"
    senders = [
        (ego_id, _msg(x=0.0, y=0.0)),
        ("aged", _msg(x=10.0, y=0.0, intent=float(1 % 6))),
    ]
    for _ in range(3):
        channel.step(senders)
    received, _ = channel.step(senders)

    assert len(received[ego_id]) == 1
    assert received[ego_id][0].age == 3

    candidates = received_to_candidates(received[ego_id], _ego())
    assert candidates[0]["age"] == 3
    # age is metadata, not part of the 6-feature candidate block
    assert "age" in candidates[0]


# --------------------------------------------------------------------------- #
# 7. candidate ordering correct (distance then actor id)
# --------------------------------------------------------------------------- #
def test_candidate_ordering_distance_then_id():
    obs = _observation_for(
        [
            {"actor_id": 9, "x": 10.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0, "valid": True},
            {"actor_id": 3, "x": 5.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0, "valid": True},
            {"actor_id": 1, "x": 5.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0, "valid": True},
        ]
    )
    block = _candidate_block(obs)
    # build_observation sorts by (distance, actor_id): id=1 first (d=5), id=3 second (d=5), id=9 third (d=10)
    assert pytest.approx(block[0, 0]) == 5.0 / CANDIDATE_RANGE   # actor 1
    assert pytest.approx(block[1, 0]) == 5.0 / CANDIDATE_RANGE   # actor 3
    assert pytest.approx(block[2, 0]) == 10.0 / CANDIDATE_RANGE  # actor 9


# --------------------------------------------------------------------------- #
# 8. candidate capacity never exceeds six
# --------------------------------------------------------------------------- #
def test_candidate_capacity_never_exceeds_six():
    candidates = [
        {"actor_id": i, "x": float(i), "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0, "valid": True}
        for i in range(10)
    ]
    obs = _observation_for(candidates)
    block = _candidate_block(obs)
    assert np.count_nonzero(block[:, 5] == 1.0) <= 6


# --------------------------------------------------------------------------- #
# 9. missing candidate slots padded correctly
# --------------------------------------------------------------------------- #
def test_missing_candidate_slots_padded():
    obs = _observation_for([])  # no candidates
    block = _candidate_block(obs)
    assert np.all(block == 0.0)
    assert np.all(block[:, 5] == 0.0)  # valid = 0 for padding


# --------------------------------------------------------------------------- #
# 10. validity features and action masks correct
# --------------------------------------------------------------------------- #
def test_validity_and_action_mask_separate():
    candidates = [
        {"actor_id": 1, "x": 10.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0, "valid": True}
    ]
    mask = np.array([True, False, True, False, True, False], dtype=bool)
    obs = build_observation(_ego(), _route(), candidates, mask)

    block = _candidate_block(obs)
    assert block[0, 5] == 1.0  # valid candidate
    assert np.all(block[1:, 5] == 0.0)  # padding invalid

    out_mask = obs[OBS_DIM - MASK:]
    assert out_mask.shape == (6,)
    assert out_mask.dtype == np.float32  # builder casts mask to float32
    # emergency brake (index 3) is forced True by the builder regardless of input
    assert out_mask[3] == 1.0
    # selection does not alter non-brake mask bits
    assert bool(out_mask[0]) == bool(mask[0])
    assert bool(out_mask[1]) == bool(mask[1])


# --------------------------------------------------------------------------- #
# 11. observation shape exactly (56,)
# --------------------------------------------------------------------------- #
def test_observation_shape_56():
    obs = _observation_for([])
    assert obs.shape == (OBS_DIM,)
    assert obs.shape == (56,)


# --------------------------------------------------------------------------- #
# 12. observation dtype float32
# --------------------------------------------------------------------------- #
def test_observation_dtype_float32():
    obs = _observation_for([])
    assert obs.dtype == np.float32


# --------------------------------------------------------------------------- #
# 13. no NaN or infinity
# --------------------------------------------------------------------------- #
def test_observation_finite():
    obs = _observation_for([])
    assert np.isfinite(obs).all()


# --------------------------------------------------------------------------- #
# 14. no duplicate candidate IDs
# --------------------------------------------------------------------------- #
def test_no_duplicate_candidate_ids():
    obs = _observation_for(
        [
            {"actor_id": 1, "x": 10.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0, "valid": True},
            {"actor_id": 1, "x": 5.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0, "valid": True},
        ]
    )
    block = _candidate_block(obs)
    assert np.count_nonzero(block[:, 5] == 1.0) == 2


# --------------------------------------------------------------------------- #
# 15. identical explicit seeds reproduce communication results
# --------------------------------------------------------------------------- #
def test_identical_seeds_reproduce_results():
    def run_twice(seed):
        results = []
        for _ in range(2):
            channel = V2XChannel(
                V2XConfig(range_m=100.0, delay_steps=1, loss_prob=0.3), seed=seed
            )
            ego_id = "ego"
            senders = [
                (ego_id, _msg(x=0.0, y=0.0)),
                ("A", _msg(x=10.0, y=0.0, intent=1.0)),
                ("B", _msg(x=20.0, y=0.0, intent=2.0)),
            ]
            trace = []
            for _step in range(10):
                received, stats = channel.step(senders)
                trace.append(
                    (
                        stats["delivered"],
                        tuple(
                            (m.sender_id, m.age, tuple(round(float(v), 6) for v in m.message))
                            for m in received.get(ego_id, [])
                        ),
                    )
                )
            results.append(trace)
        return results

    r1 = run_twice(42)
    r2 = run_twice(42)
    assert r1 == r2, "same seed must reproduce identical communication results"

    r3 = run_twice(99)
    assert r1 != r3, "different seed must produce different results"


# --------------------------------------------------------------------------- #
# 16. intent and message age do not overwrite observation features
# --------------------------------------------------------------------------- #
def test_intent_and_age_do_not_overwrite_features():
    channel = V2XChannel(V2XConfig(range_m=100.0, delay_steps=2, loss_prob=0.0), seed=1)
    ego_id = "ego"
    senders = [
        (ego_id, _msg(x=0.0, y=0.0)),
        ("A", _msg(x=10.0, y=0.0, vx=1.0, vy=2.0, heading=0.5, intent=3.0)),
    ]
    # step 0: send; step 2: receive with age=2
    channel.step(senders)
    channel.step(senders)
    received, _ = channel.step(senders)

    candidates = received_to_candidates(received[ego_id], _ego())
    assert candidates[0]["age"] == 2          # metadata preserved
    assert candidates[0]["intent"] == 3.0     # metadata preserved

    obs = _observation_for(candidates)
    block = _candidate_block(obs)
    # The 6-feature block must be [rel_x/50, rel_y/50, rel_vx/30, rel_vy/30, heading/pi, 1.0]
    # intent (3.0) and age (2) must NOT appear in the feature columns.
    assert pytest.approx(block[0, 0]) == 10.0 / CANDIDATE_RANGE   # rel_x
    assert pytest.approx(block[0, 1]) == 0.0                      # rel_y
    assert pytest.approx(block[0, 2]) == 1.0 / V_MAX              # rel_vx
    assert pytest.approx(block[0, 3]) == 2.0 / V_MAX              # rel_vy
    assert pytest.approx(block[0, 4]) == 0.5 / math.pi            # rel_heading
    assert block[0, 5] == 1.0                                     # valid
    # Sanity: intent value 3.0 and age value 2 do not appear anywhere in the block
    assert not np.any(np.isclose(block, 3.0))
    assert not np.any(np.isclose(block, 2.0))


# --------------------------------------------------------------------------- #
# adapter returns empty list for empty inbox
# --------------------------------------------------------------------------- #
def test_adapter_empty_inbox():
    assert received_to_candidates([], _ego()) == []
