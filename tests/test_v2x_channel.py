"""Tests for the pure-NumPy V2X channel."""

import math

import numpy as np
import pytest

from envs.obs_builder import CANDIDATE_RANGE, CAND_FEATURES, M_CANDIDATES, V_MAX
from envs.v2x_channel import (
    ReceivedMessage,
    V2XChannel,
    V2XConfig,
    messages_to_candidates,
)


def _msg(x=0.0, y=0.0, vx=0.0, vy=0.0, heading=0.0, intent=0.0):
    return np.array([x, y, vx, vy, heading, intent], dtype=np.float64)


def _senders(*positions):
    return [(name, _msg(x=x, y=0.0)) for name, x in positions]


def _trace(channel, steps):
    trace = []
    for t in range(steps):
        inbox, stats = channel.step(
            [
                ("A", _msg(x=0.0, vx=float(t), intent=float(t % 6))),
                ("B", _msg(x=5.0, vx=float(100 + t), intent=float((t + 1) % 6))),
                ("C", _msg(x=8.0, vx=float(200 + t), intent=float((t + 2) % 6))),
            ]
        )
        rendered = tuple(
            (receiver, tuple((m.sender_id, m.age, tuple(m.message)) for m in msgs))
            for receiver, msgs in sorted(inbox.items())
        )
        trace.append((tuple(stats.items()), rendered))
    return trace


# ---- configuration ---------------------------------------------------------
def test_config_defaults_and_valid_values():
    default = V2XConfig()
    assert default.range_m == 50.0
    assert default.delay_steps == 0
    assert default.loss_prob == 0.0

    custom = V2XConfig(range_m=30, delay_steps=3, loss_prob=0.3)
    assert custom.range_m == 30.0
    assert custom.delay_steps == 3
    assert custom.loss_prob == 0.3


def test_config_is_frozen():
    config = V2XConfig()
    with pytest.raises(AttributeError):
        config.range_m = 10.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"range_m": -0.1},
        {"range_m": float("nan")},
        {"range_m": float("inf")},
        {"range_m": "wide"},
        {"delay_steps": -1},
        {"delay_steps": 1.5},
        {"delay_steps": "2"},
        {"delay_steps": True},
        {"loss_prob": -0.01},
        {"loss_prob": 1.01},
        {"loss_prob": float("nan")},
        {"loss_prob": "half"},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        V2XConfig(**kwargs)


# ---- range -----------------------------------------------------------------
@pytest.mark.parametrize("range_m", [30.0, 50.0, 100.0])
def test_sender_at_exactly_range_is_delivered(range_m):
    channel = V2XChannel(V2XConfig(range_m=range_m), seed=0)
    inbox, stats = channel.step([("A", _msg(x=0.0)), ("B", _msg(x=range_m))])

    assert [m.sender_id for m in inbox["A"]] == ["B"]
    assert [m.sender_id for m in inbox["B"]] == ["A"]
    assert stats["dropped_by_range"] == 0
    assert stats["delivered"] == 2


@pytest.mark.parametrize("range_m", [30.0, 50.0, 100.0])
def test_sender_beyond_range_is_dropped(range_m):
    channel = V2XChannel(V2XConfig(range_m=range_m), seed=0)
    inbox, stats = channel.step([("A", _msg(x=0.0)), ("B", _msg(x=range_m + 1e-6))])

    assert inbox["A"] == []
    assert inbox["B"] == []
    assert stats["dropped_by_range"] == 2
    assert stats["delivered"] == 0


# ---- loss ------------------------------------------------------------------
@pytest.mark.parametrize("probability", [0.1, 0.2, 0.3])
def test_empirical_loss_rate_is_close_to_probability(probability):
    channel = V2XChannel(V2XConfig(loss_prob=probability), seed=123)
    for _ in range(10_000):
        channel.step([("A", _msg()), ("B", _msg(x=1.0))])

    stats = channel.episode_stats()
    attempted = stats["sent"] - stats["dropped_by_range"]
    assert stats["sent"] == 20_000
    assert attempted == 20_000
    rate = stats["dropped_by_loss"] / attempted
    assert abs(rate - probability) <= 0.02


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_loss_extremes_are_exact(probability):
    channel = V2XChannel(V2XConfig(loss_prob=probability), seed=9)
    for _ in range(100):
        channel.step([("A", _msg()), ("B", _msg(x=1.0))])

    stats = channel.episode_stats()
    if probability == 0.0:
        assert stats["dropped_by_loss"] == 0
        assert stats["delivered"] == stats["sent"]
    else:
        assert stats["dropped_by_loss"] == stats["sent"]
        assert stats["delivered"] == 0
        assert stats["in_flight"] == 0


# ---- delay -----------------------------------------------------------------
@pytest.mark.parametrize("delay", [0, 1, 2, 3])
def test_delay_arrival_step_age_and_stale_content(delay):
    channel = V2XChannel(V2XConfig(delay_steps=delay), seed=0)
    for step in range(delay + 5):
        message_a = _msg(x=0.0, vx=float(step), intent=float(step % 6))
        message_b = _msg(x=5.0, vx=float(1000 + step), intent=float((step + 1) % 6))
        inbox, stats = channel.step([("A", message_a), ("B", message_b)])

        if step < delay:
            assert inbox["A"] == []
            assert inbox["B"] == []
            assert stats["delivered"] == 0
            continue

        source = step - delay
        assert len(inbox["A"]) == 1
        assert len(inbox["B"]) == 1
        received_b = inbox["A"][0]
        received_a = inbox["B"][0]
        assert received_b.sender_id == "B"
        assert received_a.sender_id == "A"
        assert received_b.age == delay
        assert received_a.age == delay
        assert received_b.message[2] == float(1000 + source)
        assert received_a.message[2] == float(source)
        assert received_b.message[5] == float((source + 1) % 6)
        assert received_a.message[5] == float(source % 6)
        assert stats["delivered"] == 2


# ---- determinism and reset -------------------------------------------------
def test_same_seed_reproduces_and_different_seeds_differ():
    config = V2XConfig(delay_steps=1, loss_prob=0.5)
    first = _trace(V2XChannel(config, seed=1), steps=40)
    second = _trace(V2XChannel(config, seed=1), steps=40)
    other = _trace(V2XChannel(config, seed=2), steps=40)

    assert first == second
    assert first != other


def test_reset_replays_seed_and_discards_in_flight():
    channel = V2XChannel(V2XConfig(delay_steps=3), seed=11)
    channel.step([("A", _msg()), ("B", _msg(x=1.0))])
    assert channel.in_flight == 2

    channel.reset(seed=11)
    assert channel.in_flight == 0
    assert channel.episode_stats()["sent"] == 0
    inbox, stats = channel.step([("A", _msg()), ("B", _msg(x=1.0))])
    assert inbox["A"] == [] and inbox["B"] == []
    assert stats["delivered"] == 0

    channel.reset(seed=11)
    first = _trace(channel, steps=25)
    channel.reset(seed=11)
    second = _trace(channel, steps=25)
    assert first == second

    channel.reset(seed=5)
    seeded = _trace(channel, steps=25)
    channel.reset()
    replayed = _trace(channel, steps=25)
    assert seeded == replayed


# ---- receivers, ranges and counts -----------------------------------------
def test_no_self_reception_out_of_range_excluded_and_counts_consistent():
    channel = V2XChannel(V2XConfig(range_m=10.0), seed=0)
    inbox, stats = channel.step(
        [("A", _msg(x=0.0)), ("B", _msg(x=5.0)), ("C", _msg(x=20.0))]
    )

    assert [m.sender_id for m in inbox["A"]] == ["B"]
    assert [m.sender_id for m in inbox["B"]] == ["A"]
    assert inbox["C"] == []
    assert all(m.sender_id != receiver for receiver, msgs in inbox.items() for m in msgs)
    assert stats["sent"] == 6
    assert stats["dropped_by_range"] == 4
    assert stats["dropped_by_loss"] == 0
    assert stats["delivered"] == 2

    episode = channel.episode_stats()
    assert episode["sent"] == stats["sent"]
    assert episode["delivered"] == stats["delivered"]
    assert episode["dropped"] == stats["dropped_by_range"] + stats["dropped_by_loss"]


def test_counts_are_consistent_with_delay():
    channel = V2XChannel(
        V2XConfig(range_m=10.0, delay_steps=2, loss_prob=0.25), seed=4
    )
    senders = [
        ("A", _msg(x=0.0)),
        ("B", _msg(x=5.0)),
        ("C", _msg(x=20.0)),
        ("D", _msg(x=8.0)),
    ]
    previous_in_flight = 0
    for _ in range(200):
        inbox, stats = channel.step(senders)
        accepted = (
            stats["sent"] - stats["dropped_by_range"] - stats["dropped_by_loss"]
        )
        assert stats["sent"] == (
            stats["dropped_by_range"] + stats["dropped_by_loss"] + accepted
        )
        assert stats["in_flight"] == previous_in_flight + accepted - stats["delivered"]
        assert stats["in_flight"] >= 0
        for messages in inbox.values():
            for message in messages:
                assert message.age == 2
        previous_in_flight = stats["in_flight"]

    episode = channel.episode_stats()
    assert episode["sent"] == (
        episode["delivered"] + episode["dropped"] + episode["in_flight"]
    )
    assert episode["mean_age"] == pytest.approx(2.0)


# ---- conversion helper -----------------------------------------------------
def test_conversion_layout_dtype_and_zero_padding():
    ego = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0}
    received = [
        ReceivedMessage(
            "B", 0, _msg(x=10.0, y=0.0, vx=1.0, vy=2.0, heading=0.5, intent=3.0)
        )
    ]
    block = messages_to_candidates(received, ego)

    assert block.shape == (M_CANDIDATES, CAND_FEATURES)
    assert block.dtype == np.float32
    assert block[0] == pytest.approx(
        [
            10.0 / CANDIDATE_RANGE,
            0.0,
            1.0 / V_MAX,
            2.0 / V_MAX,
            0.5 / math.pi,
            1.0,
        ]
    )
    assert np.array_equal(block[1:], np.zeros((M_CANDIDATES - 1, 6), np.float32))


def test_conversion_keeps_m_nearest_and_marks_validity():
    ego = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0}
    received = [
        ReceivedMessage(str(index), 0, _msg(x=float(index))) for index in range(1, 9)
    ]
    block = messages_to_candidates(received, ego)

    assert block[:, 0] == pytest.approx(
        [index / CANDIDATE_RANGE for index in range(1, M_CANDIDATES + 1)]
    )
    assert np.all(block[:, 5] == 1.0)


def test_conversion_uses_ego_frame_and_accepts_sequence_ego():
    ego_mapping = {"x": 5.0, "y": 5.0, "vx": 1.0, "vy": 0.0, "yaw": math.pi / 2}
    ego_sequence = [5.0, 5.0, 1.0, 0.0, math.pi / 2]
    received = [
        ReceivedMessage(
            "B", 0, _msg(x=5.0, y=6.0, vx=2.0, vy=0.0, heading=math.pi / 2 + 0.1)
        )
    ]

    block = messages_to_candidates(received, ego_mapping)
    assert block[0] == pytest.approx(
        [1.0 / CANDIDATE_RANGE, 0.0, 0.0, -1.0 / V_MAX, 0.1 / math.pi, 1.0]
    )
    assert np.array_equal(block, messages_to_candidates(received, ego_sequence))


# ---- combined effects ------------------------------------------------------
def test_range_loss_and_delay_combine():
    channel = V2XChannel(
        V2XConfig(range_m=10.0, delay_steps=2, loss_prob=0.25), seed=3
    )
    steps = 2_000
    for step in range(steps):
        inbox, stats = channel.step(
            [
                ("A", _msg(x=0.0, vx=float(step), intent=float(step % 6))),
                ("B", _msg(x=5.0, vx=float(100 + step))),
                ("C", _msg(x=20.0, vx=float(200 + step))),
                ("D", _msg(x=8.0, vx=float(300 + step))),
            ]
        )
        assert stats["sent"] == 12
        assert stats["dropped_by_range"] == 6
        if step < 2:
            assert all(not messages for messages in inbox.values())
            continue
        for messages in inbox.values():
            for message in messages:
                assert message.sender_id != "C"
                assert message.age == 2
                expected = {
                    "A": float(step - 2),
                    "B": float(100 + step - 2),
                    "D": float(300 + step - 2),
                }
                assert message.message[2] == expected[message.sender_id]

    episode = channel.episode_stats()
    attempted = episode["sent"] - episode["dropped_by_range"]
    rate = episode["dropped_by_loss"] / attempted
    assert abs(rate - 0.25) <= 0.03
    assert episode["sent"] == (
        episode["delivered"] + episode["dropped"] + episode["in_flight"]
    )
    assert episode["dropped_by_range"] == 6 * steps
