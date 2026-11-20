"""CARLA-free integration tests for the V2X candidate source.

Exercises the real environment wiring -- ``_configure_selection``,
``_configure_candidate_source``, ``_build_v2x_senders``, ``_get_obs``,
``_extract_observation_inputs``, ``_v2x_candidates``, ``_reset_v2x_state`` --
together with the real V2X channel, the real adapter and the real observation
builder. Only CARLA world/vehicle objects are faked; no CARLA server is needed.

``CarlaMultiAgentEnv.reset()`` / ``step()`` themselves are not executed here
because they drive a live simulator. ``_reset_v2x_state()``, the helper that
``reset()`` delegates to, *is* exercised.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from envs.carla_env import KEEP, CarlaMultiAgentEnv
from envs.obs_builder import (
    CAND,
    CAND_FEATURES,
    CANDIDATE_RANGE,
    EGO,
    M_CANDIDATES,
    MASK,
    OBS_DIM,
    ROUTE,
    build_observation,
    make_action_mask,
)
from envs.v2x_channel import ReceivedMessage, V2XChannel, V2XConfig, INTENT

# --------------------------------------------------------------------------- #
# CARLA-free fakes
# --------------------------------------------------------------------------- #
class _Location:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def distance(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)


class _Rotation:
    def __init__(self, yaw_deg=0.0):
        self.yaw = float(yaw_deg)


class _Transform:
    def __init__(self, x, y, yaw_deg=0.0):
        self.location = _Location(x, y)
        self.rotation = _Rotation(yaw_deg)

    def get_right_vector(self):
        return _Location(1.0, 0.0)


class _Velocity:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = float(x), float(y)

    def length(self):
        return math.hypot(self.x, self.y)


class _Waypoint:
    def __init__(self, x, y, yaw_deg=0.0):
        self.transform = _Transform(x, y, yaw_deg)
        self.lane_width = 3.5
        self.is_junction = False


class _Vehicle:
    """Minimal stand-in for ``carla.Vehicle``."""

    def __init__(self, actor_id, x, y, vx=0.0, vy=0.0, yaw_deg=0.0):
        self.id = int(actor_id)
        self._location = _Location(x, y)
        self._velocity = _Velocity(vx, vy)
        self._transform = _Transform(x, y, yaw_deg)

    def get_location(self):
        return self._location

    def get_velocity(self):
        return self._velocity

    def get_transform(self):
        return self._transform

    def get_acceleration(self):
        return _Velocity(0.0, 0.0)

    def get_speed_limit(self):
        return 50.0

    def get_traffic_light(self):
        return None


class _Actors:
    def __init__(self, actors):
        self._actors = list(actors)

    def filter(self, _pattern):
        return self._actors


class _Map:
    def __init__(self, waypoint):
        self._waypoint = waypoint

    def get_waypoint(self, _location, project_to_road=True, lane_type=None):
        return self._waypoint


class _World:
    def __init__(self, actors=(), waypoint=None):
        self._actors = _Actors(actors)
        self._map = _Map(waypoint if waypoint is not None else _Waypoint(0.0, 0.0))

    def get_actors(self):
        return self._actors

    def get_map(self):
        return self._map


def _default_vehicles(n_agents, spacing=10.0):
    return {
        f"cav_{index}": _Vehicle(100 + index, spacing * index, 0.0)
        for index in range(n_agents)
    }


def _make_env(
    *,
    n_agents=2,
    vehicles=None,
    background=(),
    candidate_source="v2x",
    v2x_config=None,
    v2x_seed=None,
    v2x_broadcasters="cavs_only",
    selection_mode="legacy",
    selection_k=2,
    selection_seed=None,
    seed=0,
):
    """Build a ``CarlaMultiAgentEnv`` without touching CARLA.

    The real configuration helpers are used, so these tests validate the same
    validation and seed policy as the live constructor.
    """
    env = object.__new__(CarlaMultiAgentEnv)
    env.agents = [f"cav_{index}" for index in range(n_agents)]
    env.n_agents = n_agents
    env.obs_dim = OBS_DIM
    env.seed = seed
    env.town = "Town03"
    env.step_count = 0
    env.max_steps = 1000
    env._configure_selection(selection_mode, selection_k, selection_seed)
    env._configure_candidate_source(
        candidate_source, v2x_config, v2x_seed, v2x_broadcasters, seed
    )
    env.finished = {agent_id: False for agent_id in env.agents}
    env.reached = {agent_id: False for agent_id in env.agents}
    env.route_applied = {agent_id: True for agent_id in env.agents}
    env.stop_reason = {agent_id: "none" for agent_id in env.agents}
    env.off_road_steps = {agent_id: 0 for agent_id in env.agents}
    env.stuck_steps = {agent_id: 0 for agent_id in env.agents}
    env.arrival_step = {agent_id: None for agent_id in env.agents}
    env.world = _World(actors=background)
    env._lane_context = lambda _waypoint: (None, None, 0, 1)
    vehicles = _default_vehicles(n_agents) if vehicles is None else vehicles
    env.vehicles = dict(vehicles)
    env.route = {
        agent_id: [_Waypoint(0.0, 0.0), _Waypoint(10.0, 0.0)]
        for agent_id in env.vehicles
    }
    env.goal = {
        agent_id: _Location(10.0, 0.0) for agent_id in env.vehicles
    }
    return env


def _candidate_block(obs):
    return obs[EGO + ROUTE : EGO + ROUTE + CAND].reshape(M_CANDIDATES, CAND_FEATURES)


def _assert_valid_observation(obs):
    assert obs.shape == (OBS_DIM,)
    assert obs.shape == (56,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


# --------------------------------------------------------------------------- #
# 1. ground-truth default / backward compatibility
# --------------------------------------------------------------------------- #
def test_ground_truth_is_the_default_and_has_no_channel():
    signature = inspect.signature(CarlaMultiAgentEnv.__init__)
    assert signature.parameters["candidate_source"].default == "ground_truth"
    assert signature.parameters["v2x_broadcasters"].default == "cavs_and_background"

    env = _make_env(candidate_source="ground_truth")
    assert env.candidate_source == "ground_truth"
    assert env._v2x_channel is None

    record = {
        "actor_id": 7,
        "x": 10.0,
        "y": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "yaw": 0.0,
    }
    env.get_candidates = lambda _agent_id: [dict(record)]
    obs = env._get_obs()
    _assert_valid_observation(obs["cav_0"])
    assert np.count_nonzero(_candidate_block(obs["cav_0"])[:, 5] == 1.0) == 1


def test_invalid_candidate_source_and_broadcasters_are_rejected():
    with pytest.raises(ValueError):
        _make_env(candidate_source="telepathy")
    with pytest.raises(ValueError):
        _make_env(v2x_broadcasters="everyone")
    with pytest.raises(TypeError):
        _make_env(v2x_config={"range_m": 50.0})


# --------------------------------------------------------------------------- #
# 2. V2X perfect channel
# --------------------------------------------------------------------------- #
def test_v2x_perfect_channel_builds_candidates():
    env = _make_env(
        n_agents=2,
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
    )
    obs = env._get_obs()

    for agent_id, actor_id in (("cav_0", 100), ("cav_1", 101)):
        meta = env._last_candidate_meta[agent_id]
        assert meta["received_count"] == 1
        assert meta["received_ids"] == [101 if actor_id == 100 else 100]
        assert meta["received_ages"] == [0]
        _assert_valid_observation(obs[agent_id])
        assert np.count_nonzero(_candidate_block(obs[agent_id])[:, 5] == 1.0) == 1

    # 2 senders -> 2 directed transmissions (self-reception excluded)
    assert env._v2x_stats["sent"] == 2
    assert env._v2x_stats["delivered"] == 2
    assert env._v2x_stats["dropped_by_range"] == 0
    assert env._v2x_stats["dropped_by_loss"] == 0


def test_v2x_stats_are_exactly_the_channel_statistics():
    env = _make_env(
        n_agents=2,
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
    )
    env._get_obs()
    assert set(env._v2x_stats) == {
        "sent",
        "dropped_by_range",
        "dropped_by_loss",
        "delivered",
        "in_flight",
    }


# --------------------------------------------------------------------------- #
# 3. communication range
# --------------------------------------------------------------------------- #
def test_v2x_communication_range_excludes_sender():
    env = _make_env(
        n_agents=2,
        v2x_config=V2XConfig(range_m=5.0, delay_steps=0, loss_prob=0.0),
    )
    obs = env._get_obs()
    for agent_id in env.agents:
        assert env._last_candidate_meta[agent_id]["received_count"] == 0
        assert np.all(_candidate_block(obs[agent_id]) == 0.0)
    assert env._v2x_stats["dropped_by_range"] == 2
    assert env._v2x_stats["delivered"] == 0


# --------------------------------------------------------------------------- #
# 4. candidate range is independent of communication range
# --------------------------------------------------------------------------- #
def test_v2x_candidate_range_excludes_beyond_50m_but_not_reception():
    # 70 m apart: inside a 100 m communication range, outside the 50 m
    # candidate observation radius.
    env = _make_env(
        n_agents=2,
        vehicles={
            "cav_0": _Vehicle(100, 0.0, 0.0),
            "cav_1": _Vehicle(101, 70.0, 0.0),
        },
        v2x_config=V2XConfig(range_m=100.0, delay_steps=0, loss_prob=0.0),
    )
    obs = env._get_obs()
    for agent_id in env.agents:
        meta = env._last_candidate_meta[agent_id]
        assert meta["received_count"] == 1          # communication succeeded
        assert meta["received_ids"] == [100 if agent_id == "cav_1" else 101]
        assert np.all(_candidate_block(obs[agent_id]) == 0.0)  # but not a candidate
    assert env._v2x_stats["delivered"] == 2
    assert env._v2x_stats["dropped_by_range"] == 0

    # Adding a vehicle at 30 m: received *and* admitted as a candidate.
    env = _make_env(
        n_agents=3,
        vehicles={
            "cav_0": _Vehicle(100, 0.0, 0.0),
            "cav_1": _Vehicle(101, 70.0, 0.0),
            "cav_2": _Vehicle(102, 30.0, 0.0),
        },
        v2x_config=V2XConfig(range_m=100.0, delay_steps=0, loss_prob=0.0),
    )
    obs = env._get_obs()
    block = _candidate_block(obs["cav_0"])
    assert np.count_nonzero(block[:, 5] == 1.0) == 1
    assert block[0, 0] == pytest.approx(30.0 / CANDIDATE_RANGE)
    assert env._last_candidate_meta["cav_0"]["received_ids"] == [101, 102]
    pool = env._v2x_candidates("cav_0", env._last_received)
    assert [candidate["actor_id"] for candidate in pool] == [102]


# --------------------------------------------------------------------------- #
# 5. packet loss
# --------------------------------------------------------------------------- #
def test_v2x_packet_loss_removes_candidate():
    env = _make_env(
        n_agents=2,
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=1.0),
    )
    obs = env._get_obs()
    for agent_id in env.agents:
        assert env._last_candidate_meta[agent_id]["received_count"] == 0
        assert np.all(_candidate_block(obs[agent_id]) == 0.0)
    assert env._v2x_stats["dropped_by_loss"] == 2
    assert env._v2x_stats["delivered"] == 0


# --------------------------------------------------------------------------- #
# 6. delay semantics
# --------------------------------------------------------------------------- #
def test_v2x_delay_arrival_follows_channel_semantics():
    env = _make_env(
        n_agents=2,
        v2x_config=V2XConfig(range_m=50.0, delay_steps=2, loss_prob=0.0),
    )
    env._get_obs(advance_v2x=False)          # reset-like observation, no step
    assert env._v2x_channel.step_index == 0

    env._get_obs()                            # decision step 1
    assert env._v2x_channel.step_index == 1
    assert env._last_candidate_meta["cav_0"]["received_count"] == 0

    env._get_obs()                            # decision step 2
    assert env._v2x_channel.step_index == 2
    assert env._last_candidate_meta["cav_0"]["received_count"] == 0

    obs = env._get_obs()                      # decision step 3: age 2 arrives
    meta = env._last_candidate_meta["cav_0"]
    assert meta["received_count"] == 1
    assert meta["received_ages"] == [2]
    _assert_valid_observation(obs["cav_0"])


# --------------------------------------------------------------------------- #
# 7. one channel advancement per decision step
# --------------------------------------------------------------------------- #
def test_v2x_channel_advances_once_per_decision_step():
    env = _make_env(
        n_agents=4,
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
    )
    calls = []
    real_step = env._v2x_channel.step

    def counting_step(senders):
        calls.append(len(senders))
        return real_step(senders)

    env._v2x_channel.step = counting_step

    env._get_obs(advance_v2x=False)           # reset path never advances
    assert calls == []

    env._get_obs()
    env._get_obs()
    assert calls == [4, 4]                    # once per step, all 4 agents


# --------------------------------------------------------------------------- #
# 8. self-reception
# --------------------------------------------------------------------------- #
def test_v2x_no_self_reception():
    env = _make_env(
        n_agents=3,
        v2x_config=V2XConfig(range_m=100.0, delay_steps=0, loss_prob=0.0),
    )
    env._get_obs()
    own_ids = {"cav_0": 100, "cav_1": 101, "cav_2": 102}
    for agent_id, actor_id in own_ids.items():
        assert actor_id not in env._last_candidate_meta[agent_id]["received_ids"]


# --------------------------------------------------------------------------- #
# 9. duplicate senders
# --------------------------------------------------------------------------- #
def test_v2x_senders_are_unique_and_duplicates_collapse_to_freshest():
    env = _make_env(n_agents=2)
    senders = env._build_v2x_senders()
    assert len({sender_id for sender_id, _ in senders}) == len(senders)

    message = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    received = {
        100: [
            ReceivedMessage(101, 5, message),
            ReceivedMessage(101, 1, message),
        ]
    }
    pool = env._v2x_candidates("cav_0", received)
    assert [candidate["actor_id"] for candidate in pool] == [101]
    assert pool[0]["age"] == 1


# --------------------------------------------------------------------------- #
# 10. empty inbox / missing slots
# --------------------------------------------------------------------------- #
def test_v2x_empty_inbox_pads_missing_candidate_slots():
    env = _make_env(
        n_agents=2,
        selection_mode="nearest_k",
        selection_k=2,
        v2x_config=V2XConfig(range_m=5.0, delay_steps=0, loss_prob=0.0),
    )
    obs = env._get_obs()
    block = _candidate_block(obs["cav_0"])
    assert np.all(block == 0.0)
    selection = env._last_selection["cav_0"]
    assert selection.pool_size == 0
    assert selection.selected_count == 0
    assert selection.missing_slots == 2


# --------------------------------------------------------------------------- #
# 11. numeric actor / sender ids
# --------------------------------------------------------------------------- #
def test_v2x_actor_ids_stay_numeric_and_order_ties_by_id():
    env = _make_env(
        n_agents=3,
        vehicles={
            "cav_0": _Vehicle(100, 0.0, 0.0),
            "cav_1": _Vehicle(101, 10.0, 0.0),
            "cav_2": _Vehicle(102, 10.0, 0.0),
        },
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
    )
    env._get_obs()
    senders = env._build_v2x_senders()
    assert all(isinstance(sender_id, int) for sender_id, _ in senders)

    meta = env._last_candidate_meta["cav_0"]
    assert meta["received_ids"] == [101, 102]
    assert all(isinstance(sender_id, int) for sender_id in meta["received_ids"])

    pool = env._v2x_candidates("cav_0", env._last_received)
    assert [candidate["actor_id"] for candidate in pool] == [101, 102]
    assert all(isinstance(candidate["actor_id"], int) for candidate in pool)


# --------------------------------------------------------------------------- #
# 12. selectors operate on V2X candidates
# --------------------------------------------------------------------------- #
def test_v2x_nearest_k_selects_from_received_candidates():
    env = _make_env(
        n_agents=3,
        vehicles={
            "cav_0": _Vehicle(100, 0.0, 0.0),
            "cav_1": _Vehicle(101, 10.0, 0.0),
            "cav_2": _Vehicle(102, 5.0, 0.0),
        },
        selection_mode="nearest_k",
        selection_k=1,
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
    )
    env._get_obs()
    selection = env._last_selection["cav_0"]
    assert selection.strategy == "nearest_k"
    assert selection.pool_size == 2
    assert selection.ids == (102,)
    assert selection.selected_count == 1
    assert selection.missing_slots == 0


def test_v2x_random_k_is_reproducible_under_selection_seed():
    def run(seed):
        env = _make_env(
            n_agents=5,
            selection_mode="random_k",
            selection_k=1,
            selection_seed=seed,
            v2x_config=V2XConfig(range_m=200.0, delay_steps=0, loss_prob=0.0),
        )
        trace = []
        for _ in range(10):
            env._get_obs()
            trace.append(env._last_selection["cav_0"].ids)
        return trace

    assert run(7) == run(7)
    assert run(7) != run(8)


# --------------------------------------------------------------------------- #
# 13/14. observation contract and action mask
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "v2x_config",
    [
        V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
        V2XConfig(range_m=50.0, delay_steps=2, loss_prob=0.5),
        V2XConfig(range_m=100.0, delay_steps=1, loss_prob=0.0),
    ],
)
def test_v2x_observation_contract_is_stable(v2x_config):
    env = _make_env(
        n_agents=3,
        selection_mode="nearest_k",
        selection_k=2,
        v2x_config=v2x_config,
        v2x_seed=11,
    )
    for _ in range(6):
        obs = env._get_obs()
        for agent_id in env.agents:
            _assert_valid_observation(obs[agent_id])
            assert obs[agent_id].shape == (56,)


def test_v2x_action_mask_stays_separate_from_candidates():
    env = _make_env(
        n_agents=2,
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
    )
    obs = env._get_obs()
    expected = make_action_mask(False, False).astype(np.float32)
    for agent_id in env.agents:
        assert np.array_equal(obs[agent_id][-MASK:], expected)
        # candidate selection never changes the mask
        env._last_candidate_meta.pop(agent_id, None)
    assert np.array_equal(obs["cav_0"][-MASK:], expected)


# --------------------------------------------------------------------------- #
# 15. reset clears channel state and stale messages
# --------------------------------------------------------------------------- #
def test_reset_clears_channel_state_and_stale_messages():
    env = _make_env(
        n_agents=2,
        v2x_config=V2XConfig(range_m=50.0, delay_steps=3, loss_prob=0.0),
    )
    env._get_obs()
    env._get_obs()
    assert env._v2x_channel.in_flight > 0

    env._reset_v2x_state()

    assert env._v2x_channel.in_flight == 0
    assert env._v2x_channel.step_index == 0
    assert env._v2x_stats is None
    assert env._last_received == {}
    assert env._last_candidate_meta == {}
    assert env._last_action == {"cav_0": KEEP, "cav_1": KEEP}

    # No stale message can be delivered after the reset.
    env._get_obs()
    assert env._last_candidate_meta["cav_0"]["received_count"] == 0


def test_reset_does_not_advance_the_channel():
    env = _make_env(
        n_agents=2,
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
    )
    env._get_obs(advance_v2x=False)
    assert env._v2x_channel.step_index == 0
    assert env._v2x_channel.episode_stats()["sent"] == 0


# --------------------------------------------------------------------------- #
# 16. reproducibility
# --------------------------------------------------------------------------- #
def _trace(env, steps=10):
    trace = []
    for _ in range(steps):
        env._get_obs()
        trace.append(
            (
                tuple(
                    sorted(
                        (agent_id, tuple(meta["received_ids"]), tuple(meta["received_ages"]))
                        for agent_id, meta in env._last_candidate_meta.items()
                    )
                ),
                dict(env._v2x_stats),
            )
        )
    return trace


def test_v2x_seed_reproducibility():
    def run(seed):
        env = _make_env(
            n_agents=3,
            v2x_config=V2XConfig(range_m=50.0, delay_steps=1, loss_prob=0.4),
            v2x_seed=seed,
        )
        return _trace(env)

    assert run(42) == run(42)
    assert run(42) != run(99)


def test_v2x_seed_defaults_to_environment_seed():
    env = _make_env(n_agents=2, seed=123, v2x_seed=None)
    assert env._v2x_seed == 123
    explicit = _make_env(n_agents=2, seed=123, v2x_seed=5)
    assert explicit._v2x_seed == 5


# --------------------------------------------------------------------------- #
# background vehicles / finished agents (population semantics)
# --------------------------------------------------------------------------- #
def test_background_vehicles_are_broadcast_only_when_configured():
    background = [_Vehicle(500, 5.0, 0.0)]
    with_background = _make_env(
        n_agents=1,
        background=background,
        v2x_broadcasters="cavs_and_background",
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
    )
    with_background._get_obs()
    assert with_background._last_candidate_meta["cav_0"]["received_ids"] == [500]

    cavs_only = _make_env(
        n_agents=1,
        background=background,
        v2x_broadcasters="cavs_only",
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
    )
    cavs_only._get_obs()
    assert cavs_only._last_candidate_meta["cav_0"]["received_count"] == 0


def test_background_message_uses_documented_placeholder_intent():
    env = _make_env(
        n_agents=1,
        background=[_Vehicle(500, 5.0, 0.0)],
        v2x_broadcasters="cavs_and_background",
    )
    messages = dict(env._build_v2x_senders())
    assert messages[500][INTENT] == 0.0
    assert messages[100][INTENT] == float(KEEP)


def test_finished_agent_is_not_a_candidate_or_sender():
    env = _make_env(
        n_agents=2,
        v2x_config=V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0),
    )
    env.finished["cav_1"] = True
    env._get_obs()
    assert env._last_candidate_meta["cav_0"]["received_count"] == 0
    assert 101 not in {sender_id for sender_id, _ in env._build_v2x_senders()}


# --------------------------------------------------------------------------- #
# adapter mapping sanity (message -> world-frame record)
# --------------------------------------------------------------------------- #
def test_received_message_maps_to_world_frame_candidate():
    env = _make_env(n_agents=2)
    message = np.array(
        [10.0, 2.0, 1.0, 0.5, 0.25, 3.0], dtype=np.float64
    )
    received = {100: [ReceivedMessage(101, 2, message)]}
    pool = env._v2x_candidates("cav_0", received)
    assert len(pool) == 1
    candidate = pool[0]
    assert candidate["actor_id"] == 101
    assert candidate["x"] == 10.0 and candidate["y"] == 2.0
    assert candidate["vx"] == 1.0 and candidate["vy"] == 0.5
    assert candidate["yaw"] == 0.25
    assert candidate["age"] == 2
    assert candidate["intent"] == 3.0
    # Raw world coordinates: normalization is the observation builder's job.
    assert candidate["x"] != pytest.approx(10.0 / CANDIDATE_RANGE)

    obs = build_observation(
        {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "yaw": 0.0},
        {
            "next_waypoint_x": 10.0,
            "next_waypoint_y": 0.0,
            "speed_limit": 15.0,
            "at_intersection": False,
            "traffic_light_state": "none",
            "distance_to_next": 10.0,
        },
        pool,
        make_action_mask(False, False),
    )
    _assert_valid_observation(obs)
    assert obs[EGO + ROUTE] == pytest.approx(10.0 / CANDIDATE_RANGE)
    assert obs[EGO + ROUTE + 5] == 1.0
