"""Contract test. Every env (toy, highway adapter, CARLA) must pass it.

    pytest tests\test_contract.py -v                 all envs available on this PC
    pytest tests\test_contract.py -v -k toy          only the toy env
    pytest tests\test_contract.py -v -k carla        only CARLA (server must be running)

Core checks always run. STRICT adds attributes, seeding and info keys:
    set CONTRACT_STRICT=1
    pytest tests\test_contract.py -v
(Use two separate lines in cmd; "set X=1 && pytest" would store "1 " with a trailing space.)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from envs.base import REQUIRED_EPISODE_INFO, REQUIRED_STEP_INFO

STRICT = os.environ.get("CONTRACT_STRICT", "").strip() == "1"
strict_only = pytest.mark.skipif(not STRICT, reason="set CONTRACT_STRICT=1 to run")
MAX_CHECK_STEPS = 400


# ---- env factories (skip cleanly when an env is not available on this PC) ---
def _make_toy():
    from envs.toy_env import ToyEnv
    return ToyEnv()


def _make_highway():
    try:
        from envs.highway_adapter import HighwayAdapter   # class name is part of the contract
    except ImportError as e:
        pytest.skip(f"highway adapter not available: {e}")
    return HighwayAdapter()


def _make_carla():
    pytest.importorskip("carla")
    from envs.carla_env import CarlaMultiAgentEnv
    try:
        return CarlaMultiAgentEnv(town="Town03", n_agents=2)
    except RuntimeError as e:
        pytest.skip(f"CARLA server not reachable: {e}")


FACTORIES = {"toy": _make_toy, "highway": _make_highway, "carla": _make_carla}


@pytest.fixture(params=list(FACTORIES))
def env(request):
    e = FACTORIES[request.param]()
    yield e
    if hasattr(e, "close"):
        e.close()


# ---- helpers ----------------------------------------------------------------
def _reset(env, seed=0):
    out = env.reset(seed=seed)
    assert isinstance(out, tuple) and len(out) == 2, "reset() must return (obs, info)"
    obs, info = out
    assert isinstance(obs, dict) and isinstance(info, dict), "obs and info must be dicts"
    return obs, info


def _random_actions(env):
    masks = env.action_masks()
    return {a: int(np.random.choice(np.flatnonzero(masks[a]))) for a in env.agents}


def _ended(term, trunc):
    return all(term.values()) or any(trunc.values())


# ---- core checks ------------------------------------------------------------
def test_reset_format(env):
    obs, _ = _reset(env)
    assert set(obs) == set(env.agents), "obs keys must equal env.agents"
    shapes = {np.asarray(v).shape for v in obs.values()}
    assert len(shapes) == 1 and len(next(iter(shapes))) == 1, "all obs must be 1-D, same length"
    for v in obs.values():
        assert np.all(np.isfinite(np.asarray(v, dtype=np.float64))), "obs must be finite"


def test_action_masks(env):
    _reset(env)
    masks = env.action_masks()
    assert set(masks) == set(env.agents)
    assert len({np.asarray(m).shape for m in masks.values()}) == 1
    for m in masks.values():
        m = np.asarray(m)
        assert m.dtype == bool and m.ndim == 1 and m.any(), "mask: 1-D bool, at least one True"


def test_state(env):
    _reset(env)
    s = np.asarray(env.state())
    assert s.ndim == 1 and np.all(np.isfinite(s)), "state() must be a finite 1-D vector"
    assert np.asarray(env.state()).shape == s.shape


def test_step_format(env):
    _reset(env)
    out = env.step(_random_actions(env))
    assert len(out) == 5, "step() must return (obs, rewards, terminated, truncated, info)"
    obs, rew, term, trunc, info = out
    assert set(obs) == set(env.agents)
    for name, d in (("rewards", rew), ("terminated", term), ("truncated", trunc)):
        assert isinstance(d, dict), f"{name} must be a dict keyed by agent id"
        assert set(d) == set(env.agents), f"{name} keys must equal env.agents"
    assert all(np.isfinite(r) for r in rew.values()), "rewards must be finite"
    assert all(isinstance(v, (bool, np.bool_)) for v in term.values())
    assert all(isinstance(v, (bool, np.bool_)) for v in trunc.values())
    assert isinstance(info, dict)


def test_episode_terminates(env):
    _reset(env, seed=1)
    for _ in range(MAX_CHECK_STEPS):
        _, _, term, trunc, _ = env.step(_random_actions(env))
        if _ended(term, trunc):
            return
    pytest.fail(f"episode did not end within {MAX_CHECK_STEPS} steps")


# ---- strict checks ----------------------------------------------------------
@strict_only
def test_attributes(env):
    for name in ("agents", "obs_dim", "n_actions"):
        assert hasattr(env, name), f"env must expose .{name}"
    obs, _ = _reset(env)
    assert np.asarray(obs[env.agents[0]]).shape == (env.obs_dim,)
    assert len(env.action_masks()[env.agents[0]]) == env.n_actions


@strict_only
def test_seed_determinism(env):
    a, _ = _reset(env, seed=7)
    b, _ = _reset(env, seed=7)
    for ag in env.agents:
        assert np.allclose(a[ag], b[ag], atol=1e-3), "same seed must give the same start"


@strict_only
def test_seed_changes_episode(env):
    a, _ = _reset(env, seed=1)
    b, _ = _reset(env, seed=2)
    assert any(not np.allclose(a[ag], b[ag], atol=1e-6) for ag in env.agents), \
        "different seeds must give different starts (is the seed used?)"


@strict_only
def test_info_keys(env):
    _reset(env, seed=3)
    for _ in range(MAX_CHECK_STEPS):
        _, _, term, trunc, info = env.step(_random_actions(env))
        for k in REQUIRED_STEP_INFO:
            assert k in info, f"info is missing '{k}'"
        if _ended(term, trunc):
            for k in REQUIRED_EPISODE_INFO:
                assert k in info, f"info is missing '{k}' on the last step of the episode"
            return
    pytest.fail(f"episode did not end within {MAX_CHECK_STEPS} steps")