from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.milestone9.config import Milestone9Config, default_config
from scripts.milestone9.logging import ExperimentLayout
from scripts.milestone9.run_paired import run_episode, run_paired_experiment


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class FakeRotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)


class FakeTransform:
    def __init__(self, x=0.0, y=0.0, z=0.0, pitch=0.0, yaw=0.0, roll=0.0):
        self.location = FakeLocation(x, y, z)
        self.rotation = FakeRotation(pitch, yaw, roll)


class FakeVelocity:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class FakeActor:
    def __init__(self, actor_id, type_id, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self.id = actor_id
        self.type_id = type_id
        self._tf = FakeTransform(x=x, y=y, z=z, yaw=yaw)
        self._vel = FakeVelocity(0.0, 0.0, 0.0)
        self.attributes = {}

    def get_transform(self):
        return self._tf

    def get_velocity(self):
        return self._vel


class FakeActorList(list):
    def filter(self, pattern):
        if pattern == "vehicle.*":
            return FakeActorList([a for a in self if str(getattr(a, "type_id", "")).startswith("vehicle")])
        if pattern == "traffic.traffic_light*":
            return FakeActorList([a for a in self if str(getattr(a, "type_id", "")).startswith("traffic.traffic_light")])
        return FakeActorList(self)


class FakeMap:
    name = "Carla/Maps/Town03"


class FakeSettings:
    synchronous_mode = True
    fixed_delta_seconds = 0.05
    no_rendering_mode = False


class FakeWeather:
    cloudiness = 10.0
    precipitation = 0.0
    precipitation_deposits = 0.0
    wind_intensity = 0.0
    sun_azimuth_angle = 0.0
    sun_altitude_angle = 45.0
    fog_density = 0.0
    fog_distance = 0.0
    wetness = 0.0


class FakeWorld:
    def get_map(self):
        return FakeMap()

    def get_settings(self):
        return FakeSettings()

    def get_weather(self):
        return FakeWeather()

    def get_actors(self):
        return FakeActorList([
            FakeActor(101, "vehicle.tesla.model3", x=1.0, y=2.0),
            FakeActor(102, "vehicle.tesla.model3", x=3.0, y=2.0),
            FakeActor(201, "traffic.traffic_light", x=20.0, y=0.0),
        ])


class FakeEnv:
    def __init__(self, *, close_raises=None, reset_raises=None, step_raises=None, route_applied=True):
        self.agents = ["cav_0", "cav_1"]
        self.max_steps = 300
        self.vehicles = {
            "cav_0": FakeActor(101, "vehicle.tesla.model3", x=1.0, y=2.0),
            "cav_1": FakeActor(102, "vehicle.tesla.model3", x=3.0, y=2.0),
        }
        self.goal = {"cav_0": FakeTransform(x=10, y=2), "cav_1": FakeTransform(x=12, y=2)}
        self.route = {
            "cav_0": [FakeLocation(1, 2, 0), FakeLocation(10, 2, 0)],
            "cav_1": [FakeLocation(3, 2, 0), FakeLocation(12, 2, 0)],
        }
        self.world = FakeWorld()
        self.town = "Town03"
        self._close_raises = close_raises
        self._reset_raises = reset_raises
        self._step_raises = step_raises
        self._route_applied = route_applied
        self._step_count = 0

    def reset(self, seed=None):
        if self._reset_raises:
            raise self._reset_raises
        obs = {aid: np.zeros(56, dtype=np.float32) for aid in self.agents}
        info = {
            "seed": seed,
            "town": "Town03",
            "spawn_mode": "cluster",
            "spawn_cluster_max_distance": 45.0,
            "route_applied": bool(self._route_applied),
            "per_agent": {aid: {"route_applied": bool(self._route_applied)} for aid in self.agents},
        }
        return obs, info

    def action_masks(self):
        return {aid: np.ones(6, dtype=bool) for aid in self.agents}

    def step(self, actions):
        if self._step_raises:
            raise self._step_raises
        self._step_count += 1
        obs = {aid: np.zeros(56, dtype=np.float32) for aid in self.agents}
        rewards = {aid: 0.0 for aid in self.agents}
        terminated = {aid: True for aid in self.agents}
        truncated = {aid: False for aid in self.agents}
        info = {
            "steps": self._step_count,
            "success": True,
            "collision": False,
            "off_road": False,
            "stuck": False,
            "team_reward": 0.0,
            "min_ttc": 10.0,
            "jerk": 0.0,
            "per_agent": {
                aid: {
                    "goal_reached": True,
                    "arrival_step": self._step_count,
                    "collision": False,
                    "off_road": False,
                    "stuck": False,
                    "selection": {"selected_ids": [1], "pool_size": 1, "selected_count": 1, "missing_slots": 1},
                    "v2x_received": {"count": 1, "sender_ids": [1], "ages": [0]},
                }
                for aid in self.agents
            },
        }
        return obs, rewards, terminated, truncated, info

    def close(self):
        if self._close_raises:
            raise self._close_raises


class FakeController:
    def act(self, obs, mask=None):
        return 0


def _patch_runtime(monkeypatch, env_factory):
    monkeypatch.setattr("scripts.milestone9.run_paired.create_env", env_factory)
    monkeypatch.setattr("scripts.milestone9.run_paired.build_controller", lambda cfg: FakeController())


def test_cleanup_failure_invalidates_episode_and_pair(tmp_path: Path, monkeypatch):
    cfg = default_config()
    _patch_runtime(monkeypatch, lambda *a, **k: FakeEnv(close_raises=RuntimeError("destroy failed")))
    layout = ExperimentLayout(tmp_path / "results", "exp_cleanup")
    summary = run_paired_experiment(
        cfg,
        experiment_id="exp_cleanup",
        results_root=tmp_path / "results",
        pairs=[(7, 0)],
        layout=layout,
    )
    assert summary["pairs_valid_for_analysis"] == 0
    assert summary["all_pairs_valid_for_analysis"] is False
    ep = next(e for e in summary["pairs"][0]["episodes"].values())
    assert ep["execution_status"] == "failed"
    # manifest should record failed/incomplete-safe final state (not completed-success)
    manifest = json.loads((layout.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] in ("failed", "incomplete")


def test_reset_failure_invalidates_episode_and_pair(tmp_path: Path, monkeypatch):
    cfg = default_config()
    _patch_runtime(monkeypatch, lambda *a, **k: FakeEnv(reset_raises=RuntimeError("reset boom")))
    layout = ExperimentLayout(tmp_path / "results", "exp_reset")
    summary = run_paired_experiment(
        cfg,
        experiment_id="exp_reset",
        results_root=tmp_path / "results",
        pairs=[(7, 0)],
        layout=layout,
    )
    assert summary["pairs_valid_for_analysis"] == 0
    assert summary["all_pairs_processed"] is True
    manifest = json.loads((layout.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_route_invalid_in_reset_info_invalidates_episode(tmp_path: Path, monkeypatch):
    cfg = default_config()
    _patch_runtime(monkeypatch, lambda *a, **k: FakeEnv(route_applied=False))
    layout = ExperimentLayout(tmp_path / "results", "exp_route")
    summary = run_paired_experiment(
        cfg,
        experiment_id="exp_route",
        results_root=tmp_path / "results",
        pairs=[(7, 0)],
        layout=layout,
    )
    assert summary["pairs_valid_for_analysis"] == 0
    pair = summary["pairs"][0]
    assert pair["valid_for_analysis"] is False
    manifest = json.loads((layout.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_successful_mocked_pair_completes_and_records_lifecycle(tmp_path: Path, monkeypatch):
    cfg = default_config()
    _patch_runtime(monkeypatch, lambda *a, **k: FakeEnv())
    layout = ExperimentLayout(tmp_path / "results", "exp_ok")
    summary = run_paired_experiment(
        cfg,
        experiment_id="exp_ok",
        results_root=tmp_path / "results",
        pairs=[(7, 0)],
        layout=layout,
    )
    assert summary["pairs_valid_for_analysis"] == 1
    assert summary["all_pairs_valid_for_analysis"] is True
    manifest = json.loads((layout.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    # episode artifacts preserved
    ep_files = list((layout.paths["episodes"]).glob("*.json"))
    assert ep_files
    ep = json.loads(ep_files[0].read_text(encoding="utf-8"))
    assert ep["lifecycle"]["cleanup_status"] in ("success", "skipped", "not_attempted")
    assert ep["lifecycle"]["live_carla_isolation_verified"] is False
    assert ep["persistence"]["status"] in ("success", "in_memory")
    assert ep["initial_scenario"]["route_applied"] is True
    assert "world" in ep["initial_scenario"]
    assert "background_and_traffic_lights" in ep["initial_scenario"]


def test_persistence_failure_invalidates_episode(tmp_path: Path, monkeypatch):
    cfg = default_config()
    _patch_runtime(monkeypatch, lambda *a, **k: FakeEnv())
    layout = ExperimentLayout(tmp_path / "results", "exp_persist")
    import scripts.milestone9.run_paired as rp

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(rp, "atomic_write_csv", boom)
    summary = run_paired_experiment(
        cfg,
        experiment_id="exp_persist",
        results_root=tmp_path / "results",
        pairs=[(7, 0)],
        layout=layout,
    )
    assert summary["pairs_valid_for_analysis"] == 0
    manifest = json.loads((layout.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_dry_run_config_still_works_without_carla(tmp_path: Path):
    from scripts.milestone9.run_paired import main

    rc = main(["--dry-run-config"])
    assert rc == 0
