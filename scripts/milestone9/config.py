"""Central, validated configuration for Milestone 9 paired experiments.

Additive infrastructure only. Does not modify environment contracts, rewards,
termination semantics, observation layout, V2X channel implementation, or MARL
algorithms.

Seed policy (explicit, never silently shared across subsystems):
- scenario_seed: controls environment/spawn/route/Traffic Manager scenario setup
  (passed to env/runner as the episode scenario seed).
- selection_seed: controls random-k teammate selection RNG only.
- v2x_seed: controls V2X channel RNG only (relevant if stochastic V2X enabled).

For each paired episode, scenario_seed is identical across strategies; selection
seeds are explicit and reproducible per episode/strategy; v2x seeds are explicit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

SCENARIO_SEEDS_DEFAULT: tuple[int, ...] = (7, 17, 27, 37, 47)
STRATEGIES_DEFAULT: tuple[str, ...] = ("nearest_k", "random_k")

CONTROLLER_MODULE = "controllers.observation_heuristic"
CONTROLLER_CLASS = "ObservationHeuristicController"


def _validate_positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_float_range(name: str, value: Any, *, low: float, high: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number in [{low}, {high}]") from None
    if not (low <= v <= high):
        raise ValueError(f"{name} must be a number in [{low}, {high}]")
    return v


def stable_seed(*parts: Any, modulo: int = 2**31 - 1) -> int:
    """Deterministically derive a non-negative seed from arbitrary parts.

    Uses SHA-256 over a canonical JSON encoding for stability across runs.
    """
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(modulo)


@dataclass(frozen=True)
class EnvironmentConfig:
    town: str = "Town03"
    n_agents: int = 4
    spawn_mode: str = "cluster"
    spawn_cluster_max_distance_m: float = 45.0
    candidate_source: str = "v2x"
    candidate_radius_m: float = 50.0
    selection_k: int = 2
    max_steps: int = 300  # decision steps (5 Hz), 60 s baseline
    sim_hz: int = 20
    decision_hz: int = 5
    obs_dim: int = 56
    n_actions: int = 6
    max_candidates: int = 6

    def validate(self) -> "EnvironmentConfig":
        if self.town != "Town03":
            raise ValueError("environment.town must be 'Town03' for this milestone config")
        if self.n_agents != 4:
            raise ValueError("environment.n_agents must be 4")
        if self.spawn_mode != "cluster":
            raise ValueError("environment.spawn_mode must be 'cluster'")
        if float(self.spawn_cluster_max_distance_m) != 45.0:
            raise ValueError("environment.spawn_cluster_max_distance_m must be 45.0")
        if self.candidate_source != "v2x":
            raise ValueError("environment.candidate_source must be 'v2x'")
        if float(self.candidate_radius_m) != 50.0:
            raise ValueError("environment.candidate_radius_m must be 50.0 (existing)")
        _validate_positive_int("environment.selection_k", self.selection_k)
        if self.selection_k != 2:
            raise ValueError("environment.selection_k must be 2")
        _validate_positive_int("environment.max_steps", self.max_steps)
        if self.max_steps != 300:
            raise ValueError("environment.max_steps must be 300 for baseline protocol")
        if self.sim_hz != 20 or self.decision_hz != 5:
            raise ValueError("environment.sim_hz/decision_hz must be 20/5")
        if self.obs_dim != 56 or self.n_actions != 6 or self.max_candidates != 6:
            raise ValueError("observation/action/candidate contract must remain 56/6/6")
        return self


@dataclass(frozen=True)
class V2XExperimentConfig:
    communication_range_m: float = 50.0
    packet_loss: float = 0.0
    delay_steps: int = 0
    broadcasters: str = "cavs_and_background"

    def validate(self) -> "V2XExperimentConfig":
        if float(self.communication_range_m) != 50.0:
            raise ValueError("v2x.communication_range_m must be 50.0")
        loss = _validate_float_range("v2x.packet_loss", self.packet_loss, low=0.0, high=1.0)
        if loss != 0.0:
            raise ValueError("v2x.packet_loss must be 0.0 for this milestone config")
        _validate_non_negative_int("v2x.delay_steps", self.delay_steps)
        if self.delay_steps != 0:
            raise ValueError("v2x.delay_steps must be 0 for this milestone config")
        if self.broadcasters != "cavs_and_background":
            raise ValueError("v2x.broadcasters must be 'cavs_and_background'")
        return self


@dataclass(frozen=True)
class ControllerConfig:
    module: str = CONTROLLER_MODULE
    class_name: str = CONTROLLER_CLASS
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "ControllerConfig":
        if self.module != CONTROLLER_MODULE or self.class_name != CONTROLLER_CLASS:
            raise ValueError(
                "controller must be controllers.observation_heuristic.ObservationHeuristicController"
            )
        if self.parameters is None or not isinstance(self.parameters, Mapping):
            raise ValueError("controller.parameters must be a mapping")
        # Parameters must be identical for both strategies (enforced by single config).
        return self

    def identity(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "class": self.class_name,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class SeedConfig:
    scenario_seeds: tuple[int, ...] = SCENARIO_SEEDS_DEFAULT
    selection_seed_base: int = 10_000
    v2x_seed_base: int = 20_000
    # Optional explicit overrides: mapping key "(scenario_seed, episode_index, strategy)" or
    # "scenario_seed:episode_index:strategy" is supported by helpers; direct maps may be used.
    explicit_selection_seeds: Mapping[str, int] = field(default_factory=dict)
    explicit_v2x_seeds: Mapping[str, int] = field(default_factory=dict)

    def validate(self) -> "SeedConfig":
        if not self.scenario_seeds:
            raise ValueError("seeds.scenario_seeds must be non-empty")
        seen = set()
        for s in self.scenario_seeds:
            _validate_non_negative_int("seeds.scenario_seeds[]", s)
            if s in seen:
                raise ValueError("seeds.scenario_seeds must not contain duplicates")
            seen.add(s)
        _validate_non_negative_int("seeds.selection_seed_base", self.selection_seed_base)
        _validate_non_negative_int("seeds.v2x_seed_base", self.v2x_seed_base)
        for name, mapping in (
            ("seeds.explicit_selection_seeds", self.explicit_selection_seeds),
            ("seeds.explicit_v2x_seeds", self.explicit_v2x_seeds),
        ):
            if mapping is None or not isinstance(mapping, Mapping):
                raise ValueError(f"{name} must be a mapping")
            for k, v in mapping.items():
                if not isinstance(k, str):
                    raise ValueError(f"{name} keys must be strings")
                _validate_non_negative_int(f"{name}[{k}]", v)
        return self


@dataclass(frozen=True)
class Milestone9Config:
    experiment_label: str = "milestone9_paired"
    strategies: tuple[str, ...] = STRATEGIES_DEFAULT
    episodes_per_seed: int = 1
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    v2x: V2XExperimentConfig = field(default_factory=V2XExperimentConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    host: str = "localhost"
    port: int = 2000
    tm_port: int = 8000
    carla_version: str = "0.9.16"

    def validate(self) -> "Milestone9Config":
        if tuple(self.strategies) != STRATEGIES_DEFAULT:
            raise ValueError("strategies must be exactly ('nearest_k', 'random_k')")
        _validate_positive_int("episodes_per_seed", self.episodes_per_seed)
        self.environment.validate()
        self.v2x.validate()
        self.controller.validate()
        self.seeds.validate()
        _validate_positive_int("port", self.port)
        _validate_positive_int("tm_port", self.tm_port)
        if self.environment.candidate_radius_m != self.v2x.communication_range_m:
            # Baseline protocol keeps candidate observation radius and comm range aligned.
            raise ValueError("candidate_radius_m and v2x.communication_range_m must match")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""
        d = asdict(self)
        # normalize tuples
        d["strategies"] = list(self.strategies)
        d["seeds"]["scenario_seeds"] = list(self.seeds.scenario_seeds)
        d["controller"]["parameters"] = dict(self.controller.parameters)
        d["seeds"]["explicit_selection_seeds"] = dict(self.seeds.explicit_selection_seeds)
        d["seeds"]["explicit_v2x_seeds"] = dict(self.seeds.explicit_v2x_seeds)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Milestone9Config":
        env = EnvironmentConfig(**dict(data.get("environment", {})))
        v2x = V2XExperimentConfig(**dict(data.get("v2x", {})))
        ctrl_data = dict(data.get("controller", {}))
        controller = ControllerConfig(
            module=ctrl_data.get("module", CONTROLLER_MODULE),
            class_name=ctrl_data.get("class_name", ctrl_data.get("class", CONTROLLER_CLASS)),
            parameters=dict(ctrl_data.get("parameters", {})),
        )
        seed_data = dict(data.get("seeds", {}))
        seeds = SeedConfig(
            scenario_seeds=tuple(seed_data.get("scenario_seeds", SCENARIO_SEEDS_DEFAULT)),
            selection_seed_base=int(seed_data.get("selection_seed_base", 10_000)),
            v2x_seed_base=int(seed_data.get("v2x_seed_base", 20_000)),
            explicit_selection_seeds=dict(seed_data.get("explicit_selection_seeds", {})),
            explicit_v2x_seeds=dict(seed_data.get("explicit_v2x_seeds", {})),
        )
        cfg = cls(
            experiment_label=str(data.get("experiment_label", "milestone9_paired")),
            strategies=tuple(data.get("strategies", STRATEGIES_DEFAULT)),
            episodes_per_seed=int(data.get("episodes_per_seed", 1)),
            environment=env,
            v2x=v2x,
            controller=controller,
            seeds=seeds,
            host=str(data.get("host", "localhost")),
            port=int(data.get("port", 2000)),
            tm_port=int(data.get("tm_port", 8000)),
            carla_version=str(data.get("carla_version", "0.9.16")),
        )
        return cfg.validate()


def seed_key(scenario_seed: int, episode_index: int, strategy: str) -> str:
    return f"{int(scenario_seed)}:{int(episode_index)}:{strategy}"


def derive_selection_seed(cfg: Milestone9Config, scenario_seed: int, episode_index: int, strategy: str) -> int:
    """Return explicit, reproducible selection seed for an episode.

    Nearest-k selection is deterministic and does not consume RNG; a stable seed
    is still derived/logged for completeness. Random-k requires an explicit
    numpy.random.Generator seed: the runner must provide a fresh per-episode
    selector stream (see run_paired / seed-management strategy).
    """
    key = seed_key(scenario_seed, episode_index, strategy)
    if key in cfg.seeds.explicit_selection_seeds:
        return int(cfg.seeds.explicit_selection_seeds[key])
    return stable_seed(cfg.seeds.selection_seed_base, scenario_seed, episode_index, strategy, "selection")


def derive_v2x_seed(cfg: Milestone9Config, scenario_seed: int, episode_index: int, strategy: str) -> int:
    key = seed_key(scenario_seed, episode_index, strategy)
    if key in cfg.seeds.explicit_v2x_seeds:
        return int(cfg.seeds.explicit_v2x_seeds[key])
    # V2X channel RNG is independent of selection RNG; include strategy for explicit per-run identity.
    return stable_seed(cfg.seeds.v2x_seed_base, scenario_seed, episode_index, strategy, "v2x")


def pair_id(experiment_id: str, scenario_seed: int, episode_index: int) -> str:
    return f"{experiment_id}__seed{int(scenario_seed)}__ep{int(episode_index)}"


def episode_id(experiment_id: str, scenario_seed: int, episode_index: int, strategy: str) -> str:
    return f"{pair_id(experiment_id, scenario_seed, episode_index)}__{strategy}"


def default_config() -> Milestone9Config:
    """Return the baseline Milestone 9 paired-experiment configuration."""
    return Milestone9Config().validate()


__all__ = [
    "CONTROLLER_CLASS",
    "CONTROLLER_MODULE",
    "ControllerConfig",
    "EnvironmentConfig",
    "Milestone9Config",
    "SCENARIO_SEEDS_DEFAULT",
    "STRATEGIES_DEFAULT",
    "SeedConfig",
    "V2XExperimentConfig",
    "default_config",
    "derive_selection_seed",
    "derive_v2x_seed",
    "episode_id",
    "pair_id",
    "seed_key",
    "stable_seed",
]
