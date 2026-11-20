"""Configuration-driven initialization for the four-CAV Highway-Env scenario.

This module only prepares native Highway-Env configuration and initial vehicle
placement. Observation, action, reward and simulation logic stay in the existing
adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MethodType
from typing import Any, Mapping

import yaml
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle

SCENARIO_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "scenarios.yaml"
)
AGENT_COUNT = 4
MIN_BACKGROUND_VEHICLES = 12
HIGHWAY_LANES = 4
MIN_SAFE_CLEARANCE_M = 8.0
MIN_CAV_SEPARATION_M = 12.0


@dataclass(frozen=True)
class SpeedDistribution:
    """Speed draw settings in m/s."""

    mode: str
    low: float = 0.0
    high: float = 0.0
    groups: tuple[tuple[float, float, float], ...] = ()

    def draw(self, count: int, rng: Any) -> list[float]:
        if self.mode == "uniform":
            return [float(rng.uniform(self.low, self.high)) for _ in range(count)]
        values = []
        for index in range(count):
            group_index = min(
                int(index * len(self.groups) / max(count, 1)),
                len(self.groups) - 1,
            )
            _, low, high = self.groups[group_index]
            values.append(float(rng.uniform(low, high)))
        return values


@dataclass(frozen=True)
class RoleVehicle:
    """One seeded background vehicle with a forcing role."""

    name: str
    lane: int
    x_m: float
    speed_mps: float
    brake_step: int | None = None
    brake_speed_mps: float | None = None


@dataclass(frozen=True)
class ForcingConfig:
    """Reusable forcing roles used by a scenario."""

    roles: tuple[RoleVehicle, ...] = ()

    @property
    def scheduled_hazard(self) -> RoleVehicle | None:
        for role in self.roles:
            if role.name == "scheduled_hazard":
                return role
        return None


@dataclass(frozen=True)
class ScenarioProfile:
    """Validated, immutable Highway scenario initialization profile."""

    scenario_id: str
    lanes_count: int
    background_vehicles: int
    vehicles_density: float
    duration_s: float
    spacing_m: float
    spacing_jitter_m: float
    speeds: SpeedDistribution
    cav_lanes: tuple[int, ...]
    cav_x_m: tuple[float, ...]
    cav_speed_mps: float
    forcing: ForcingConfig = ForcingConfig()

    def env_config(self) -> dict[str, Any]:
        """Native Highway-Env config keys changed by this profile."""
        return {
            "lanes_count": self.lanes_count,
            "vehicles_count": self.background_vehicles,
            "vehicles_density": self.vehicles_density,
            "duration": self.duration_s,
        }


def _require_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _parse_speeds(raw: Any) -> SpeedDistribution:
    if not isinstance(raw, Mapping):
        raise ValueError("speeds_mps must be a mapping.")
    mode = raw.get("mode")
    if mode == "uniform":
        low = _require_number(raw.get("min"), "speeds_mps.min", minimum=0.0)
        high = _require_number(raw.get("max"), "speeds_mps.max", minimum=0.0)
        if high <= low:
            raise ValueError("speeds_mps.max must exceed speeds_mps.min.")
        return SpeedDistribution(mode="uniform", low=low, high=high)
    if mode == "groups":
        raw_groups = raw.get("groups")
        if not isinstance(raw_groups, (list, tuple)) or not raw_groups:
            raise ValueError("speeds_mps.groups must be a non-empty list.")
        groups = []
        for group in raw_groups:
            if not isinstance(group, Mapping):
                raise ValueError("each speed group must be a mapping.")
            low = _require_number(group.get("min"), "group.min", minimum=0.0)
            high = _require_number(group.get("max"), "group.max", minimum=0.0)
            if high <= low:
                raise ValueError("group.max must exceed group.min.")
            groups.append((1.0, low, high))
        return SpeedDistribution(mode="groups", groups=tuple(groups))
    raise ValueError("speeds_mps.mode must be 'uniform' or 'groups'.")


def _parse_role(raw: Any, name: str) -> RoleVehicle | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"forcing.{name} must be a mapping.")
    lane = _require_int(raw.get("lane"), f"forcing.{name}.lane", minimum=0)
    if lane >= HIGHWAY_LANES:
        raise ValueError(f"forcing.{name}.lane must be below {HIGHWAY_LANES}.")
    x_m = _require_number(raw.get("x_m"), f"forcing.{name}.x_m", minimum=0.0)
    speed = _require_number(
        raw.get("speed_mps"), f"forcing.{name}.speed_mps", minimum=0.0
    )
    brake_step = raw.get("brake_step")
    brake_speed = raw.get("brake_speed_mps")
    if (brake_step is None) != (brake_speed is None):
        raise ValueError(f"forcing.{name} brake_step and brake_speed_mps must be paired.")
    if brake_step is None:
        return RoleVehicle(name=name, lane=lane, x_m=x_m, speed_mps=speed)
    return RoleVehicle(
        name=name,
        lane=lane,
        x_m=x_m,
        speed_mps=speed,
        brake_step=_require_int(brake_step, f"forcing.{name}.brake_step", minimum=1),
        brake_speed_mps=_require_number(
            brake_speed, f"forcing.{name}.brake_speed_mps", minimum=0.0
        ),
    )


def _parse_forcing(raw: Any) -> ForcingConfig:
    if raw is None:
        return ForcingConfig()
    if not isinstance(raw, Mapping):
        raise ValueError("forcing must be a mapping.")
    roles = []
    for key in ("slow_blocker", "fast_approacher", "scheduled_hazard"):
        role = _parse_role(raw.get(key), key)
        if role is not None:
            roles.append(role)
    alongside = raw.get("alongside", ())
    if not isinstance(alongside, (list, tuple)):
        raise ValueError("forcing.alongside must be a list.")
    for index, item in enumerate(alongside):
        role = _parse_role(item, f"alongside_{index}")
        if role is not None:
            roles.append(role)
    return ForcingConfig(roles=tuple(roles))


def _validate_positions(
    cav_lanes: tuple[int, ...],
    cav_x: tuple[float, ...],
    roles: tuple[RoleVehicle, ...],
) -> None:
    occupied: dict[int, list[tuple[float, str]]] = {lane: [] for lane in range(HIGHWAY_LANES)}
    for index, (lane, x) in enumerate(zip(cav_lanes, cav_x, strict=True)):
        occupied[lane].append((x, f"cav_{index}"))
    for role in roles:
        occupied[role.lane].append((role.x_m, role.name))
    for lane, entries in occupied.items():
        ordered = sorted(entries)
        for index in range(len(ordered) - 1):
            x_a, name_a = ordered[index]
            x_b, name_b = ordered[index + 1]
            if x_b - x_a < MIN_SAFE_CLEARANCE_M:
                raise ValueError(
                    f"initial vehicles {name_a} and {name_b} in lane {lane} are too close."
                )


def parse_profile(raw: Any) -> ScenarioProfile:
    """Validate one raw scenario mapping and return an immutable profile."""
    if not isinstance(raw, Mapping):
        raise ValueError("scenario profile must be a mapping.")
    scenario_id = raw.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string.")
    lanes_count = raw.get("lanes_count", HIGHWAY_LANES)
    if isinstance(lanes_count, bool) or lanes_count != HIGHWAY_LANES:
        raise ValueError(f"lanes_count must be {HIGHWAY_LANES}.")
    background = raw.get("background_vehicles")
    if isinstance(background, bool) or not isinstance(background, int):
        raise ValueError("background_vehicles must be an integer.")
    if background < MIN_BACKGROUND_VEHICLES:
        raise ValueError(
            f"background_vehicles must be at least {MIN_BACKGROUND_VEHICLES}."
        )
    spacing = _require_number(raw.get("spacing_m"), "spacing_m", minimum=0.0)
    jitter = _require_number(raw.get("spacing_jitter_m", 0.0), "spacing_jitter_m")
    if spacing - 2.0 * jitter < MIN_SAFE_CLEARANCE_M:
        raise ValueError("spacing_m must leave safe gaps after jitter.")

    cav = raw.get("cav")
    if not isinstance(cav, Mapping):
        raise ValueError("cav must be a mapping.")
    cav_lanes = tuple(cav.get("lanes", ()))
    raw_start = cav.get("start_x_m")
    if isinstance(raw_start, (list, tuple)):
        if len(raw_start) != AGENT_COUNT:
            raise ValueError("cav.start_x_m must have four values.")
        cav_x = tuple(_require_number(x, "cav.start_x_m", minimum=0.0) for x in raw_start)
    else:
        base_x = _require_number(raw_start, "cav.start_x_m", minimum=0.0)
        gap = _require_number(cav.get("start_gap_m"), "cav.start_gap_m", minimum=0.0)
        if gap < MIN_CAV_SEPARATION_M:
            raise ValueError("cav.start_gap_m must leave safe CAV separation.")
        cav_x = tuple(base_x + index * gap for index in range(AGENT_COUNT))
    if len(cav_lanes) != AGENT_COUNT:
        raise ValueError("cav.lanes must have four values.")
    if any(
        isinstance(lane, bool)
        or not isinstance(lane, int)
        or not 0 <= lane < HIGHWAY_LANES
        for lane in cav_lanes
    ):
        raise ValueError("cav.lanes must contain four valid lane IDs.")
    for index in range(AGENT_COUNT):
        for other in range(index + 1, AGENT_COUNT):
            if (
                cav_lanes[index] == cav_lanes[other]
                and abs(cav_x[other] - cav_x[index]) < MIN_CAV_SEPARATION_M
            ):
                raise ValueError("CAVs sharing a lane must be staggered safely.")
    cav_speed = _require_number(
        cav.get("speed_mps"), "cav.speed_mps", minimum=0.0
    )

    forcing = _parse_forcing(raw.get("forcing"))
    if len(forcing.roles) > background:
        raise ValueError("forcing roles cannot exceed background_vehicles.")
    _validate_positions(cav_lanes, cav_x, forcing.roles)

    return ScenarioProfile(
        scenario_id=scenario_id,
        lanes_count=HIGHWAY_LANES,
        background_vehicles=background,
        vehicles_density=_require_number(
            raw.get("vehicles_density"), "vehicles_density", minimum=0.0
        ),
        duration_s=_require_number(raw.get("duration_s"), "duration_s", minimum=0.0),
        spacing_m=spacing,
        spacing_jitter_m=jitter,
        speeds=_parse_speeds(raw.get("speeds_mps")),
        cav_lanes=cav_lanes,
        cav_x_m=cav_x,
        cav_speed_mps=cav_speed,
        forcing=forcing,
    )


def load_profiles(path: str | Path = SCENARIO_CONFIG_PATH) -> dict[str, ScenarioProfile]:
    """Load and validate all profiles from a YAML configuration file."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("scenarios.yaml must contain a mapping.")
    raw_profiles = raw.get("profiles") or {}
    if not isinstance(raw_profiles, Mapping):
        raise ValueError("scenarios.yaml 'profiles' must be a mapping.")
    profiles = {}
    for scenario_id, profile_raw in raw_profiles.items():
        profile = parse_profile(profile_raw)
        if scenario_id != profile.scenario_id:
            raise ValueError("profile key must match scenario_id.")
        profiles[scenario_id] = profile
    return profiles


def resolve_scenario(
    scenario: str | Mapping[str, Any] | ScenarioProfile | None,
) -> ScenarioProfile | None:
    """Resolve None, a profile ID, a raw mapping, or an already parsed profile."""
    if scenario is None:
        return None
    if isinstance(scenario, ScenarioProfile):
        return scenario
    if isinstance(scenario, str):
        profiles = load_profiles()
        if scenario not in profiles:
            raise ValueError(
                f"unknown scenario '{scenario}'; available: {sorted(profiles)}"
            )
        return profiles[scenario]
    if isinstance(scenario, Mapping):
        return parse_profile(scenario)
    raise TypeError("scenario must be a profile ID, mapping, or None.")


def _lane_index(road: Any, lane_id: int) -> tuple[str, str, int]:
    start = next(iter(road.network.graph))
    end = next(iter(road.network.graph[start]))
    return start, end, lane_id


def _offsets(count: int) -> list[int]:
    offsets = []
    ahead = behind = 1
    while len(offsets) < count:
        offsets.append(ahead)
        ahead += 1
        if len(offsets) < count:
            offsets.append(-behind)
            behind += 1
    return offsets


def _create_cavs(native_env: Any, profile: ScenarioProfile) -> None:
    road = native_env.road
    controlled_class = native_env.action_type.vehicle_class
    native_env.controlled_vehicles = []
    for lane_id, start_x in zip(profile.cav_lanes, profile.cav_x_m, strict=True):
        lane_index = _lane_index(road, int(lane_id))
        prototype = Vehicle.make_on_lane(
            road, lane_index, float(start_x), profile.cav_speed_mps
        )
        vehicle = controlled_class(
            road,
            prototype.position,
            prototype.heading,
            prototype.speed,
        )
        native_env.controlled_vehicles.append(vehicle)
        road.vehicles.append(vehicle)


def _add_background_vehicle(
    native_env: Any,
    other_class: Any,
    lane_id: int,
    x: float,
    speed: float,
    name: str | None = None,
) -> Any:
    road = native_env.road
    vehicle = other_class.make_on_lane(
        road, _lane_index(road, int(lane_id)), float(x), float(speed)
    )
    vehicle.randomize_behavior()
    if name is not None:
        vehicle.scenario_role = name
    road.vehicles.append(vehicle)
    return vehicle


def _spawn_legacy_background(native_env: Any, profile: ScenarioProfile) -> None:
    other_class = utils.class_from_path(native_env.config["other_vehicles_type"])
    rng = native_env.road.np_random
    count = profile.background_vehicles
    base, extra = divmod(count, len(profile.cav_lanes))
    for lane_position, (lane_id, start_x) in enumerate(
        zip(profile.cav_lanes, profile.cav_x_m, strict=True)
    ):
        lane_index = _lane_index(native_env.road, int(lane_id))
        per_lane = base + int(lane_position < extra)
        positions = [
            float(start_x + offset * profile.spacing_m)
            + float(rng.uniform(-profile.spacing_jitter_m, profile.spacing_jitter_m))
            for offset in _offsets(per_lane)
        ]
        speeds = profile.speeds.draw(len(positions), rng)
        for x, speed in zip(sorted(positions), sorted(speeds), strict=True):
            _add_background_vehicle(native_env, other_class, lane_id, x, speed)


def _spawn_forced_background(
    native_env: Any,
    profile: ScenarioProfile,
    occupied: dict[int, list[float]],
) -> None:
    other_class = utils.class_from_path(native_env.config["other_vehicles_type"])
    rng = native_env.road.np_random
    generic_count = profile.background_vehicles - len(profile.forcing.roles)
    base, extra = divmod(generic_count, HIGHWAY_LANES)
    for lane_id in range(HIGHWAY_LANES):
        target = base + int(lane_id < extra)
        positions = []
        offset_index = 0
        offsets = _offsets(target + len(occupied[lane_id]) + 8)
        while len(positions) < target:
            offset = offsets[offset_index]
            offset_index += 1
            x = (
                120.0
                + offset * profile.spacing_m
                + float(rng.uniform(-profile.spacing_jitter_m, profile.spacing_jitter_m))
            )
            if x < MIN_SAFE_CLEARANCE_M:
                continue
            if any(abs(x - other) < MIN_SAFE_CLEARANCE_M for other in occupied[lane_id]):
                continue
            positions.append(x)
            occupied[lane_id].append(x)
        speeds = profile.speeds.draw(len(positions), rng)
        for x, speed in zip(sorted(positions), sorted(speeds), strict=True):
            _add_background_vehicle(native_env, other_class, lane_id, x, speed)


def create_scenario_vehicles(native_env: Any, profile: ScenarioProfile) -> None:
    """Populate the native road with deterministic CAVs and background traffic."""
    _create_cavs(native_env, profile)
    occupied: dict[int, list[float]] = {lane: [] for lane in range(HIGHWAY_LANES)}
    for lane_id, x in zip(profile.cav_lanes, profile.cav_x_m, strict=True):
        occupied[lane_id].append(float(x))

    other_class = utils.class_from_path(native_env.config["other_vehicles_type"])
    native_env._scenario_roles = {}
    if profile.forcing.roles:
        for role in profile.forcing.roles:
            vehicle = _add_background_vehicle(
                native_env,
                other_class,
                role.lane,
                role.x_m,
                role.speed_mps,
                role.name,
            )
            native_env._scenario_roles[role.name] = vehicle
            occupied[role.lane].append(role.x_m)
        _spawn_forced_background(native_env, profile, occupied)
    else:
        _spawn_legacy_background(native_env, profile)
    native_env._scenario_step_count = 0


def install_scenario_spawn(native_env: Any, profile: ScenarioProfile) -> None:
    """Install profile-specific spawn and scheduled-hazard hooks on an env instance."""

    def create_vehicles(env: Any) -> None:
        create_scenario_vehicles(env, profile)

    native_env._create_vehicles = MethodType(create_vehicles, native_env)
    hazard = profile.forcing.scheduled_hazard
    if hazard is not None and not getattr(native_env, "_scenario_step_wrapped", False):
        original_step = native_env.step

        def step_with_hazard(env: Any, action: Any) -> Any:
            result = original_step(action)
            env._scenario_step_count += 1
            if env._scenario_step_count == hazard.brake_step:
                vehicle = getattr(env, "_scenario_roles", {}).get(hazard.name)
                if vehicle is not None:
                    vehicle.target_speed = hazard.brake_speed_mps
            return result

        native_env.step = MethodType(step_with_hazard, native_env)
        native_env._scenario_step_wrapped = True


__all__ = [
    "ForcingConfig",
    "RoleVehicle",
    "ScenarioProfile",
    "SpeedDistribution",
    "create_scenario_vehicles",
    "install_scenario_spawn",
    "load_profiles",
    "parse_profile",
    "resolve_scenario",
]
