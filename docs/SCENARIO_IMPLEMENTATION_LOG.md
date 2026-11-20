# Scenario Implementation Log

## Files

Created:

- `envs/scenarios/highway_scenarios.py`: profile validation, YAML loading,
  deterministic spawn, native `_create_vehicles` hook.
- `tests/test_highway_scenarios.py`, `tests/data/highway_no_scenario_baseline.json`.
- `scripts/evaluate_h1_scenarios.py`, `scenario_results/h1_h1_*.json`.
- `docs/SCENARIO_DESIGN.md`, `docs/SCENARIO_PROGRESS.md`, test/evaluation reports.

Modified (minimal):

- `envs/highway_env.py`: optional `scenario` argument, native config override,
  per-instance spawn hook, scenario ID in `info`; no reward/obs/action changes.
- `configs/scenarios.yaml`: three validated H1 profiles.
- `envs/scenarios/__init__.py`: exports.

## Decisions and initialization

- Scenario selection requires `obs_layout="contract56"`; legacy layout stays
  unchanged when no scenario is selected.
- CAVs are created through the native action vehicle class at configured lanes
  and x positions. Background vehicles use the configured other-vehicle class.
- Background x offsets alternate ahead/behind (1, -1, 2, -2, ...); each lane's
  speeds are drawn from the profile distribution and sorted with x, so slower
  vehicles start behind faster ones.
- Spacing validation requires `spacing_m - 2*jitter >= 8 m`; measured minimum
  same-lane gaps were 14.31-23.11 m across development seeds.
- All random draws use `road.np_random`, which Highway-Env seeds in `reset()`.

## Deviations and problems fixed

- `vehicles_density` is set in native config but the scenario spawn, not the
  native density-based spawner, determines actual counts and gaps.
- The pre-change baseline initially differed between identical runs only in
  `per_agent.*.native_info.action` (unseeded `action_space.sample()` at reset);
  that field is excluded from the committed baseline.
- The first evaluation script mapped vehicle IDs incorrectly; it was fixed to
  reverse-map `env._vehicle_ids` from current road vehicles.
- Keep-speed was not forced to fail; this is reported as an unresolved gate,
  not hidden or fixed by changing reward/observation/simulation logic.
- Lowest-TTC relevance was later approved for gate evaluation; see batch 1 below.

## Batch 1 implementation

Added:

- `RoleVehicle`, `ForcingConfig`, staggered scalar CAV starts, role slots and
  scheduled braking hook in `envs/scenarios/highway_scenarios.py`.
- `configs/scenarios.yaml`: `h1_forcing`, `h2_lane_change`.
- `scripts/scripted_policies.py`: one shared competent controller using only
  the 56-D observation and action mask.
- `scripts/evaluate_scenario_gate.py`: G1-G6 runner, VoI restriction, compact
  gate table and gitignored raw JSON under `results/raw/scenarios/`.
- `tests/test_scenario_suite.py`: forcing, role, reproducibility, scheduled-hazard
  and observation-only controller tests.
- `docs/SCENARIO_GATE_REPORT.md`, `docs/SCENARIO_SPEC_FOR_CARLA.md`.

Pre-measurement corrections:

- Role speeds are no longer overwritten by `randomize_behavior()`.
- `fast_approacher` uses the kinematic `Vehicle` class so it does not brake for
  a lead CAV.
- Generic spawning now protects adjacent-lane gaps around CAVs.
- Competent controller rear-TTC sign, lane-change direction, target-lane rear
  risk and protected-gap checks were corrected.
- Oracle/relevance scoring now considers both lead and rear closing risk and
  near-overlap in adjacent lanes.

Verification: 34 scenario tests passed; contract strict mode passed; full
non-CARLA/non-toy suite reached 207 passed, 4 skipped, 33 deselected.
G1-G5 passed for both revised profiles; G6 failed for both and was reported
without post-measurement tuning.
