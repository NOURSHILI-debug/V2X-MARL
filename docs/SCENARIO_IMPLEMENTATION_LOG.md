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
- Lowest-TTC relevance is evaluation-only and remains unapproved.
