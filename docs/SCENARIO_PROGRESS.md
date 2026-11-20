# Scenario / H1 Progress

## Stage A — inspect and plan
- Commit: `f1156fa` (baseline `origin/main`)
- Result: read-only inspection complete; V2X channel read via `git show` only.
- Next: add configuration-driven spawn hook.

## Stage B — scenario mechanism
- Commit: `a2102e1`
- Result: `python -m pytest tests/test_highway_scenarios.py tests/test_highway_adapter.py -q` → 20 passed.
- No-scenario baseline snapshot committed; legacy path unchanged.
- Next: add H1-A profile (`h1_moderate`) and verify safety.

## Stage C — H1 profiles
- Commits: `e6a149f` (h1_moderate), `e35b21f` (h1_dense), `6677776` (h1_mixed_speed).
- Result: all three initialize at seed 100 with 12/20/17 background vehicles,
  no collision at reset, 56-D observations, and `scenario_id` from config.
- Next: add contract tests for the profiles.
