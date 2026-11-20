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

## Stage D — tests
- Commit: `faf8ef6`
- Result: `python -m pytest tests/test_highway_scenarios.py -q` → 19 passed.
- `python -m pytest tests/test_contract.py -v -k highway` → 5 passed, 4 skipped.
- `CONTRACT_STRICT=1 python -m pytest tests/test_contract.py -v -k highway` → 9 passed.
- `python -m pytest tests/test_reward.py -q` → 20 passed.
- `python -m pytest tests/test_highway_adapter.py -q` → 16 passed.
- `python -m pytest tests -k "not carla and not toy" -q` → 192 passed, 4 skipped, 33 deselected.
- Next: run dev-seed scripted-policy measurements and relevance check.

## Stage E — scripted-policy measurements
- Commit: `665b6b4`
- Commands: `python scripts/evaluate_h1_scenarios.py --profile h1_moderate`,
  `--profile h1_dense`, `--profile h1_mixed_speed` (seeds 100-109; 3 policies).
- Result: keep_speed survived all three profiles (300 steps, 0 collision episodes);
  random and accelerate collided in all 10 episodes on all profiles.
- Relevance differ fraction (keep_speed): 0.116 / 0.314 / 0.026 for
  moderate / dense / mixed; lowest-TTC definition still needs approval.
- Raw outputs: `scenario_results/h1_h1_*_20261120T*.json`.
- Next: write design, implementation, test and evaluation reports.

## Stage F — documentation
- Commits: `4829d98` (design), `3136295` (implementation log),
  `280c96f` (test report), `34ea173` (evaluation report).
- Result: four concise factual reports written; raw episode JSON preserved.
- Final status: H1 technically implemented and contract-verified, but keep-speed
  gate fails on all three profiles. H2 not started. Await approval.

# Scenario suite batch

## Stage A — shared forcing layer
- Branch: `feat/scenario-suite` from `feat/scenario-h1-dense-highway`.
- Commit: pending.
- Added: staggered CAV starts, seeded blocker/approacher/alongside roles,
  scheduled hazard hook, shared scripted policies, G1-G6 evaluator.
- Test command: `python -m pytest tests/test_scenario_suite.py tests/test_highway_scenarios.py -q`
  → 34 passed in 25.11s.
- Next: verify old same-x random crashes, add h1_forcing, run full gate.
