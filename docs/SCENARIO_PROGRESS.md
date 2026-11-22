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
- Commit: `37b92a3`.
- Added: staggered CAV starts, seeded blocker/approacher/alongside roles,
  scheduled hazard hook, shared scripted policies, G1-G6 evaluator.
- Test command: `python -m pytest tests/test_scenario_suite.py tests/test_highway_scenarios.py -q`
  → 34 passed in 25.11s.
- Next: verify old same-x random crashes, add h1_forcing, run full gate.

## Stage B — revised H1
- Branch: `feat/scenario-suite`.
- Commit: `99c6eb7`.
- Verified old same-x starts: seed 100 CAVs all at x=100; random lane changes
  collide at step 3 (`cav_1`, `cav_2`, `cav_3`).
- Pre-measurement fixes: role speeds no longer randomized, constant-speed
  fast approacher, protected target-lane gaps, rear-hazard detection and
  corrected lane-change direction in shared competent controller.
- Test command: `python -m pytest tests/test_scenario_suite.py tests/test_highway_scenarios.py -q`
  → 34 passed in 24.02s.
- Gate command: `python scripts/evaluate_scenario_gate.py --scenario h1_forcing`
  → G1 PASS 1.00; G2 PASS 0.90; G3 PASS median 8; G4 PASS; G5 PASS headroom 0.777;
  G6 FAIL (oracle 1.00, nearest 1.00, none 1.00, all6 0.10).
- Raw: `results/raw/scenarios/h1_forcing_20261120T220706Z.json` (gitignored).
- Next: add H2 lane-change scenario and run full gate; do not tune gates.

## Stage C — H2 lane-change
- Branch: `feat/scenario-suite`.
- Commit: `b13019f`.
- Single pre-measurement layout fix: move fast approacher close behind lane-2
  CAV, add protected lane-3 escape corridor, move slow blocker ahead of
  stacked lane-1 CAVs.
- Test command: `python -m pytest tests/test_scenario_suite.py tests/test_highway_scenarios.py -q`
  → 34 passed in 24.11s.
- Gate command: `python scripts/evaluate_scenario_gate.py --scenario h2_lane_change`
  → G1 PASS 1.00; G2 PASS 1.00; G3 PASS median 7.5; G4 PASS; G5 PASS headroom 0.697;
  G6 FAIL (oracle 1.00, nearest 1.00, none 1.00, all6 0.00).
- Raw: `results/raw/scenarios/h2_lane_change_20261120T221349Z.json` (gitignored).
- Next: Stage D full regression tests, then docs; do not tune gates.

## Stage D — regression
- Branch: `feat/scenario-suite`.
- Commit: `b2ac9d3`.
- `pytest tests/test_scenario_suite.py tests/test_highway_scenarios.py -q`
  → 34 passed in 23.68s.
- `pytest tests/test_contract.py -v -k highway`
  → 5 passed, 4 skipped, 18 deselected in 3.89s.
- `CONTRACT_STRICT=1 pytest tests/test_contract.py -v -k highway`
  → 9 passed, 18 deselected in 7.19s.
- `pytest tests/test_reward.py tests/test_highway_adapter.py -q`
  → 36 passed in 34.94s.
- `pytest tests -k "not carla and not toy" -q`
  → 207 passed, 4 skipped, 33 deselected in 91.92s.
- Next: write gate report, design/implementation/test doc updates, CARLA spec.

## Stage E — documentation
- Branch: `feat/scenario-suite`.
- Commit: `6446fe1`.
- Added `docs/SCENARIO_GATE_REPORT.md` and `docs/SCENARIO_SPEC_FOR_CARLA.md`.
- Updated `docs/SCENARIO_DESIGN.md`, `docs/SCENARIO_IMPLEMENTATION_LOG.md`,
  `docs/SCENARIO_TEST_REPORT.md`.
- Final decision: H1 and H2 pass G1-G5 and fail G6; neither is a fully validated
  V2O selection benchmark. I2 and M1 remain unstarted.
- Final command status: full non-CARLA/non-toy suite 207 passed, 4 skipped,
  33 deselected in 91.92s.
- Batch complete. Do not start I2 or M1 without a new instruction.

# Phase 1 close-out

## Stage 1 — G6 diagnosis (read-only)
- Branch: `feat/scenario-suite`.
- Commit: `90f6d50` (diagnosis and initial progress entry).
- Added `docs/g6_diagnosis.md` with measured values only.
- Design-seed gate re-measurement: h1_forcing G1 PASS 1.00, G2 PASS 0.90,
  G4 PASS median 8, G5 PASS 0.777, G3 FAIL oracle 1.00 = nearest 1.00,
  paired diff 0 and bootstrap CI `[0,0]`; h2_lane_change G1 PASS 1.00,
  G2 PASS 1.00, G4 PASS median 7.5, G5 PASS 0.697, G3 FAIL with the same
  oracle/nearest collision rates.
- Diagnosis trace: pool means 4.22-4.83 with fewer than six candidates on
  65.2-75.6% of CAV-steps; headroom 0.777/0.697; determinism 20/20 equal
  observation and outcome hashes; front-hazard brake lead >=2.111 s.
- Test result: no code changed in this stage; gate and diagnosis commands all
  returned `STATUS: COMPLETE`. No pytest run in this read-only stage.
- Next: STOP for user approval. After approval, fix pool density and the
  generic controller/oracle-definition issues in at most two iterations per
  scenario, then re-measure on design seeds.

## Stage 2 — fixes started, incomplete (design seeds only)

### Plain language (5 lines)

The oracle now uses the approved ego-lane/target-lane TTC definition, and a new
unit test first failed (`1 != 2`) then passed after the fix.
The competent controller is extended generically: it reads all valid rows, uses
rear and adjacent-lane TTC for lane-change safety, and brakes when information
coverage is too low to prove a lane change safe.
`h1_dense` is unchanged and remains the naturalistic reference; only constructed
`h1_forcing` and `h2_lane_change` got higher background density, ending iteration
2 at 48 vehicles / 10 m spacing and 54 vehicles / 9 m spacing respectively.
Pool measurement reached the target on iteration 2: h1 83.9% of CAV-steps and
h2 87.8% had at least six candidates over design seeds.
Full G3/G5 re-measurement is incomplete: the high-density evaluator became too
slow for the required one-process, under-five-minute command budget, so Stage 3
confirmation was not started.

### Changes made before iteration measurements

- Added `tests/test_scenario_suite.py::test_oracle_selects_target_lane_hazard_outside_two_nearest`;
  before the code fix it failed with `assert 1 == 2`, after the fix it passed.
- `scripts/evaluate_scenario_gate.py::_oracle_ranks` now passes the ego and
  target lanes to `_candidate_ttc` for every candidate and applies to all
  scenarios.
- Gate summary now computes the G3 paired oracle-minus-nearest drop with a 2000-sample
  95% bootstrap CI and checks the k=0 vs k=6 condition.
- Added `--keys` and partial raw-output support so gate episodes can be split into
  short commands; aggregation is done from the gitignored partial JSON files.
- `scripts/scripted_policies.py` extension is generic and observation-only:
  documented thresholds `TTC_WARNING_S=3`, `TTC_BRAKE_S=1.5`,
  `SAFE_LANE_GAP_M=12`, `CLEAR_AHEAD_M=35`, `MAX_TTC_S=10`, and
  `MIN_LANE_CHANGE_VALID_ROWS=3`; adjacent-lane hazards zero that lane's clearance,
  and ego/rear hazards trigger lane change or braking.
- Added `tests/test_scenario_suite.py::test_competent_avoids_target_lane_hazard_in_candidate_set`.
- `pytest tests/test_scenario_suite.py tests/test_highway_scenarios.py -q`
  → 36 passed in 102.53s.

### Iteration 1 (design seeds)

- Config: both constructed profiles used 28 vehicles / 14 m spacing.
- Pool: h1 mean 5.70, 83.9% at least six; h2 iteration-1 final config was
  32 vehicles / 13 m with mean 5.49 and 77.9% at least six.
- `h1_forcing` full gate from 16 partial raw files:
  G1 PASS 1.00; G2 FAIL 0.00; fixed-numbering G4 PASS median 7.5;
  fixed-numbering G5 PASS 0.814 (fallback 0.244); fixed-numbering G3 FAIL with
  oracle 1.00, nearest 1.00, none 1.00, all6 1.00, oracle drop 0.00, CI `[0,0]`.
  Raw: `results/raw/scenarios/iter1_h1_forcing_gate.json` (gitignored).
- The adjacent-only hazard trigger was removed after this result because dense
  traffic made the controller brake repeatedly and get rear-ended; the change is
  generic and applies to every condition and scenario.
- `h2_lane_change` iteration-1 full G3/G5 was not run; only pool was measured.

### Iteration 2 (design seeds, incomplete)

- Config: `h1_forcing` 48 vehicles / 10 m; `h2_lane_change` 54 vehicles / 9 m.
- Pool: h1 mean 5.70 and 83.9% at least six; h2 mean 5.85 and 87.8% at least six
  (five seeds per half, 300 steps per seed, split into under-five-minute commands).
- Single-seed sanity: h1 seed 100 competent/all6 lasted 300 steps with no collision,
  raw `results/raw/scenarios/prescreen_h1_48_10_competent_all6_100.json` (gitignored).
- Full G3/G5 and full validation gates were not completed: repeated evaluator
  commands with high density exceeded the one-process, under-five-minute budget.
- No Stage 3 confirmation run was started; both scenarios remain unvalidated.

### Next

- STOP: two iterations were used per constructed scenario and G3 is not shown to pass.
- Missing evidence: full design-seed G3/G5 for both scenarios on iteration 2, and
  confirmation seeds 110-129 for any frozen scenario.
- Options for a new instruction: reduce density and re-run the full gate, use a
  faster evaluator implementation, or switch to a scenario family where hazards
  do not require dense background traffic. Do not implement without approval.
