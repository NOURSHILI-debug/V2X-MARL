# Scenario Gate Report — H1/H2 Batch 1

## Scope

This report covers the first Task 3 batch: revised H1 (`h1_forcing`) and H2
lane-change (`h2_lane_change`) on `feat/scenario-suite`. I2 and M1 are not
started. Raw JSON is gitignored under `results/raw/scenarios/`.

## Method

```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
python scripts/evaluate_scenario_gate.py --scenario h1_forcing
python scripts/evaluate_scenario_gate.py --scenario h2_lane_change
```

- Seeds: 100-109 (10 episodes per condition).
- Horizon: 300 steps / 60 s at 5 Hz.
- Conditions: `keep_speed:all6`, `random:all6`, `accelerate:all6`,
  `competent:all6`, and competent with `none`, `nearest2`, `random2`, `oracle2`.
- Gates were not edited. `G5_HEADROOM_THRESHOLD = 0.05`.
- G6 pass criterion remains `oracle_collision < nearest_collision and
  none_collision > all6_collision`.

## Gate table

| Gate | Threshold | `h1_forcing` | Verdict | `h2_lane_change` | Verdict |
| --- | --- | ---: | --- | ---: | --- |
| G1 | keep-speed collision rate >= 0.50 | 1.00 | PASS | 1.00 | PASS |
| G2 | competent collision-free 300-step rate >= 0.80 | 0.90 | PASS | 1.00 | PASS |
| G3 | random median survival >= 5 steps | 8.0 | PASS | 7.5 | PASS |
| G4 | no unavoidable reset collisions, shown by G2 | 0.90 | PASS | 1.00 | PASS |
| G5 | top-2 miss on hazard agent-steps > 0.05 | 0.777 | PASS | 0.697 | PASS |
| G6 | oracle < nearest and none > all6 | see below | FAIL | see below | FAIL |

## Per-seed detail

### `h1_forcing`

- G1 collisions: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`.
- G2 successes: `[1, 0, 1, 1, 1, 1, 1, 1, 1, 1]` (seed 101 failed).
- G3 random survival: `[4, 7, 8, 7, 8, 18, 9, 6, 8, 20]`.
- G4 initialization failures: all 0.
- G5: 1333 hazard agent-steps; fallback fraction 0.503; headroom 0.777.
- G6 collision rates: none 1.00, nearest2 1.00, random2 1.00, oracle2 1.00,
  all6 0.10.
- Raw: `results/raw/scenarios/h1_forcing_20261120T220706Z.json`
  (metadata `git_sha=37b92a3`).

### `h2_lane_change`

- G1 collisions: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`.
- G2 successes: all 1.
- G3 random survival: `[4, 7, 8, 6, 10, 2, 9, 6, 8, 16]`.
- G4 initialization failures: all 0.
- G5: 895 hazard agent-steps; fallback fraction 0.491; headroom 0.697.
- G6 collision rates: none 1.00, nearest2 1.00, random2 1.00, oracle2 1.00,
  all6 0.00.
- Raw: `results/raw/scenarios/h2_lane_change_20261120T221349Z.json`
  (metadata `git_sha=99c6eb7`).

## Interpretation

- Both revised profiles satisfy G1-G5. Keep-speed now fails deterministically;
  the competent controller survives almost every episode with all six candidate
  slots available; random decisions die quickly; reset states are feasible.
- G6 fails on both profiles. Every restricted-view condition crashes in every
  episode, while `all6` is collision-free or nearly collision-free. This shows
  that the scenarios depend on the full six-slot observation in gate testing, but
  it does **not** demonstrate the intended ranking `oracle < nearest` or a
  `none > all6` margin attributable to the oracle.
- No further tuning was applied after the measured G6 failures. I2 and M1 remain
  unstarted, and no learner was retrained.
