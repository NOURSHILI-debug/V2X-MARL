# G6 Diagnosis — Stage 1 (read-only measurement)

Branch: `feat/scenario-suite`. Measurement only: no scenario, controller, gate,
environment, or test file was changed for this stage. Design seeds 100-109.
All raw JSON files are under `results/raw/scenarios/` (gitignored).

## Plain language (5 lines)

The candidate pool is too small: per CAV and step there are only about 4.2-4.8
vehicles within 50 m, and 65-76% of steps have fewer than six.
The good news is that headroom exists (70-78% of hazard steps have the lowest-TTC
candidate outside the two nearest) and timing is not the problem (brake lead
2.111 s for `slow_blocker`, 19.254-23.147 s for `scheduled_hazard`; all visible
margins exceed the 0.4 s decision-plus-latency allowance).
The information-value gate fails because every restricted condition crashes in
all 10 seeds (oracle2 = nearest2 = 1.00 collision rate, paired difference 0,
bootstrap 95% interval `[0, 0]`), while all6 crashes in 0-1 seeds.
The oracle selection currently ranks candidates in all lanes: 53-55% of the
steps where oracle differs from nearest have a top-1 candidate more than 5.25 m
from the ego lane, which is outside the approved ego/target-lane TTC definition.
Repeatability is not a failure: all 20 doubled design seeds have equal observation
and outcome hashes.

## Gate numbering mapping (earlier report -> fixed numbering)

| Earlier label | Fixed label | Stage 1 status on design seeds |
| --- | --- | --- |
| G1 Difficulty | G1 Difficulty | PASS: keep-speed collision rate 1.00 in both scenarios |
| G2 Feasibility | G2 Feasibility | PASS: competent 0.90 (`h1_forcing`) / 1.00 (`h2_lane_change`) |
| Earlier G3 Fair start | G4 Fair start | PASS: random median survival 8.0 / 7.5 steps |
| Earlier G4 (reset collisions) | folded into G2 evidence | PASS: 0/10 initialization failures in both scenarios |
| Earlier G5 Headroom | G5 Headroom | PASS: top-2 miss 0.777 / 0.697 on hazard agent-steps |
| Earlier G6 VOI probe | **G3 Information value** | **FAIL**: oracle 1.00, nearest 1.00, paired diff 0, CI `[0, 0]` |
| not previously measured | G6 Repeatability | PASS: 20/20 doubled seeds hash-equal |

This diagnosis is not a gate edit. The approved G3 thresholds remain: oracle k=2
at least 15 percentage points below nearest k=2; 95% bootstrap interval of the
paired difference excludes zero; k=0 worse than k=6.

Stage 2 correction note: all oracle2 values in this Stage 1 diagnosis were
produced by the uncorrected all-lane `_oracle_ranks`. Stage 2 adds a unit test
and changes `_oracle_ranks` to rank only approved ego-lane and target-lane TTC.
Those earlier oracle2 numbers are retained as diagnosis evidence only and must
not be read as valid Stage 2 G3 results.

## Commands and raw outputs

Design gate re-measurement (current code, because the old `h1_forcing` raw file
predates commit `99c6eb7`):

```powershell
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
python scripts/evaluate_scenario_gate.py --scenario h1_forcing --seeds (100..109) --output results/raw/scenarios/diag_h1_current_100_109.json
python scripts/evaluate_scenario_gate.py --scenario h2_lane_change --seeds (100..109) --output results/raw/scenarios/diag_h2_current_100_109.json
```

Diagnosis trace (one process, seeds split so every command stays under 5 min):

```powershell
python <temp>\diagnose_scenarios.py h1_forcing trace 100-104
python <temp>\diagnose_scenarios.py h1_forcing trace 105-109
python <temp>\diagnose_scenarios.py h2_lane_change trace 100-104
python <temp>\diagnose_scenarios.py h2_lane_change trace 105-109
python <temp>\diagnose_scenarios.py h1_forcing determinism 100-104
python <temp>\diagnose_scenarios.py h1_forcing determinism 105-109
python <temp>\diagnose_scenarios.py h2_lane_change determinism 100-104
python <temp>\diagnose_scenarios.py h2_lane_change determinism 105-109
```

Raw files: `diag_h1_current_100_109.json`, `diag_h2_current_100_109.json`,
`diagnosis_stage1_{scenario}_{trace|determinism}_{seed_start}_{seed_end}_*.json`.

## (a) Candidate pool

`pool` = all vehicles within 50 m from `env.get_candidates`; `visible` = valid
rows in the 56-D observation (maximum six). Means are over all CAV-steps of the
10 design episodes. `lt6` = share of CAV-steps with fewer than six candidates.

| Scenario | CAV | pool mean | pool min | pool lt6 | visible mean | visible min | visible lt6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| h1_forcing | cav_0 | 4.497 | 0 | 0.731 | 3.977 | 0 | 0.731 |
| h1_forcing | cav_1 | 4.828 | 0 | 0.652 | 4.205 | 0 | 0.652 |
| h1_forcing | cav_2 | 4.620 | 1 | 0.733 | 4.165 | 1 | 0.733 |
| h1_forcing | cav_3 | 4.732 | 0 | 0.663 | 3.919 | 0 | 0.663 |
| h2_lane_change | cav_0 | 4.268 | 1 | 0.756 | 4.039 | 1 | 0.756 |
| h2_lane_change | cav_1 | 4.336 | 0 | 0.729 | 4.175 | 0 | 0.729 |
| h2_lane_change | cav_2 | 4.663 | 1 | 0.728 | 4.354 | 1 | 0.728 |
| h2_lane_change | cav_3 | 4.220 | 0 | 0.722 | 3.909 | 0 | 0.722 |

Evidence: the six-slot contract is never the binding limit; the physical 50 m
pool is. At no CAV does the pool reach six on at least 80% of steps.

## (b) Headroom

`hazard` = agent-steps with TTC < 5 s in the ego or target lane; `miss share` =
share of those where the lowest-TTC candidate is outside the two nearest;
`fallback share` = share of all agent-steps with no candidate closing.

| Scenario | seed | hazard steps | miss share | fallback share |
| --- | ---: | ---: | ---: | ---: |
| h1_forcing | 100 | 71 | 0.817 | 0.410 |
| h1_forcing | 101 | 51 | 0.902 | 0.388 |
| h1_forcing | 102 | 224 | 0.790 | 0.487 |
| h1_forcing | 103 | 64 | 0.812 | 0.378 |
| h1_forcing | 104 | 139 | 0.755 | 0.546 |
| h1_forcing | 105 | 210 | 0.800 | 0.492 |
| h1_forcing | 106 | 172 | 0.820 | 0.667 |
| h1_forcing | 107 | 147 | 0.837 | 0.529 |
| h1_forcing | 108 | 72 | 0.931 | 0.566 |
| h1_forcing | 109 | 183 | 0.541 | 0.460 |
| h1_forcing | **all** | **1333** | **0.777** | **0.503** |
| h2_lane_change | 100 | 58 | 0.759 | 0.224 |
| h2_lane_change | 101 | 47 | 0.681 | 0.531 |
| h2_lane_change | 102 | 110 | 0.818 | 0.485 |
| h2_lane_change | 103 | 44 | 0.682 | 0.502 |
| h2_lane_change | 104 | 145 | 0.807 | 0.632 |
| h2_lane_change | 105 | 117 | 0.684 | 0.659 |
| h2_lane_change | 106 | 74 | 0.730 | 0.401 |
| h2_lane_change | 107 | 94 | 0.543 | 0.435 |
| h2_lane_change | 108 | 70 | 0.657 | 0.550 |
| h2_lane_change | 109 | 136 | 0.588 | 0.489 |
| h2_lane_change | **all** | **895** | **0.697** | **0.491** |

## (c) Usefulness, paired by seed

Notation per cell: `collision / min_TTC_s / mean_speed_mps`. Aggregates at the
bottom of each block use collision rate, mean of per-episode min TTC, and mean
of per-episode mean speed.

### h1_forcing

| seed | none | nearest2 | random2 | oracle2 | all6 |
| ---: | --- | --- | --- | --- | --- |
| 100 | 1 / 0.715 / 21.8 | 1 / 1.861 / 19.0 | 1 / 1.999 / 19.1 | 1 / 0.156 / 19.0 | 0 / 0.075 / 18.7 |
| 101 | 1 / 0.117 / 22.7 | 1 / 1.399 / 19.3 | 1 / 1.584 / 20.5 | 1 / 1.455 / 20.2 | 1 / 1.606 / 14.9 |
| 102 | 1 / 0.552 / 21.8 | 1 / 1.924 / 19.3 | 1 / 1.739 / 19.6 | 1 / 0.544 / 18.9 | 0 / 0.221 / 20.5 |
| 103 | 1 / 0.959 / 22.1 | 1 / 1.864 / 19.7 | 1 / 2.085 / 19.7 | 1 / 2.168 / 19.4 | 0 / 0.262 / 20.7 |
| 104 | 1 / 0.817 / 22.0 | 1 / 1.929 / 19.4 | 1 / 1.648 / 20.1 | 1 / 2.192 / 19.3 | 0 / 0.260 / 21.1 |
| 105 | 1 / 0.035 / 23.0 | 1 / 1.923 / 19.3 | 1 / 1.786 / 20.4 | 1 / 0.654 / 19.0 | 0 / 0.248 / 18.0 |
| 106 | 1 / 0.567 / 21.7 | 1 / 1.926 / 19.2 | 1 / 1.104 / 23.9 | 1 / 2.370 / 19.4 | 0 / 1.042 / 21.4 |
| 107 | 1 / 0.647 / 21.9 | 1 / 1.924 / 19.1 | 1 / 2.184 / 20.6 | 1 / 2.184 / 19.2 | 0 / 1.457 / 21.5 |
| 108 | 1 / 0.806 / 21.9 | 1 / 1.914 / 19.2 | 1 / 1.933 / 20.1 | 1 / 2.169 / 19.0 | 0 / 0.168 / 18.7 |
| 109 | 1 / 0.810 / 22.1 | 1 / 1.919 / 19.4 | 1 / 2.256 / 20.7 | 1 / 2.274 / 20.2 | 0 / 1.450 / 22.4 |
| **all** | **1.00 / 0.603 / 22.098** | **1.00 / 1.858 / 19.297** | **1.00 / 1.832 / 20.464** | **1.00 / 1.617 / 19.356** | **0.10 / 0.679 / 19.795** |

### h2_lane_change

| seed | none | nearest2 | random2 | oracle2 | all6 |
| ---: | --- | --- | --- | --- | --- |
| 100 | 1 / 0.790 / 23.3 | 1 / 1.285 / 21.2 | 1 / 1.446 / 23.2 | 1 / 1.798 / 22.9 | 0 / 1.519 / 19.6 |
| 101 | 1 / 0.143 / 22.7 | 1 / 1.301 / 21.7 | 1 / 1.764 / 23.6 | 1 / 1.802 / 22.6 | 0 / 1.955 / 20.6 |
| 102 | 1 / 0.131 / 22.8 | 1 / 1.285 / 21.9 | 1 / 2.096 / 22.3 | 1 / 1.757 / 22.8 | 0 / 1.463 / 22.8 |
| 103 | 1 / 0.141 / 22.7 | 1 / 1.297 / 21.6 | 1 / 1.568 / 23.2 | 1 / 1.806 / 22.6 | 0 / 1.159 / 18.8 |
| 104 | 1 / 0.122 / 22.9 | 1 / 1.280 / 21.9 | 1 / 1.690 / 22.7 | 1 / 1.814 / 22.6 | 0 / 1.929 / 21.1 |
| 105 | 1 / 0.131 / 22.9 | 1 / 1.070 / 22.1 | 1 / 1.544 / 23.6 | 1 / 1.765 / 22.9 | 0 / 1.784 / 22.9 |
| 106 | 1 / 0.217 / 22.8 | 1 / 1.278 / 21.9 | 1 / 1.442 / 23.1 | 1 / 1.816 / 22.6 | 0 / 1.760 / 21.1 |
| 107 | 1 / 0.119 / 22.9 | 1 / 1.278 / 21.9 | 1 / 1.872 / 22.5 | 1 / 1.810 / 22.6 | 0 / 0.678 / 21.8 |
| 108 | 1 / 0.143 / 22.7 | 1 / 1.299 / 21.7 | 1 / 1.577 / 23.0 | 1 / 1.806 / 22.6 | 0 / 1.333 / 21.5 |
| 109 | 1 / 0.145 / 22.8 | 1 / 1.302 / 21.7 | 1 / 1.305 / 23.8 | 1 / 1.807 / 22.6 | 0 / 1.803 / 24.6 |
| **all** | **1.00 / 0.208 / 22.831** | **1.00 / 1.268 / 21.758** | **1.00 / 1.630 / 23.098** | **1.00 / 1.798 / 22.681** | **0.00 / 1.538 / 21.497** |

G3 paired test on collision outcomes: oracle - nearest = 0.000 in both
scenarios; 10,000-resample bootstrap with `numpy.default_rng(0)` gives 95%
interval `[0.000, 0.000]` in both. k=0 (1.00) is worse than k=6 (0.10 / 0.00),
so G3 fails only on the oracle-vs-nearest requirements.

## (d) Reaction and fields read

The competent controller reads exactly these observation entries: `observation[0]`
(ego speed); for each candidate row 0-5, field 5 (valid), field 0 (`rel_x`),
field 1 (`rel_y`), field 2 (`rel_vx`); and the action mask entries for
accelerate and lane-left/lane-right. Derived quantities are front gap, front
closing speed, rear gap, rear closing speed, TTC capped at 10 s, and adjacent-lane
clearance. It does **not** read field 3 (`rel_vy`) or field 4 (`rel_heading`).
It already scans all valid rows, not only the nearest two.

Measured reaction counts over all design agent-steps:

| Scenario | oracle set != nearest2 set | oracle action != full action | oracle action != nearest action | oracle top-1 outside adjacent bands |
| --- | ---: | ---: | ---: | ---: |
| h1_forcing | 3081 | 912 | 1048 | 1630 / 3081 (0.529) |
| h2_lane_change | 3552 | 883 | 1187 | 1953 / 3552 (0.550) |

`outside adjacent bands` means `|rel_y| > 5.25 m`, i.e. neither the ego lane nor
either adjacent lane. Because the evaluator's `_oracle_ranks` does not apply the
approved ego/target-lane TTC restriction, a majority of oracle-selected top-1
rows are in lanes the controller correctly treats as irrelevant. Other top-1
classes were `adjacent_front` 830/825, `adjacent_rear` 443/573,
`ego_rear_approacher` 59/54, and `ego_front_closing` 119/147 (h1/h2).

Examples of action changes (full -> oracle / nearest):

| Scenario | seed | step | CAV | actions | selected rows (index: x/y/vx, class) |
| --- | ---: | ---: | --- | --- | --- |
| h1_forcing | 100 | 2 | cav_3 | 2 -> 4 / 1 | oracle 2: -14.10/1.22/7.35 ego_rear_approacher; 4: 16.33/-13.52/-2.82 other_lane; nearest 0,1 |
| h1_forcing | 100 | 9 | cav_1 | 0 -> 1 / 0 | oracle 0: -3.33/5.08/4.08 adjacent_rear; 2: -20.05/-7.39/8.92 other_lane; nearest 0,1 |
| h2_lane_change | 100 | 6 | cav_3 | 2 -> 4 / 1 | oracle 0: -8.30/-0.36/7.49 ego_rear_approacher; 4: 28.35/-13.46/-5.95 other_lane; nearest 0,1 |
| h2_lane_change | 100 | 10 | cav_0 | 4 -> 5 / 0 | oracle 0: -2.78/8.00/-6.03 other_lane; 5: 39.48/0.00/-12.34 ego_front_closing; nearest 0,1 |

Relevant situations the controller does not score as TTC: adjacent-lane front
vehicles are reduced to a clearance test (12 m threshold), rear approachers in
adjacent lanes only zero that lane's clearance when rear TTC < 3 s, and
vehicles beyond the adjacent bands are ignored. These are generic omissions,
not scenario-specific branches.

## (e) Determinism

Two runs per design seed, same seed and config, competent controller, SHA-256 of
all four per-step observation vectors and of the outcome record (steps,
collision, return, final vehicle states). `ok` means both hashes equal between
runs; the hashes shown are from run A.

| Scenario | seed | obs hash (first 12) | outcome hash (first 12) | ok |
| --- | ---: | --- | --- | --- |
| h1_forcing | 100 | 6a83d715ed01 | b018b6b45345 | yes |
| h1_forcing | 101 | 622bb9ea4e37 | 0b71808d156b | yes |
| h1_forcing | 102 | a1d4fbb155c4 | 08f4357d8429 | yes |
| h1_forcing | 103 | 8cfa9081a521 | b5a299112c36 | yes |
| h1_forcing | 104 | 70642824bdb1 | 80f84682f784 | yes |
| h1_forcing | 105 | ef3e7133f976 | 8bc2b4b9ebc5 | yes |
| h1_forcing | 106 | b7a3223de4c2 | d330606511d9 | yes |
| h1_forcing | 107 | 43647a253a6c | da0e833e3f8b | yes |
| h1_forcing | 108 | 5c78a446e07c | 68b7ef151aeb | yes |
| h1_forcing | 109 | 567cb6309f02 | 0c93ac8afe15 | yes |
| h2_lane_change | 100 | 4c18541b278b | 8df09ead6370 | yes |
| h2_lane_change | 101 | 4938fa73d458 | df8f46da343e | yes |
| h2_lane_change | 102 | 5fbc48525bc6 | 8ba7842f0db7 | yes |
| h2_lane_change | 103 | 543aaf5a46f8 | 0fe632b4c9df | yes |
| h2_lane_change | 104 | 85ad5c29f260 | aefab52edbea | yes |
| h2_lane_change | 105 | 3bf11312ecfb | bd305d905b58 | yes |
| h2_lane_change | 106 | 03dea6f4269e | 5e6356ba82ee | yes |
| h2_lane_change | 107 | b04623f2378b | 065dc270386d | yes |
| h2_lane_change | 108 | 023ea5861add | 2853357602e2 | yes |
| h2_lane_change | 109 | 3dd97b33e71b | 57b9b2019536 | yes |

Full 64-character hashes are in the four `diagnosis_stage1_*_determinism_*.json`
raw files. Result: 20/20 observation hashes equal and 20/20 outcome hashes equal.
No nondeterminism failure mode.

## (f) Timing / avoidability margin

Method: at the first step where a named forcing vehicle appears in any CAV's
50 m candidate pool, record front TTC = `(gap - 5 m) / positive_closing_speed`
(capped at 10 s in the gate, uncapped here for the reported value) and
`brake_lead = TTC - ego_speed / 29.5`. The 29.5 m/s² value is the measured
one-step emergency-brake deceleration for a CAV at about 20 m/s (probe raw:
`[29.484, 27.886, 17.514, 8.587] m/s²` across four CAVs at seed 100; the
20.062 m/s pre-brake speed was measured in a separate probe). `margin` is the
observed time from first appearance to the last step where that front TTC still
exceeded the measured braking time; latency allowance is 0.2 s decision + 0.2 s
one-step latency = 0.4 s. `vis` = whether the hazard was in the six observation
rows when first seen. `m` = margin in seconds.

### h1_forcing

| hazard | per-seed values: seed:step/vis/TTC/brake_lead/margin |
| --- | --- |
| slow_blocker | 100:0/N/2.857/2.111/28.6 101:0/N/2.857/2.111/4.8 102:0/N/2.857/2.111/59.8 103:0/N/2.857/2.111/4.6 104:0/N/2.857/2.111/28.8 105:0/N/2.857/2.111/20.2 106:0/N/2.857/2.111/28.0 107:0/N/2.857/2.111/4.6 108:0/N/2.857/2.111/18.2 109:0/N/2.857/2.111/4.6 |
| fast_approacher (rear) | 100:0/Y/N/A/N/A/14.6 101:0/Y/N/A/N/A/N/A 102:0/Y/N/A/N/A/16.8 103:0/Y/N/A/N/A/11.2 104:0/Y/N/A/N/A/15.8 105:0/Y/N/A/N/A/35.4 106:0/Y/N/A/N/A/16.2 107:0/Y/N/A/N/A/10.8 108:0/Y/N/A/N/A/15.4 109:0/Y/N/A/N/A/11.0 |
| scheduled_hazard | 100:0/N/20.000/19.254/57.6 101:0/N/20.000/19.254/1.0 102:0/N/20.000/19.254/31.2 103:0/N/20.000/19.254/31.4 104:0/N/20.000/19.254/42.0 105:0/N/20.000/19.254/57.8 106:0/N/20.000/19.254/49.2 107:0/N/20.000/19.254/30.4 108:0/N/20.000/19.254/55.8 109:0/N/20.000/19.254/40.0 |

### h2_lane_change

| hazard | per-seed values: seed:step/vis/TTC/brake_lead/margin |
| --- | --- |
| slow_blocker | 100:0/N/2.857/2.111/4.0 101:0/N/2.857/2.111/4.2 102:0/N/2.857/2.111/4.0 103:0/N/2.857/2.111/4.2 104:0/N/2.857/2.111/4.0 105:0/N/2.857/2.111/4.0 106:0/N/2.857/2.111/4.0 107:0/N/2.857/2.111/4.0 108:0/N/2.857/2.111/4.2 109:0/N/2.857/2.111/4.2 |
| fast_approacher (rear) | 100:0/N/N/A/N/A/11.0 101:0/N/N/A/N/A/15.0 102:0/N/N/A/N/A/12.2 103:0/N/N/A/N/A/13.8 104:0/N/N/A/N/A/14.2 105:0/N/N/A/N/A/14.8 106:0/N/N/A/N/A/14.0 107:0/N/N/A/N/A/14.2 108:0/N/N/A/N/A/11.6 109:0/N/N/A/N/A/14.2 |
| scheduled_hazard | 100:1/N/28.042/27.316/28.6 101:1/N/28.589/27.863/33.8 102:1/N/27.829/27.103/31.4 103:1/N/17.093/16.367/53.0 104:1/N/17.812/17.086/59.6 105:1/N/28.920/28.194/30.6 106:1/N/17.093/16.367/24.6 107:1/N/25.481/24.755/30.2 108:1/N/27.891/27.165/28.4 109:1/N/17.093/16.367/21.2 |

Aggregate front-hazard brake lead: `slow_blocker` 2.111 s (min 2.111) in both
scenarios; `scheduled_hazard` 19.254 s (min 19.254) for h1 and 23.147 s
(min 16.367) for h2. All 20 front-hazard seed margins exceed 0.4 s
(share 1.00). For the rear `fast_approacher`, maximum braking by the ego is not
an avoiding action, so brake lead is N/A; its observed visible windows are
10.8-35.4 s (h1) and 11.0-15.0 s (h2). Hazards are visible in the candidate
pool but often not in the six observation rows (`slow_blocker` and
`scheduled_hazard` first-seen `vis = N` in every seed). Information arrives
early enough; the problem is which candidates are supplied and selected.

## Failure modes and smallest fixes

| Failure mode | Verdict | Evidence | Smallest supported fix |
| --- | --- | --- | --- |
| Pool too small | **YES** | Pool means 4.22-4.83; fewer than six on 65.2-75.6% of CAV-steps | Increase background density/reduce spacing so at least six eligible candidates exist on >=80% of steps; keep 50 m range and 12-background minimum |
| Relevant car already among nearest 2 | NO | Headroom 0.777 / 0.697 on hazard steps | Preserve the hazard/closer-vehicle layout; no fix needed |
| Information not useful (k=0 as good as k=6) | PARTIAL: full info helps, oracle subset does not | k=0 1.00 vs k=6 0.10/0.00; oracle 1.00 = nearest 1.00; paired diff 0, CI `[0,0]` | Align oracle ranking with the approved ego/target-lane TTC definition without changing thresholds; keep density so the relevant car is actually in the pool |
| Controller ignores it | PARTIAL | Controller reads all six valid rows and changes action on 912/883 agent-steps when oracle rows are supplied, but ignores `rel_vy`, `rel_heading`, adjacent-front TTC, and non-adjacent lanes; oracle top-1 is out-of-band in 52.9%/55.0% of differing steps | One generic controller extension using all valid candidates (ego lead, adjacent-lane safety, rear approachers) with documented thresholds; no scenario-specific logic |
| Nondeterminism | NO | 20/20 doubled seeds have equal observation and outcome hashes | None |
| Information too late | NO | Brake lead >=2.111 s for front hazards; all margins >0.4 s | None; preserve timing and background profile |
