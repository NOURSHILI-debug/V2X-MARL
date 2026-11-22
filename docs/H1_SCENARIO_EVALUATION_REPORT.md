# H1 Scenario Evaluation Report

This is the first H1 evaluation report. Raw episode JSON files are preserved;
later results must be appended, not substituted.

## Setup

- Code: commit `665b6b4`; profiles from `configs/scenarios.yaml`.
- Seeds: development-only 100-109. Policies: keep_speed, random, accelerate.
  10 episodes per policy per profile = 90 episodes, max 300 steps each.
- Hardware: CPU-only i3 laptop; Windows 11, Python 3.13.5,
  `Intel64 Family 6 Model 126 Stepping 5`; single process, `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`.
- Commands (one profile per command):

```powershell
$env:PYTHONPATH="C:\Users\SINORFI\Desktop\V2X-MARL"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
python scripts/evaluate_h1_scenarios.py --profile h1_moderate
python scripts/evaluate_h1_scenarios.py --profile h1_dense
python scripts/evaluate_h1_scenarios.py --profile h1_mixed_speed
```

- Raw output:
  `scenario_results/h1_h1_moderate_20261120T211123Z.json`,
  `scenario_results/h1_h1_dense_20261120T211450Z.json`,
  `scenario_results/h1_h1_mixed_speed_20261120T211926Z.json`.

Measured initial density (mean over seeds 100-109):

| Profile | Background | Mean same-lane gap | Min same-lane gap | Speed spread |
| --- | ---: | ---: | ---: | ---: |
| h1_moderate | 12.0 | 23.941 m | 23.105 m | 4.235 m/s |
| h1_dense | 20.0 | 15.005 m | 14.310 m | 6.507 m/s |
| h1_mixed_speed | 17.0 | 18.992 m | 18.337 m | 12.044 m/s |

H1-B is measurably denser than H1-A (20 vs 12 vehicles; 15.0 vs 23.9 m gaps).
H1-C is measurably more speed-heterogeneous (12.0 vs 4.2/6.5 m/s spread).

## Per-seed results

`col` is collision occurred (1/0); `rel-diff` is fraction of CAV-steps where
nearest and lowest-TTC candidates differed. Returns are summed team rewards.

### h1_moderate

| seed | keep steps | keep col | keep return | keep rel-diff | rand steps | rand col | rand return | accel steps | accel col | accel return |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 300 | 0 | 198.13 | 0.098 | 2 | 1 | -14.90 | 11 | 1 | -8.34 |
| 101 | 300 | 0 | 198.04 | 0.118 | 1 | 1 | -14.82 | 11 | 1 | -8.63 |
| 102 | 300 | 0 | 198.03 | 0.103 | 2 | 1 | -16.82 | 11 | 1 | -8.23 |
| 103 | 300 | 0 | 198.21 | 0.085 | 2 | 1 | -14.53 | 11 | 1 | -8.33 |
| 104 | 300 | 0 | 198.29 | 0.093 | 2 | 1 | -14.71 | 12 | 1 | -10.92 |
| 105 | 300 | 0 | 196.76 | 0.133 | 4 | 1 | -18.53 | 11 | 1 | -11.24 |
| 106 | 300 | 0 | 197.60 | 0.114 | 2 | 1 | -19.55 | 11 | 1 | -8.67 |
| 107 | 300 | 0 | 196.77 | 0.182 | 2 | 1 | -14.17 | 11 | 1 | -11.53 |
| 108 | 300 | 0 | 197.62 | 0.128 | 3 | 1 | -16.65 | 11 | 1 | -8.59 |
| 109 | 300 | 0 | 197.51 | 0.111 | 3 | 1 | -15.75 | 11 | 1 | -10.56 |

### h1_dense

| seed | keep steps | keep col | keep return | keep rel-diff | rand steps | rand col | rand return | accel steps | accel col | accel return |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 300 | 0 | 183.82 | 0.235 | 2 | 1 | -15.16 | 7 | 1 | -16.91 |
| 101 | 300 | 0 | 183.91 | 0.243 | 1 | 1 | -15.18 | 6 | 1 | -11.79 |
| 102 | 300 | 0 | 183.75 | 0.237 | 2 | 1 | -17.47 | 6 | 1 | -11.95 |
| 103 | 300 | 0 | 178.10 | 0.442 | 2 | 1 | -15.09 | 6 | 1 | -15.02 |
| 104 | 300 | 0 | 184.36 | 0.219 | 2 | 1 | -14.96 | 7 | 1 | -16.94 |
| 105 | 300 | 0 | 182.53 | 0.265 | 4 | 1 | -24.36 | 6 | 1 | -11.99 |
| 106 | 300 | 0 | 174.24 | 0.522 | 2 | 1 | -20.56 | 6 | 1 | -15.05 |
| 107 | 300 | 0 | 182.77 | 0.389 | 2 | 1 | -14.86 | 6 | 1 | -14.41 |
| 108 | 300 | 0 | 183.36 | 0.343 | 3 | 1 | -17.45 | 6 | 1 | -11.88 |
| 109 | 300 | 0 | 182.81 | 0.246 | 3 | 1 | -16.12 | 6 | 1 | -11.93 |

### h1_mixed_speed

| seed | keep steps | keep col | keep return | keep rel-diff | rand steps | rand col | rand return | accel steps | accel col | accel return |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 300 | 0 | 195.76 | 0.022 | 2 | 1 | -15.17 | 9 | 1 | -10.37 |
| 101 | 300 | 0 | 195.45 | 0.033 | 1 | 1 | -14.76 | 9 | 1 | -10.65 |
| 102 | 300 | 0 | 195.69 | 0.031 | 2 | 1 | -16.81 | 9 | 1 | -10.66 |
| 103 | 300 | 0 | 196.08 | 0.022 | 2 | 1 | -14.50 | 9 | 1 | -10.55 |
| 104 | 300 | 0 | 195.86 | 0.022 | 2 | 1 | -14.99 | 9 | 1 | -10.63 |
| 105 | 300 | 0 | 195.63 | 0.021 | 4 | 1 | -18.41 | 9 | 1 | -10.58 |
| 106 | 300 | 0 | 195.19 | 0.024 | 2 | 1 | -19.66 | 8 | 1 | -10.57 |
| 107 | 300 | 0 | 195.65 | 0.033 | 2 | 1 | -14.09 | 8 | 1 | -10.54 |
| 108 | 300 | 0 | 195.49 | 0.027 | 3 | 1 | -17.12 | 9 | 1 | -10.65 |
| 109 | 300 | 0 | 195.89 | 0.028 | 3 | 1 | -15.71 | 9 | 1 | -10.60 |

## Aggregate measurements (n=10; mean, std)

| Profile / policy | Steps | Return | Min TTC | Near-collision steps | Mean CAV speed | Occupancy | Valid candidates | Rel-diff |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| moderate / keep | 300.0 ± 0.0 | 197.70 ± 0.53 | 6.40 | 0.0 | 19.92 | 5.04 | 6.10 | 0.116 ± 0.026 |
| moderate / random | 2.3 ± 0.78 | -16.04 ± 1.73 | 7.62 | 0.0 | 20.63 | 6.00 | 14.36 | 0.598 ± 0.130 |
| moderate / accel | 11.1 ± 0.30 | -9.50 ± 1.30 | 0.06 | 9.3 | 27.79 | 6.00 | 14.97 | 0.995 ± 0.009 |
| dense / keep | 300.0 ± 0.0 | 181.96 ± 3.07 | 8.92 | 0.0 | 18.86 | 5.65 | 7.77 | 0.314 ± 0.100 |
| dense / random | 2.3 ± 0.78 | -17.12 ± 2.95 | 4.85 | 0.6 | 19.75 | 6.00 | 22.92 | 0.602 ± 0.280 |
| dense / accel | 6.2 ± 0.40 | -13.79 ± 2.02 | 0.03 | 6.0 | 26.43 | 6.00 | 23.00 | 0.992 ± 0.017 |
| mixed / keep | 300.0 ± 0.0 | 195.67 ± 0.24 | 6.52 | 0.0 | 19.81 | 4.88 | 6.01 | 0.026 ± 0.004 |
| mixed / random | 2.3 ± 0.78 | -16.12 ± 1.74 | 7.80 | 0.0 | 20.13 | 6.00 | 19.00 | 0.621 ± 0.182 |
| mixed / accel | 8.8 ± 0.40 | -10.58 ± 0.08 | 0.06 | 7.8 | 27.15 | 6.00 | 17.11 | 1.000 ± 0.000 |

## Keep-speed gate

- `h1_moderate`: keep-speed survived: 10/10 episodes reached 300 steps, 0
  collision episodes, mean return 197.70. **Fails the usefulness gate.**
- `h1_dense`: keep-speed survived: 10/10 episodes reached 300 steps, 0
  collision episodes, mean return 181.96. **Fails the usefulness gate.**
- `h1_mixed_speed`: keep-speed survived: 10/10 episodes reached 300 steps, 0
  collision episodes, mean return 195.67. **Fails the usefulness gate.**

None of the implemented profiles forces cooperation. They must not be described
as cooperation or teammate-selection benchmarks. Smallest proposed change
(not implemented, needs approval): place a slow lead vehicle blocking one CAV's
lane with a fast background vehicle approaching from behind in the adjacent
lane, so keep-speed loses progress/safety while a coordinated lane change is the
clear alternative.

## Relevance check (evaluation-only)

Definition used: nearest valid candidate by Euclidean distance versus candidate
with lowest TTC, where TTC = longitudinal gap / positive closing speed for
candidates in the ego lane or target lane, capped at 10 s; nearest is the fallback
when no candidate closes. **"Most relevant = lowest TTC" needs approval.**

- Keep-speed: rel-diff 0.116 (moderate), 0.314 (dense), 0.026 (mixed). In mixed
  the two measures almost always coincide, so radius selection cannot lose in
  that profile under keep-speed. Dense has the largest measured disagreement.
- Random: 0.598 / 0.602 / 0.621. Accelerate: 0.995 / 0.992 / 1.000.
- No observation or reward field was changed for this check.

## Technical validation

Across all 90 episodes: 4 agents (`cav_0..cav_3`); 56-D observations; candidate
block `(6, 6)`; 50 m candidate range; six actions 0-5; masks bool `(6,)`;
padding correct in 90/90; masks valid in 90/90; invalid actions 0; exceptions 0;
initialization failures 0; step count never exceeded 300. Keep-speed episodes
ended by truncation at 300; random/accelerate ended by collision termination.
Reward formula equality and no-scenario bitwise baseline are covered in
`docs/SCENARIO_TEST_REPORT.md`; no reward file was edited.

Goal completion is `N/A (highway-env has no goal)`. No VDN/QMIX/MAPPO runs were
performed; no learner performance or research hypothesis is validated here.

## Interpretation and readiness

Measured facts: denser traffic and mixed speeds changed count, spacing, speed
spread, candidate occupancy and scripted-policy returns, but keep-speed still
survived all profiles. Random and accelerate policies crashed quickly, which
shows these scripted baselines are unsafe, not that cooperation is required.
There is no evidence here that denser traffic improves cooperative learning.

Readiness decision: H1 is technically ready as a contract/spawn harness, but
**not ready as the benchmark baseline for H2** because the keep-speed gate fails
on every profile. H2 work should wait for approval of a gate-forcing change.
