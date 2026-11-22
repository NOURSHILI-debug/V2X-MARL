# H1 Scenario Design

## Purpose

H1 is the first configuration-driven Highway-Env benchmark for the four-CAV
project. A profile counts as useful only if keep-speed fails; current scripted
measurements show that all three implemented H1 profiles still let keep-speed
survive, so none is yet a cooperation benchmark.

## Implemented profiles

All profiles use the native `highway-v0` road (four lanes, 10000 m straight
lanes), four CAV IDs `cav_0..cav_3`, 5 Hz decisions and 60 s / 300-step episodes.
The `scenario` argument requires `obs_layout="contract56"`.

| Profile | Background | Density key | Spacing | Speeds | CAV starts |
| --- | ---: | ---: | ---: | --- | --- |
| `h1_moderate` | 12 | 3.0 | 24.0 m ±0.75 | uniform 20-25 m/s | x=80 m, lanes 0-3, 22 m/s |
| `h1_dense` | 20 | 4.0 | 15.0 m ±0.5 | uniform 18-25 m/s | x=100 m, lanes 0-3, 22 m/s |
| `h1_mixed_speed` | 17 | 3.5 | 19.0 m ±0.5 | groups 15-18, 21-23, 26-28 m/s | x=90 m, lanes 0-3, 21.5 m/s |

The three profiles were chosen to vary traffic count/spacing and speed
heterogeneity independently. `vehicles_density` is applied to the native config
to prove scenario override; the scenario spawn itself controls actual vehicle
count and gap.

## Contracts

- Observation: canonical 56-D `[ego 8, route 6, candidates 6x6, mask 6]`;
  candidate range 50 m; six sorted slots, zero padded.
- Actions: 0 keep, 1 accelerate, 2 decelerate, 3 emergency brake, 4 left, 5 right.
- Reward: unchanged `RewardCalculator` and team reward.
- Spawn: CAVs in distinct lanes; background vehicles placed ahead/behind each
  CAV lane in alternating slots; speeds sorted by longitudinal position so
  faster vehicles start ahead, avoiding unavoidable initial collisions.
- Reproducibility: all draws use `road.np_random`, seeded through `reset(seed)`.

## Selection and launch

```python
from envs.highway_env import HighwayV2XEnv

env = HighwayV2XEnv(render_mode=None, obs_layout="contract56", scenario="h1_dense")
observations, info = env.reset(seed=100)
```

```powershell
python scripts/evaluate_h1_scenarios.py --profile h1_moderate
python scripts/evaluate_h1_scenarios.py --profile h1_dense
python scripts/evaluate_h1_scenarios.py --profile h1_mixed_speed
```

## Limitations and roadmap

- Keep-speed currently survives every H1 profile; H1 is not a selection
  benchmark as implemented.
- V2X channel, grouping/Top-K=2, reward changes, H2, merge, intersection and
  roundabout scenarios are not implemented here.
- No learner performance is validated.
- Planned next step (not started): evaluate the smallest blocking-lead change
  that would make keep-speed fail, after approval.

## Batch 1: forcing-layer revision

The shared forcing layer replaces same-x CAV starts with staggered starts and
adds deterministic background roles. This was necessary because seed-100
inspection showed all old `h1_dense` CAVs starting at x=100; random lane changes
collided at step 3.

Reusable roles:

- `slow_blocker`: low-speed lead in a CAV lane.
- `fast_approacher`: faster same-lane follower behind a CAV, using a kinematic
  vehicle so native car-following cannot make it brake.
- `alongside`: seeded adjacent-lane traffic that shapes lane-change gaps.
- `scheduled_hazard`: lane-preserving vehicle that brakes once at a configured
  decision step.

The generic background spawner reserves protected adjacent-lane gaps around
CAVs. Role speeds are assigned directly rather than randomized. A CAV may change
lanes only when the target lane has at least 12 m clearance, no faster rear
approacher within a 3 s warning horizon, and the action mask permits it.

### Revised profiles

| Profile | Background | CAV lanes | CAV starts | Blocker | Approacher | Hazard |
| --- | ---: | --- | --- | --- | --- | --- |
| `h1_forcing` | 20 | `[1,1,0,2]` | x=80, gap 25 m | lane 1 x=150, 8 m/s | lane 2 x=138, 28 m/s | lane 3 x=200, brake step 80 to 12 m/s |
| `h2_lane_change` | 20 | `[1,1,0,2]` | x=80, gap 20 m | lane 1 x=145, 8 m/s | lane 2 x=123, 28 m/s | lane 3 x=190, brake step 80 to 12 m/s |

Both profiles keep the 56-D observation, six-action mask, reward, termination and
4-CAV contract unchanged. Measured gate results are in
`docs/SCENARIO_GATE_REPORT.md`; G1-G5 pass and G6 fails on both profiles.
