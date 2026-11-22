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
