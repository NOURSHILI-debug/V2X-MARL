# Highway-Env Contract Adapter

`envs/highway_env.py` exposes the installed `highway-v0` simulation through the
same agent dictionary API as the CARLA environment. The default observation is
the native 25-value kinematic vector. Contract-v1 consumers can opt into the
56-value observation built by `envs/obs_builder.py`, which is the same builder
used by the CARLA adapter.

```python
from envs.highway_env import HighwayV2XEnv

env = HighwayV2XEnv(render_mode=None, obs_layout="kinematics25")  # default
env = HighwayV2XEnv(render_mode=None, obs_layout="contract56")

obs, info = env.reset(seed=0)
obs, rewards, terminated, truncated, info = env.step(
    {agent_id: 0 for agent_id in env.agents}
)
```

No arguments preserves the existing behavior: `obs_dim == 25` and
`n_actions == 6`. `obs_layout="contract56"` sets `obs_dim == 56`.

## Contract layout

Offsets are stable and defined by `envs/obs_builder.py`.

| Block | Indices | Width | Contents |
| --- | --- | --- | --- |
| Ego | 0-7 | 8 | speed, longitudinal acceleration, heading error, lateral offset, lane index, lane count, distance to goal, progress fraction |
| Route | 8-13 | 6 | next waypoint x/y in ego frame, speed limit, at-intersection flag, traffic-light state, distance to next waypoint |
| Candidates | 14-49 | 36 | six rows of `[rel_x, rel_y, rel_vx, rel_vy, rel_heading, valid]` |
| Action mask | 50-55 | 6 | keep, accelerate, decelerate, emergency brake, lane left, lane right |

Scaling constants are shared with CARLA:

| Constant | Value |
| --- | --- |
| `V_MAX` | 30 m/s |
| `A_MAX` | 8 m/s² |
| `D_MAX` | 150 m |
| `LANE_WIDTH_MAX` | 6 m |
| `MAX_LANES` | 6 |
| `ROUTE_FRAME_MAX` | 20 m |
| `CANDIDATE_RANGE` | 50 m |

Candidate selection:

- Every non-ego vehicle within 50 m is eligible.
- Candidates are sorted by `(distance, actor_id)` and truncated to six.
- `actor_id` is a stable per-episode integer assigned from the order of
  `road.vehicles` at reset.
- Unused rows are all zeros, including `valid == 0`.
- Candidate positions and velocities are rotated into the ego frame, headings
  are wrapped to `[-pi, pi]`, and all values are clipped to `[-1, 1]`.

`get_candidates(agent_id)` returns the same pre-vectorization candidate records
used to build the observation. The 56-value observation path never uses the
native highway observation; it calls `obs_builder.build_observation` directly.

## Highway stand-ins

The highway scenario has no traffic lights, intersections, or a fixed route
length, so the route block uses these values:

- `next_waypoint_x/y`: point 10 m ahead on the current lane.
- `distance_to_next`: the same 10 m lookahead.
- `speed_limit`: the current lane's speed limit.
- `at_intersection`: always `False`.
- `traffic_light_state`: always `"none"` (encoded as `0.0`).

The ego's `distance_to_goal` and `progress_fraction` are only meaningful when
`goal_distance_m` is set. Otherwise they are `D_MAX` and `0.0` respectively.

## Goal mode

`goal_distance_m=None` (default) disables goal logic. When set to a positive
distance, an agent reaches the goal once its longitudinal travel from the reset
position meets or exceeds that distance. Each agent is then frozen at zero
target speed, and the goal reward is emitted once.

`info["goal_reached"]` and `info["success"]` become true only when every agent
has reached its goal; `success` additionally requires no collision. All agents
receive `terminated == True` at that point. Without a goal, episodes end through
the native highway termination or truncation only.

## Contract mode environment defaults

In `contract56` mode the adapter requests at least 12 vehicles and a density of
`3.0` so each ego normally sees the full six candidate slots. The default
`kinematics25` layout keeps the original configuration unchanged. Pass
`background_vehicles` to raise the vehicle count in either layout.

## Training and dimensions

The adapter exposes `obs_dim`, `n_actions`, and `state()`. `state()` returns
five values per controlled vehicle (position, velocity, heading), so its length
is `5 * len(agents) == 20` for the four CAVs. Network input dimensions must be
derived from `env.obs_dim`; they must not be hard-coded to 25.

A short smoke run of the existing VDN training script works in either layout:

```powershell
python scripts/train_toy.py --env highway --episodes 3 --obs-layout kinematics25
python scripts/train_toy.py --env highway --episodes 3 --obs-layout contract56
```

The trainer reads `obs_dim` from the environment and sizes its replay buffer
and networks accordingly.

## Verification

```powershell
python -m pytest tests/test_highway_adapter.py -v
python -m pytest tests/test_contract.py -v -k highway
$env:CONTRACT_STRICT=1; python -m pytest tests/test_contract.py -v -k highway
```
