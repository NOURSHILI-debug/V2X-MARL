# Env spec v1 (proposal to agree on)

Design principle: **keep the env generic and put the research in the grouping module.** The env exposes a candidate neighbor set, and the grouping variants (radius, random, learned) choose the k teammates from it. This keeps the comparison clean.

## 1. Simulation setup

| Item | Choice | Why |
|---|---|---|
| CARLA | 0.9.16, synchronous mode, `fixed_delta_seconds=0.05` (20 Hz) | Deterministic, reproducible |
| Decision rate | 5 Hz (1 action every 4 sim ticks) | Standard for driving RL, 4× cheaper |
| Traffic Manager | Sync mode, `set_random_device_seed(seed)` | Paired evaluation needs this |
| Agents | N = 4 CAVs (scale to 6 later), plus 0 to 6 scripted background vehicles | Enough for teammate selection to matter |
| Sensors | **Collision sensor only**, no cameras | Fits 4 GB GPU, fast |
| Rendering | `-quality-level=Low -RenderOffScreen` | Training speed |

## 2. Control layer (key choice)

The RL policy outputs **discrete meta-actions**. A low-level controller (CARLA's `GlobalRoutePlanner` + `LocalPlanner` / PID) turns them into throttle, brake and steer.

| ID | Action |
|---|---|
| 0 | Keep speed |
| 1 | Accelerate (+target speed) |
| 2 | Decelerate (−target speed) |
| 3 | Emergency brake |
| 4 | Lane change left |
| 5 | Lane change right |

Why: low-level steering from RL is slow to learn and unstable. Meta-actions make training feasible in 2 months and suit value-decomposition methods (VDN/QMIX need discrete actions). An action mask disables invalid actions (for example no lane change when there is no adjacent lane).

## 3. Observation (per agent, vector, normalized)

| Block | Content | Dim |
|---|---|---|
| Ego | speed, acceleration, heading error, lateral offset, lane id/index, distance to goal | ~8 |
| Route | next waypoint relative position (x, y), speed limit, at-intersection flag, traffic light state | ~6 |
| Candidates | M = 6 nearest vehicles within 50 m: relative (x, y, vx, vy, heading), + validity mask | 6 × 6 = 36 |
| V2X messages | Same features as candidates but **received through the V2X channel** (see 4) | in candidates |
| Action mask | 6 booleans | 6 |

Total ≈ 56 dimensions per agent. Pad with zeros and mask when fewer than M vehicles are present.

**Global state (training only, for the QMIX mixer):** concatenation of all CAV states (position, velocity, heading, goal distance). It is never used at execution.

## 4. V2X layer

Each CAV broadcasts `[x, y, vx, vy, heading, intent (current action)]`.

| Parameter | Default | Robustness test |
|---|---|---|
| Range | 50 m | 30 / 100 m |
| Delay | 0 ticks | 1 to 3 decision steps |
| Packet loss | 0% | 10 to 30% |

The candidate set comes from the messages actually received, so delay and loss affect the grouping variants naturally.

## 5. Grouping interface (the research hook)

```
candidates (M=6 with mask)  →  grouping module  →  k selected teammates (k=2 fixed)
                                 radius | random | learned (attention top-k)
```

- The group size k stays **fixed across variants**, so you test selection quality and not size.
- The env only provides candidates and features. It does not know which variant is used.

## 6. Reward (shared team reward + individual terms)

`r_i = w_p·progress + w_s·safety + w_c·comfort + w_g·goal`, with a team term mixed in.

| Term | Definition | Weight (start) |
|---|---|---|
| Progress | Δ distance along route, normalized | +1.0 |
| Safety | −10 on collision; −0.5 × (1 − TTC/3 s) when time-to-collision < 3 s; small penalty for gap < 5 m | −10 / −0.5 |
| Comfort | −|jerk|, −|lateral acceleration| | −0.05 |
| Goal | +10 on reaching the goal | +10 |
| Idle | −0.1 when speed < 1 m/s and not stopped for a red light or yielding | −0.1 |

**Team reward for CTDE:** `R_team = mean(r_i)` plus a −10 team collision penalty. The mixer learns the credit assignment, and your grouping decides who informs whom. Weights are tuned in week 2, then **frozen** before the main experiments.

## 7. Termination

| Condition | Result |
|---|---|
| Collision (any CAV) | Episode ends, flagged as failure |
| All CAVs reached goal | Episode ends, success |
| Timeout (e.g. 60 s sim = 300 decision steps) | Episode ends |
| Off-road or stuck > 10 s | Agent flagged, episode ends |

Success = all CAVs reach their goals with no collision.

## 8. Scenarios (seeded)

| Scenario | Why it needs cooperation |
|---|---|
| Unsignalized intersection | Negotiating priority |
| Highway or road merge | Gap creation |
| Lane change with dense traffic | Coordinated yielding |

**Train towns:** Town03, Town04, Town05. **Unseen test towns:** Town01, Town02, Town06 (check availability). The seed fixes spawn points, routes, goals and background traffic.

## 9. API (PettingZoo parallel style)

```python
obs, info = env.reset(seed=s)          # dict: agent_id -> obs vector
obs, rewards, terminated, truncated, info = env.step(actions)
# info: collisions, ttc_min, jerk, goal_reached, scenario_id, town, seed
env.state()        # global state (training only)
env.action_masks() # per-agent masks
```

## 10. Metrics logged each episode

Success rate, collision rate, mean travel time, mean speed, min TTC, mean jerk, V2X messages received and dropped, and the selected teammate ids (for the selection analysis).

## 11. Decisions to confirm together

1. N = 4 agents, M = 6 candidates, k = 2 teammates.
2. Meta-actions (6 actions) with a low-level controller.
3. Decision rate 5 Hz, sim 20 Hz.
4. Reward weights as above, frozen after week 2.
5. Train and test town split as above.

For the meeting, present this as **"env spec v1"**. The main selling points are the meta-action control, the grouping hook with fixed k, the simulated V2X with robustness tests, and the seeded paired evaluation.

I can write the `carla_env.py` skeleton following this spec next.

**## 12. Contract v1 (frozen; overrides §9 and §6 where they differ)**



**\*\*Precedence:\*\* this section wins over §9 and §6. Any change needs a message to the other person and a spec update first.**



**\*\*12.1 API\*\***

**- `reset(seed=None) -> (obs, info)`; `obs` is `dict\[agent\_id -> np.ndarray(obs\_dim,)]`.**

**- `step(actions) -> (obs, rewards, terminated, truncated, info)`; `rewards`, `terminated` and `truncated` are dicts keyed by agent id (float / bool / bool).**

**- Attributes on every env: `agents` (list of ids), `obs\_dim`, `n\_actions`, plus the methods `state()` and `action\_masks()`.**

**- Actions are the 6 meta-actions of §2. The toy env may expose `n\_actions=3`, and the toy and highway envs may use their own `obs\_dim`. Only CARLA is fixed at 56.**



**\*\*12.2 Shared termination (team objective)\*\***

**- A collision by any CAV ends the episode for all agents: `terminated\[a]=True` for every `a`, and `info\["collision"]=True`.**

**- All agents at their goal with no collision: `terminated` all True, `success=True`.**

**- Timeout (300 decision steps): `truncated` all True.**

**- A stuck agent for more than 10 s: episode ends and `info\["stuck"]=True`.**

**- Per-agent details go in `info\["per\_agent"]\[agent\_id]` (for example who collided). They are never used for termination.**



**\*\*12.3 Info keys\*\***

**- Every step: `collision` (bool), `goal\_reached` (bool, True only when all agents have reached their goals), `steps` (int).**

**- Last step of an episode, additionally: `success` (bool), `min\_ttc` (float, seconds), `jerk` (float, mean), `scenario\_id`, `town`, `seed`.**

**- Names are singular: `collision` and `min\_ttc`, not `collisions` and `ttc\_min`.**



**\*\*12.4 Reward\*\***

**- Per-agent weights are those of §6: progress +1.0, collision -10, TTC penalty -0.5, comfort -0.05, goal +10 (once, on arrival), idle -0.1. `RewardCalculator` defaults must match.**

**- The team reward used by VDN and QMIX is `mean(r\_i)` plus a -10 team collision penalty, reported in `info\["team\_reward"]`.**

**- A vehicle that reached its goal is frozen and earns no further progress reward.**



**\*\*12.5 Observation\*\***

**- The CARLA observation is 56-D (ego 8, route 6, candidates 6x6, mask 6).**

**- Highway-env is a development fixture. A 25-D observation is accepted for now. A highway adapter producing the 56-D layout is a Phase 1 task for B.**

**- Models trained on 25-D do not transfer to CARLA, so QMIX and VDN are retrained on CARLA with 56-D.**



**\*\*12.6 Reproducibility\*\***

**- The same seed gives the same starts, routes and goals. Different seeds give different starts.**

