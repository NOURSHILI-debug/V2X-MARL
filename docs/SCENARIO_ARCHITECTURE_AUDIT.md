# V2X-MARL Scenario Architecture Audit

**Audit scope:** repository at the current `main` worktree; source code, tests,
configuration, documentation, and installed `highway-env` 1.12.1 were inspected.
No source files or experimental results were modified.

## 1. Existing architecture summary

The project has two environment implementations behind a shared PettingZoo-style
dictionary API:

- `envs/highway_env.py`: adapter around installed `highway-env` `highway-v0`,
  exposing four controlled vehicles in `cav_0` through `cav_3`.
- `envs/carla_env.py`: CARLA implementation for 4 CAVs, synchronous mode, a
  20 Hz simulation tick, 5 Hz decisions, route planning, collision monitoring,
  56-D observations, and 6 meta-actions.
- `envs/base.py`: shared contracts for agent IDs, candidate constants, action
  IDs, API shape, state, and action masks.
- `envs/obs_builder.py`: canonical 56-D observation builder.
- `envs/v2x_channel.py`: standalone, tested range/delay/loss channel. It is
  **not wired into either environment**.
- `envs/reward.py` and `envs/reward_integration.py`: per-agent and team reward
  calculation.
- `marl/vdn.py`, `marl/qmix.py`, `marl/mappo.py`: centralized-training
  implementations with decentralized masked actor/policy behavior.
- `grouping/`: all four source modules/files are present but zero-length:
  `__init__.py`, `radius.py`, `random_sel.py`, and `learned_sel.py`.
- `envs/scenarios/__init__.py`: zero-length placeholder; no scenario registry or
  scenario implementations exist.
- `configs/env.yaml`, `configs/scenarios.yaml`, and `configs/train.yaml`: all
  zero-length. Existing training scripts hard-code or accept CLI parameters
  instead of reading these files.

There is no source `tests/test_grouping.py` in the worktree. Stale grouping
`__pycache__` artifacts exist and cannot be used as evidence of an implementation.

## 2. Verified project specifications

The active code and Contract v1 agree on these fixed values:

| Item | Verified implementation |
|---|---|
| CAV count | Highway adapter always 4; CARLA constant 4, although constructor can accept another count |
| Candidate slots | 6 |
| Candidate features | relative x, relative y, relative vx, relative vy, heading, valid flag |
| Candidate range | 50 m |
| Canonical observation | 56-D, only for canonical/CARLA/contract56 paths |
| Ego block | 8-D |
| Route block | 6-D |
| Candidate block | 36-D |
| Action-mask block | 6-D |
| Actions | 0 keep, 1 accelerate, 2 decelerate, 3 emergency brake, 4 lane left, 5 lane right |
| Agent IDs | `cav_0` through `cav_3` |
| Top-K | No implementation is present; contract-level target remains 2 |
| Communication | Standalone V2X channel supports range, fixed decision-step delay, per-link loss |
| Algorithms | VDN, QMIX, MAPPO |
| Training | CTDE: decentralized masked actions; global state in QMIX/MAPPO training |
| CARLA | 0.9.16 is documented; CARLA integration is not a highway-env requirement |

The canonical 56-D layout is `[ego(8), route(6), candidates(6x6), mask(6)]`.
The 25-D native Highway-Env observation is explicitly a legacy/development
fixture and is not canonical. The Highway-Env `contract56` layout is already
implemented as a development adapter.

## 3. Existing environment and V2X interfaces

### 3.1 Environment contract

`reset(seed=None) -> (obs, info)`:

- Observations are dictionaries keyed exactly by the configured agent IDs.
- `CARLA` is 56-D. The Highway-Env adapter can expose 25-D or 56-D.
- CARLA `reset(seed=None, town=None)` additionally accepts an optional town.
- The Highway-Env adapter accepts `background_vehicles`, `duration`,
  `obs_layout`, and `goal_distance_m`.

`step(actions) -> (obs, rewards, terminated, truncated, info)`:

- Actions must contain exactly every agent ID and must be integer IDs in `0..5`.
- The Highway adapter validates lane-mask validity.
- CARLA maps lane changes to invalid/keep behavior when unavailable; this is
  an additional compatibility behavior that was not yet frozen as contract
  behavior.
- CARLA uses 4 simulation ticks per 5 Hz decision step (0.05 s x 4).
- Highway Env uses its native policy/simulation configuration.

Agent ordering is the list order `env.agents`; MARL learners stack observations
and masks in that order. CARLA is `cav_0...cav_3`, and Highway-Env is the same.

Candidates:

- Highway: every non-ego road vehicle within 50 m, sorted by `(Euclidean
  distance, stable integer actor id)`, capped at 6 by the observation builder.
- CARLA: every vehicle actor within 50 m except ego and finished CAVs, sorted
  by `(distance, CARLA actor id)`, capped at 6.
- Missing slots are zero-filled by `obs_builder`; the sixth candidate feature
  is `valid=0`.
- `get_candidates()` exposes at most the nearest 6 only through the candidate
  list; it does not implement Top-K=2 selection.

Termination:

- CARLA: shared collision termination, all-goal success, 300-step timeout,
  stuck termination after the configured grace/limit, and off-road termination.
- Highway: native collision/truncation plus adapter-specific all-goal
  termination when `goal_distance_m` is set.
- Contract v1 requires team-level termination dictionaries, episode success,
  min TTC, mean jerk, scenario/town/seed metadata on final steps.

Reward:

- Per-agent progress, collision, TTC, comfort, goal, and idle terms are in
  `envs/reward.py`.
- Team reward is mean per-agent reward plus -10 if a team collision occurred.
- The audit does not change these values or formulas.

### 3.2 V2X contract

Candidate identification is currently direct world-state access. The standalone
V2X channel does not modify candidate creation.

Selection:

- No grouping source implementation or Top-K=2 integration exists.
- The observation builder only sorts candidates by distance and actor ID.
- The missing radius/random/learned grouping modules are confirmed by
  zero-length source files.

Transmission:

- `V2XChannel.step(senders)` accepts four-to-six value sender records:
  `[x, y, vx, vy, heading, intent]`.
- It advances once per call and is designed to be called once per decision step.
- Range is checked by Euclidean distance at transmission time.
- Loss is an independent Bernoulli draw per directed sender/receiver/step.
- Delay is an integer number of decision steps; delivered content is stale
  message content captured at transmission time.
- Received messages include sender ID and age.
- `messages_to_candidates()` converts received messages to the canonical
  `(6, 6)` candidate block.

Not currently implemented:

- CAV-only versus background broadcast policy.
- Intent transfer into the canonical candidate features.
- Channel integration into CARLA or Highway-Env reset/step.
- Per-CAV independent sending or receiving call sites.
- Candidate selection by radius/random/learned policy.
- Channel range values larger than candidate 50 m without changing normalization
  semantics.

### 3.3 MARL contract

- VDN sums selected per-agent Q-values for team Q; replay stores local
  observations and a global state despite VDN not using a mixing network.
- QMIX uses per-agent observations plus agent one-hot and a monotonic
  state-conditioned mixer.
- MAPPO uses a shared masked actor of local observation plus agent identity and
  a centralized team-value critic receiving `env.state()`.
- Execution is local observation and action mask; central state is training-only.
- Highway state is environment-specific and currently 20-D for 4 CAVs.
- CARLA state is 24-D (`4 CAV x [x,y,vx,vy,yaw,goal_distance]`).
- Learner dimensions must derive from `obs_dim` and `state()`; no observed
  scenario or road type is added to the observation contract.

## 4. Eight-scenario feasibility matrix

Definitions:

- **Direct**: existing adapter/environment can already represent it.
- **Scenario initialization**: requires a seed/spawn/traffic profile on an
  existing compatible environment.
- **Adapter**: requires an adapter/generalization implementation.
- **Not yet compatible**: cannot preserve contracts with existing code only.

| ID | Scenario | Installed Highway-Env environment | 4-CAV / 6-candidate compatibility | Existing actions/masks | Feasibility |
|---|---|---|---|---|---|
| H1 | Dense highway traffic | `highway-v0` directly | Existing adapter; contract56 requests at least 12 vehicles | Lane actions and masks supported | **Direct adapter support** |
| H2 | Cooperative lane changing | `highway-v0` directly | Existing adapter; requires enough spaced vehicles | Native 4 lane/longitudinal meta-actions map to 6 | **Scenario initialization** |
| M1 | On-ramp merging | `merge-v0`, `merge-generic-v0`, `merge-v1` installed | No existing adapter with 4 controlled vehicles | Merge env may need controlled-vehicle/multi-agent wrapping | **Adapter + scenario initialization** |
| M2 | Cooperative bottleneck | Generic merge environment available | Not configured for 4 CAVs or six candidate slots | Lane masks may be synthesizable from road | **Adapter + scenario initialization** |
| I1 | Four-way intersection | `intersection-v0/v1/v2` and intersection multi-agent variants | Multi-agent intersections exist, but repository adapter does not use them | 6-action mapping and masks need adapter verification | **Adapter + scenario initialization** |
| I2 | Conflicting trajectories | Multi-agent intersection environment can create conflicts | 4-CAV claim requires actual installed variant/config verification | Actions/masks need verification | **Adapter + scenario initialization** |
| R1 | Roundabout entry | `roundabout-v0/v1/generic-v0/v1` installed | No repository roundabout adapter or verified 4-CAV setup | Existing 6 meta-actions may map to native actions; mask verification required | **Adapter + scenario initialization** |
| R2 | Circulation and exit | Same roundabout environments | No verified 4-CAV setup or six full candidate slots | Same as R1 | **Adapter + scenario initialization** |

H1 and H2 preserve 56-D through `obs_layout="contract56"` only after the adapter
is generalized as necessary. I1 has the strongest native Highway-Env candidate
support. Roundabout scenarios are **provisional** until a 4-CAV road model,
mask implementation, and goal/exit termination are exercised.

## 5. Scenario design recommendations

All positions, densities, and speeds below are initial research defaults to
verify; they are not implemented behavior.

### H1: Dense highway traffic

- Objective: safe high-throughput flow and robust Top-K relevance under dense
  candidate occupancy.
- Road: straight 4-lane highway.
- Initial distribution: 4 CAVs in separate lanes or two staggered pairs; 12+
  background vehicles.
- CAV starts: 20-30 m longitudinal spacing; mixed initial lanes.
- Six candidate slots: full where possible with 12+ vehicles.
- Density: high but collision-safe initial gaps (15-25 m in-lane).
- Speed: CAV 20-25 m/s; background 18-28 m/s mixed per seed.
- Relevance: far but incoming vehicles can be more relevant than a stationary
  nearby vehicle.
- Termination: existing native collision/time or 300 decision steps/goal mode.
- Success: all goals reached; no collision.
- Safety: no collision and TTC threshold only as reward support.
- V2X: 50 m baseline, robustness 30/100 m, delay/loss as configured.

### H2: Cooperative lane changing

- Objective: coordinated lane changes without blocking adjacent CAVs.
- Road: straight 4-lane highway with a lead obstacle in one CAV's lane.
- CAV starts: staggered; at least one CAV is not the nearest physical vehicle
  to a CAV but is the cooperative yielding partner.
- Six slots: include lead, adjacent lane vehicles, and a partner in the next
  relevant lane.
- Density: medium/high.
- Speed: CAV 18-24 m/s; mixed surrounding speeds.
- Termination: native collision/time or explicit lane-change goal distance.
- Success: all planned lane changes complete and remain on-road.
- Safety: no collision; no unsafe lane action from action masks.
- V2X: selected peers exchange intent and relative motion.

### M1: On-ramp merging

- Objective: create a usable gap or yield sequence for a ramp vehicle.
- Road: merge-v0-compatible straight highway plus sine-ramp.
- CAV starts: 2 highway CAVs ahead of the merge and 2 staggered at/behind the
  ramp point.
- Candidate slots: ramp vehicle may be farther than an in-lane vehicle but
  should be relevant; this is a key relevance case.
- Density: medium; 8-12 traffic vehicles.
- Speeds: highway 20-28 m/s, ramp 12-20 m/s.
- Termination: passed merge, collision, timeout; success requires complete
  merge and no collision.
- V2X: ramp-to-highway and highway-peer coordination opportunities.

### M2: Cooperative bottleneck

- Objective: platoon/bottleneck throughput where the leader and adjacent CAV
  coordinate speed and lane choices.
- Road: generic merge narrowing to a target lane segment.
- CAV starts: distributed before the narrowing, no artificial distance
  relevance guarantee.
- Candidate slots: include out-of-nearest lane vehicles that affect bottleneck
  occupancy.
- Density: high enough for queues but with positive success probability.
- Speed: per-lane 10-25 m/s distribution.
- Termination: all pass bottleneck without collision or timeout.
- V2X: selected partners should cover the lead blocker and entering/relevant
  side traffic.

### I1: Four-way intersection

- Objective: decentralized negotiation of right of way and conflicting routes.
- Road: four-way road with 4 approach directions.
- CAV starts: one CAV per approach, away from collision conflict region.
- Candidate slots: all cross traffic and CAV approaches within 50 m.
- Density: medium initial, configurable spawn probability.
- Speeds: 7-12 m/s.
- Termination: all reached destination, collision, timeout.
- Success: all reach destinations safely.
- V2X: CAVs opposite/adjacent may need intent even if not closest.

### I2: Conflicting trajectories

- Objective: prevent simultaneous crossing conflicts while preserving
  throughput.
- Road: four-way intersection with selected left/straight/right routes.
- CAV starts: 2-2 opposing conflict combinations, with route crossing in the
  center.
- Candidate slots: include cross-approach and on-conflict-path actors.
- Density: medium.
- Termination: all exits complete, collision, timeout.
- Success: coordinated passage with no collision and no unnecessary deadlock.
- V2X: exchange route intent and arrival timing; distant conflicting actors
  can outrank physically nearer non-conflicting actors.

### R1: Roundabout entry

- Objective: safe gap acceptance and cooperative entry yield.
- Road: one roundabout with 4 entry/exit arms.
- CAV starts: 4 CAVs distributed across entries, or a staged 2-entry/2-ring
  setup if native control permits.
- Candidate slots: include circulating vehicles even if the nearest rank is an
  entry arm vehicle with lower relevance.
- Density: medium/high.
- Speeds: 5-12 m/s entry/ring.
- Termination: all CAVs exit or timeout/collision.
- V2X: communicate yield/exit intent.

### R2: Roundabout circulation and exit

- Objective: safe coordinated lane choice and exit sequencing.
- Road: multi-lane roundabout with selected exits.
- CAV starts: ring and adjacent exit arms.
- Candidate slots: distinguish leader/follower and adjacent ring vehicles.
- Density: medium/high.
- Termination: all exits completed, collision, timeout.
- Success: complete selected exits without collision.
- V2X: upstream ring partner can be more relevant than a nearby entry vehicle.

## 6. Compatibility issues

1. **Canonical 56-D is implemented but not universally active:** Highway
   `kinematics25` remains the default and scripts can train a different layout
   than CARLA.
2. **Fixed 4 CAVs are not enforced in CARLA:** constructor `n_agents` could
   violate the research specification. It should remain 4 for new scenarios.
3. **V2X channel is unwired:** no env reset/step integration and no
   per-step cache.
4. **Grouping modules are empty:** no radius, random, learned, or Top-K
   implementation.
5. **No relevance ground truth:** precision/recall/NDCG cannot be calculated
   until a scenario-aware relevance label is defined.
6. **Scenario configurations are absent:** all `*.yaml` scenario files are
   zero-length.
7. **CARLA scenario file:** `carla_env.py` is free-drive/Town-based, with
   `scenario_id="free_drive"` in terminal info. No merge/intersection/roundabout
   initialization API exists.
8. **Roundabout/merge Highway-Env compatibility is unverified:** package
   environments exist, but the repository has no adapter for them and no 4-CAV
   reset test.
9. **Vehicle intent cannot enter candidates:** `intent` is currently excluded
   by `messages_to_candidates()`.
10. **Range mismatch:** channel range may exceed the 50 m candidate normalization
    range.
11. **Existing invalid lane action fallback:** CARLA replaces unavailable left/
    right lane actions with KEEP, so an action is not rejected at env step. This
    should be specified consistently with masks.
12. **State dimensions vary:** Highway `contract56` is 20-D state; CARLA is
    24-D. This is acceptable only with algorithm configuration per environment.
13. **Reward integration TODO remains in the CARLA module header**, although
    reward integration source exists and tests cover it. This is stale
    documentation/development debt, not a testable blocker.

## 7. Risks and dependencies

- CARLA 0.9.16 availability and town assets are external prerequisites.
- Highway-Env 1.12.1 is available in the virtual environment, but scenario
  compatibility still requires live 4-agent smoke tests.
- Traffic Manager route and spawn determinism may vary across CARLA and package
  versions.
- Learned relevance training needs labels or a causal utility definition.
- Packet loss delay RNG ordering must be frozen before paired comparisons.
- Cross-scenario success metrics require comparable success and goal definitions.
- Eight scenarios may overextend first implementation work; no scenario is
  feasible until the shared adapter contract is proven.
- Paired evaluation across seeds must distinguish traffic initialization RNG
  from packet-loss/selection RNG.

## 8. Recommended implementation order

1. Add and test a configuration-driven Highway-Env multi-scenario adapter
   around the four-CAV contract.
2. Implement H1 first and verify 56-D, 6 candidates, masks, 300-step/goal
   termination, and the V2X channel separately.
3. Add H2 lane-changing initialization and tests.
4. Add I1, preferably using the package's installed multi-agent intersection
   support, and define cross-approach relevance labels.
5. Add M1 and M2 merge variants.
6. Add R1/R2 only after roundabout 4-CAV control and exit success are validated.
7. Wire selection/radius/random/learned modules and Top-K=2 only after candidate
   and ground-truth contracts are frozen.
8. Add common evaluation logging last, without changing reward values.

Recommended first scenario: **H1 Dense highway traffic**, because native
`highway-v0` support and the existing 56-D Highway adapter provide the smallest
contract change. H2 is the second choice because it reuses the road model.

## 9. Evaluation compatibility

### Selection quality

- Precision@2, Recall@2, F1@2, NDCG@2: require relevance labels and selected
  candidate IDs; not supported.
- Switching rate: require recorded selected Top-K set per step; not supported.

### Communication

- Messages, deliveries, range/loss/delay drops: V2X channel stats support
  counts and mean age.
- Packet delivery ratio: not in episode CSV; calculable as delivered/sent.
- Bandwidth consumption: not supported; requires message count/size logging.
- Communication latency: mean age is supported by channel; end-to-end
  wall-clock latency is not.
- Useful-message ratio: requires relevance labels and receipt/use logging.

### Driving

- Collision rate: supported in Highway/CARLA episode evaluators.
- Mean TTC/min TTC: CARLA and Highway final info expose TTC support; near
  collision count is not explicit.
- Mean TTC: can be calculated from step TTC if logging is added.
- Goal completion/success: Highway goal mode and CARLA success support it.
- Mean travel time: episode step count/steps at 5 Hz is available; not a
  separately normalized travel-time field.
- Mean speed: supported in evaluator CSV.
- Coordinated maneuver rate: requires scenario-specific success event logging.

### Computational efficiency

- Selection inference time: not supported.
- CPU and memory: not supported; require evaluation-process instrumentation.

## 10. Required tests

Audit minimum for implementation approval:

- Contract tests for 4 agents, 56-D observations, six candidates, and six masks.
- CARLA-independent builder tests already cover canonical features.
- Highway adapter tests already cover 56-D, candidate sorting/padding,
  masks, goals, and seed determinism.
- V2X channel tests already cover range/loss/delay/reset/count invariants.
- Add source grouping tests for radius, random, learned, Top-K=2, ties,
  invalid/empty candidates, and set stability.
- Add V2X integration tests: once-per-step clock, independent per-agent inbox,
  reset, packet counters, delayed stale candidate values.
- Add H1 dense traffic test: seed determinism, candidate full slots,
  terminal behavior, and no contract change.
- Add H2 lane-change test: mask changes, adjacent-lane vehicles, and
  coordinated maneuver success.
- Add merge/intersection/roundabout adapter tests with fake or live
  environments, if dependencies are available.
- Add state-dimension tests for CARLA and every Highway-Env layout.
- Add selection-quality synthetic tests with fixed relevance labels.
- Add evaluation metric tests for PDR, useful ratio, switching rate,
  travel time, near-collision threshold, and coordinated maneuver.
- Add paired-seed tests to ensure scenario, traffic, and channel RNG streams
  are reproducible and controlled.

## 11. Explicit decisions requiring human approval

1. Select the implementation backend for the first scenario: existing
   Highway-Env adapter or CARLA 0.9.16.
2. Decide whether Highway-Env scenarios or CARLA scenarios are the primary
   scenario suite; the repository cannot currently preserve a common
   scenario implementation through both.
3. Approve 4-CAV enforcement in every constructor and freeze out
   `n_agents` variations.
4. Approve the first relevance ground-truth definition and TTC/time-horizon
   thresholds.
5. Approve whether intent is added outside the canonical candidate block or
   left excluded.
6. Approve delay unit and loss timing if V2X is wired: decision step versus
   simulator tick, loss at send versus delivery.
7. Approve V2X broadcast membership: CAV-only or CAV plus background vehicles.
8. Approve separate RNG streams for scenario, traffic, loss, and selection.
9. Approve the initial default traffic density/speed distributions for H1 and
   all later scenarios.
10. Approve near-collision TTC threshold and the successful-coordinated-maneuver
    event definition.
11. Approve whether R1/R2 remain in the common 4-CAV suite or are deferred as
    adapter-dependent research scenarios.
