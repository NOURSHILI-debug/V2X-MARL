# Scenario Implementation Plan

This plan describes the future implementation stages only. It does not modify
source code or run retraining. It preserves the frozen 4-agent, 6-candidate,
56-D observation, 6-action, Top-K=2, V2X, CTDE, and reward contracts.

## Stage 0: Contract and test baseline

**Objective:** make the existing 56-D Highway/CARLA contract the only
implementation baseline before scenario work.

Expected files:

- `envs/base.py`
- `envs/obs_builder.py`
- `envs/highway_env.py`
- `envs/carla_env.py`
- `docs/env_spec.md`
- `docs/env_spec_section12.md`
- `tests/test_contract.py`
- `tests/test_obs_builder.py`
- `tests/test_highway_adapter.py`

Work:

- Keep 56-D canonical.
- Enforce/freeze 4 agent IDs for scenario implementations.
- Add no reward or action-ID changes.
- Correct stale CARLA reward TODO documentation only if it is part of a later
  contract documentation update.

Acceptance:

- Both environment families expose exactly 4 agents, 56-D observations, 6
  candidates, and 6 action masks.
- Contract tests pass with strict mode.
- Existing reward tests remain unchanged.

## Stage 1: Configuration and scenario registry

**Objective:** add a data-only scenario selection layer without duplicating
environment logic.

New files potentially required:

- `envs/scenarios/base.py`
- `envs/scenarios/registry.py`
- `envs/scenarios/highway_dense.py`
- `envs/scenarios/highway_lane_change.py`
- Per-scenario test modules under `tests/scenarios/`

Existing files to update:

- `envs/scenarios/__init__.py`
- `configs/scenarios.yaml`
- `configs/env.yaml`
- `envs/highway_env.py`
- `envs/carla_env.py` only if a CARLA scenario backend is approved
- `docs/SCENARIO_ARCHITECTURE_AUDIT.md` and this plan as findings evolve

Dependencies:

- Human approval of configuration schema and backend.
- No new package dependency is required for Highway-Env.

Acceptance:

- A scenario name resolves to 4 CAV IDs, 6 candidate slots, and the canonical
  observation layout.
- Invalid or unavailable scenarios fail explicitly.
- Seed, traffic density, spawn profile, and termination are reproducible.

## Stage 2: V2X integration

**Objective:** wire the existing channel once per decision step.

Existing files:

- `envs/v2x_channel.py`
- `envs/highway_env.py`
- `envs/carla_env.py`
- `envs/obs_builder.py`
- `docs/v2x_channel.md`

New files potentially required:

- `envs/v2x_integration.py` for shared send/receive caching
- `tests/test_v2x_integration.py`

Dependencies:

- Human decisions on broadcast membership, intent, delay unit, and loss timing.
- Exact channel range/candidate-range relationship.

Acceptance:

- Channel advances exactly once per decision step.
- Every CAV receives an independent inbox.
- The candidate block is canonical and zero-padded.
- Range/loss/delay counters and stale age are observable.
- Reset clears in-flight messages and replays the seed deterministically.

## Stage 3: H1 dense highway

**Objective:** implement and prove the smallest Highway-Env scenario.

Existing files:

- `envs/highway_env.py`
- `configs/scenarios.yaml`
- `scripts/train_common.py`
- `scripts/train_qmix_highway.py`
- `scripts/train_vdn_highway.py` if present
- `scripts/train_mappo_highway.py`
- `scripts/evaluate_comparison.py`
- `tests/test_highway_adapter.py`

New files potentially required:

- `envs/scenarios/highway_dense.py`
- `tests/scenarios/test_h1_dense_highway.py`

Acceptance:

- H1 creates 4 CAVs and at least 12 road actors where configured.
- Contract56 is 56-D on every step.
- Six candidate slots are sorted and padded consistently.
- All six actions remain available or are masked by the lane context.
- Collision, goal, timeout, and native truncation are consistent.
- Fixed seed reproducibility passes.

## Stage 4: H2 cooperative lane changing

**Objective:** add cooperative lane-change initialization and success logging.

Existing files:

- `envs/highway_env.py`
- `envs/scenarios/__init__.py`
- `configs/scenarios.yaml`
- `tests/test_highway_adapter.py`

New files:

- `envs/scenarios/highway_lane_change.py`
- `tests/scenarios/test_h2_lane_change.py`

Acceptance:

- Lane-change masks match adjacent lanes.
- Lead-blocking vehicle and cooperative yielding partner are initialized
  deterministically.
- Successful coordinated lane change is separately logged.
- No artificial relevance guarantee is encoded.

## Stage 5: Grouping Top-K=2

**Objective:** implement the three selection variants behind one interface.

Existing files:

- `grouping/__init__.py`
- `grouping/radius.py`
- `grouping/random_sel.py`
- `grouping/learned_sel.py`
- `envs/highway_env.py`
- `envs/carla_env.py`
- `envs/obs_builder.py`
- `marl/*`

New files potentially required:

- `tests/test_grouping.py`
- `tests/test_selection_metrics.py`
- `grouping/learned_model.py` only if a learned relevance model is required

Dependencies:

- Approved relevance labels/ground truth.
- Candidate IDs stable through the channel and observation block.

Acceptance:

- Radius/random/learned return exactly Top-K=2 valid IDs when at least 2 valid
  candidates exist.
- Fewer candidates return only valid IDs with documented padding behavior.
- Random and learned selection are seeded deterministically.
- Switching and selected IDs are logged per agent and step.

## Stage 6: I1/I2 intersections

**Objective:** add Highway-Env multi-agent intersection support or a verified
adapter.

Existing/package dependencies:

- Installed `intersection-multi-agent-v0/v1/v2`
- `envs/highway_env.py`
- `envs/obs_builder.py`
- `envs/termination.py`
- `configs/scenarios.yaml`

New files:

- `envs/scenarios/intersection_four_way.py`
- `envs/scenarios/intersection_conflicts.py`
- `tests/scenarios/test_i1_intersection.py`
- `tests/scenarios/test_i2_conflicts.py`

Acceptance:

- 4 CAVs receive independent observations/masks.
- Six candidates include cross-traffic and conflicting approaches.
- Success means all selected routes reached or exited.
- Collision and timeout behavior is shared across agents.

## Stage 7: M1/M2 merging

**Objective:** add merge and bottleneck scenarios after base merge feasibility
is verified.

New files:

- `envs/scenarios/merge_on_ramp.py`
- `envs/scenarios/merge_bottleneck.py`
- `tests/scenarios/test_m1_merge.py`
- `tests/scenarios/test_m2_bottleneck.py`

Existing files:

- `envs/highway_env.py`
- `envs/scenarios/__init__.py`
- `configs/scenarios.yaml`

Acceptance:

- Ramp entrants remain in the 6-candidate set where relevant.
- A merge-success event is distinct from general goal completion.
- Dense and bottleneck configurations are reproducible.

## Stage 8: R1/R2 roundabouts

**Objective:** add roundabout support only after 4-CAV adapter feasibility is
proven.

New files:

- `envs/scenarios/roundabout_entry.py`
- `envs/scenarios/roundabout_exit.py`
- `tests/scenarios/test_r1_entry.py`
- `tests/scenarios/test_r2_circulation.py`

Acceptance:

- 4 CAV control and selected exits are verified.
- Candidate relevance includes circulating and entry/exit vehicles.
- No direct highway-env assumptions are accepted without adapter tests.

## Stage 9: Common evaluation

**Objective:** add common scenario/selection/communication/driving metrics
without changing reward.

Existing files:

- `scripts/evaluate_comparison.py`
- `scripts/plot_comparison.py`
- `scripts/train_common.py`
- `marl/*`

New files potentially required:

- `scripts/evaluate_scenarios.py`
- `scripts/selection_metrics.py`
- `scripts/communication_metrics.py`
- `tests/test_scenario_metrics.py`

Dependencies:

- Ground-truth relevance label and near-collision threshold.
- A process/resource instrumentation library or standard-library sampling;
  no dependency before approval.

Acceptance:

- Selection: Precision@2, Recall@2, F1@2, NDCG@2, switching rate.
- Communication: messages, delivered/dropped, PDR, mean latency/age,
  useful-message ratio, bandwidth estimate.
- Driving: collision, near-collision, mean/min TTC, goal completion,
  travel time, mean speed, coordinated-maneuver rate.
- Efficiency: selection inference time, CPU, memory.
- Episode CSV and summary output include scenario ID and seed.

## Testing strategy

Run in increasing scope:

1. Unit contract and pure builder tests.
2. V2X channel and integration tests.
3. Highway-H1/H2 tests.
4. Intersection/merge/roundabout fake or live tests.
5. Synthetic selection-quality tests.
6. Held-out episode evaluation across fixed seeds.
7. Full non-CARLA suite; separate CARLA suite only with a live server or its
   fake-world test harness.

Key existing commands:

```powershell
python -m pytest tests/test_obs_builder.py tests/test_highway_adapter.py tests/test_v2x_channel.py -v
python -m pytest tests/test_contract.py -v
$env:CONTRACT_STRICT="1"; python -m pytest tests/test_contract.py -v
```

CARLA tests should remain server-gated unless the fake-CARLA test harness
is sufficient for the modified behavior.

## Documentation requirements

Update:

- `docs/env_spec.md` only if the frozen contract is explicitly changed.
- `docs/env_spec_section12.md` if contract precedence or exact dimensions change.
- `docs/v2x_channel.md` after channel integration.
- `docs/highway_adapter.md` after new Highway-Env scenarios.
- `docs/SCENARIO_ARCHITECTURE_AUDIT.md` after architecture decisions.
- This plan after stage completions and acceptance outcomes.
- Create per-scenario documentation for road geometry, spawn distributions,
  success, safety, and relevance opportunities.

## Current blockers requiring approval

- H1/H2 backend and first scenario approval.
- Contract for scenario-specific relevance labels.
- V2X integration semantics.
- 4-CAV and Top-K=2 enforcement location.
- Evaluation thresholds and metrics.
- CARLA versus Highway-Env priority.
