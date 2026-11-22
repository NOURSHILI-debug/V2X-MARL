# Scenario Spec for CARLA Partner

## Status

Forcing roles and seed ranges in this document are specified from the Highway
implementation. They are ready for CARLA translation as configuration and
acceptance criteria. CARLA files remain owned by the CARLA partner; no CARLA
code was changed in this batch.

## Scope

- In scope: revised H1 (`h1_forcing`) and H2 lane-change (`h2_lane_change`).
- Not in scope: I2, M1, reward changes, observation/action changes, Top-K or
  grouping changes, learner training.
- Seeds for CARLA parity checks: 100-109. Never use reserved seeds 7001-7003 or
  9001-9030.

## Fixed contract

- Four CAV IDs: `cav_0..cav_3`.
- At least 12 background vehicles per episode.
- Decision rate 5 Hz; 300 decision steps / 60 s.
- Canonical `contract56` observation: 8 ego + 6 route + 6x6 candidates + 6 mask.
- Candidate range 50 m; six nearest valid candidates; zero padding.
- Actions: 0 keep, 1 accelerate, 2 decelerate, 3 emergency brake, 4 left, 5 right.
- Spawn states must be collision-free and seed-reproducible.

## Forcing semantics

Each profile combines four reusable role types:

| Role | Required behavior |
| --- | --- |
| slow blocker | Low-speed lead in a CAV lane; creates a sustained LeadTTC pressure below 3 s. |
| fast approacher | Faster same-lane follower behind a CAV; constant approach speed until conflict, not car-following. In Highway this uses a kinematic vehicle so keep-speed is rear-ended. |
| alongside | Seeded traffic in adjacent lanes that shapes legal/unsafe lane-change gaps. |
| scheduled hazard | Nominal-speed vehicle that brakes once at a configured step, creating a later longitudinal conflict. |

Implementation details that must be preserved:

- CAVs may share a lane only when longitudinally staggered; minimum longitudinal
  separation is validated.
- Generic background traffic must avoid occupied slots and preserve protected
  adjacent-lane gaps around CAVs so at least one escape lane is feasible.
- Role speeds must not be randomized away by the native background behavior.
- Scheduled hazard braking is time-based on decision step, not distance-based.

## `h1_forcing`

- Background: 20 vehicles; spacing 18.0 m ±0.5; speeds uniform 18-25 m/s.
- CAV lanes `[1, 1, 0, 2]`; scalar start x 80.0 m; longitudinal gap 25.0 m;
  CAV speed 22.0 m/s.
- Slow blocker: lane 1, x 150.0 m, 8.0 m/s.
- Fast approacher: lane 2, x 138.0 m, 28.0 m/s.
- Alongside: lane 0 x 145.0 m; lane 2 x 240.0 m; lane 3 x 110.0 m, all 22.0 m/s.
- Scheduled hazard: lane 3, x 200.0 m, 20.0 m/s, brake at step 80 to 12.0 m/s.

Acceptance on Highway seeds 100-109: G1 1.00, G2 0.90, G3 median 8, G4 pass,
G5 headroom 0.777. G6 currently fails; CARLA should reproduce the measurement
rather than tune the criterion.

## `h2_lane_change`

- Background: 20 vehicles; spacing 18.0 m ±0.5; speeds uniform 18-25 m/s.
- CAV lanes `[1, 1, 0, 2]`; scalar start x 80.0 m; longitudinal gap 20.0 m;
  CAV speed 22.0 m/s.
- Slow blocker: lane 1, x 145.0 m, 8.0 m/s.
- Fast approacher: lane 2, x 123.0 m, 28.0 m/s.
- Alongside: lane 0 x 160.0 m; lane 2 x 190.0 m; lane 3 x 100.0 m, all 22.0 m/s.
- Scheduled hazard: lane 3, x 190.0 m, 20.0 m/s, brake at step 80 to 12.0 m/s.

Acceptance on Highway seeds 100-109: G1 1.00, G2 1.00, G3 median 7.5, G4 pass,
G5 headroom 0.697. G6 currently fails; CARLA should reproduce the measurement
rather than tune the criterion.

## Gate definitions for CARLA

- G1: keep-speed collision in >=50% episodes.
- G2: competent controller collision-free for 300 steps in >=80% episodes.
- G3: random median survival >=5 steps.
- G4: no unavoidable reset collisions, demonstrated by G2.
- G5: hazard defined by min TTC <5 s; report fallback share; require lowest-TTC
  candidate outside top-2 nearest share >0.05.
- G6: VoI conditions `none`, `nearest2`, `random2`, `oracle2`, `all6`; pass only
  if oracle collision rate < nearest collision rate and none collision rate >
  all6 collision rate.

Do not edit gate thresholds or criteria to force a pass. Report failures with
per-seed data.
