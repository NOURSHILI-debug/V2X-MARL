# Milestone 9A-Fix — Pre-Pilot Readiness Corrections

**Repository:** `C:\Users\souro\Desktop\V2X-MARL`  
**Branch/HEAD (inspected):** `feat/teammate-selection` @ `a82dc6d`  
**Scope:** Corrective, narrowly scoped fixes to M9A paired-experiment infrastructure identified in the final pre-pilot read-only review. **No CARLA launched/connected. No pilot/simulation/experiments run. No new experimental results generated.**

## 1. Original blockers

From the M9A final pre-pilot review:
1. **Route validity:** environment could continue when route application failed; validator compared `route_applied` but could accept pairs where both values were false/missing/malformed were not strictly rejected.
2. **CARLA lifecycle/simulator-state isolation:** fresh Python env object/`close()` did not guarantee clean CARLA server state (unmanaged actors, TM state, traffic lights/phase, weather/world settings, sync config, cleanup failures not invalidating pairs).
3. **Initial scenario snapshot/validation:** snapshot/validator did not sufficiently capture CARLA-specific conditions (world/map/settings/weather, controlled-agent descriptors/route status, background/unmanaged vehicles, traffic-light state/timing/phase limitations) and policy for must-match/diagnostic/missing-required was not explicit.
4. **Incomplete-run safety:** manifest/run state did not fully distinguish running/incomplete/failed/completed, cleanup/persistence failures and missing artifacts could be insufficiently gated, and final completion could be represented too optimistically.

## 2. Root causes found in source code

- **`envs/carla_env.py:reset` (route application):** after `tm.set_path` retry, code logged a warning when route remained unapplied (`logger.warning`) but still set `route_applied[agent_id]=applied`, built successful `reset_info`, set `reset_complete=True`, and returned observations/info. No exception/failure reason prevented episode continuation.
- **`scripts/milestone9/validate_pair.py`:** `route_applied` was compared only as optional spawn-metadata scalar equality when present (`spawn_metadata.route_applied`), so missing/None/malformed were not strictly rejected as required preconditions and both-false-equal cases were not invalidated by a route-validity gate.
- **`scripts/milestone9/run_paired.py`:** no post-reset route-validity gate; `env.close()` exceptions were swallowed in `finally`; no lifecycle isolation status/readiness metadata; persistence errors could propagate/leave inconsistent registration; experiment status initialized as `created` and set to `completed` at end regardless of pair validity/interruption semantics.
- **`scripts/milestone9/logging.py`:** `ExperimentLayout` supported `created`/`completed` usage but not explicit `running/incomplete/failed/completed` lifecycle semantics; manifest did not encode cleanup/persistence lifecycle details.
- **Snapshot/validator coverage:** `collect_initial_scenario`/`scenario_signature_from_reset` captured limited reset/agent initial state only (no world settings/weather, no controlled-agent descriptors/route-status schema, no background/unmanaged vehicles, no traffic-light descriptors/state/timing limitations). Validator lacked explicit must-match/allowed-differ/diagnostic-only/missing-required policy and could accept records missing newly required information.

## 3. Files changed and why

| File | Purpose | Exact change (summary) | Previously modified/untracked? | Existing user changes preserved? |
|---|---|---|---|---|
| `envs/carla_env.py` | Enforce route validity at reset | In `reset()`, store per-agent `route_applied` as bool; if any route remains unapplied after retry, log error and raise `RuntimeError("Route application failed for agents: ...")` so reset does not complete successfully (existing `finally` cleans up partial reset). | **Yes, pre-existing uncommitted modifications present** | **Yes** — only a narrow additive block inserted in route-application handling; prior uncommitted M9A/pre-existing changes preserved. No protected files touched. |
| `scripts/milestone9/run_paired.py` | Runner safety gates, lifecycle, snapshot, persistence/incomplete-run handling | Added defensive snapshot helpers (`world`, controlled-agent descriptors, background/unmanaged vehicles, traffic lights with best-effort timing/limitations); enriched `collect_initial_scenario` with world/controlled/background/traffic-light/snapshot policy fields; added post-reset route-validity precondition (fails episode before stepping, preserves reset/scenario artifacts); recorded lifecycle metadata (`cleanup_attempted/status/error`, isolation protocol, `live_carla_isolation_verified=False`, readiness gate) and treated cleanup failure as episode failure; made persistence best-effort with final-status recording, artifact existence checks, checksum registration on final content, and persistence failure invalidation; updated `run_paired_experiment` to mark `running`, persist incremental state, catch interruptions, and finalize manifest as `completed` only when all requested pairs processed and all episodes/pair validations succeed, else `failed`/`incomplete`. `--dry-run-config` behavior preserved. | **Yes, untracked M9A file (created earlier in session)** | **Yes** — additive corrective edits to M9A implementation. |
| `scripts/milestone9/validate_pair.py` | Strict route/lifecycle/persistence/snapshot validation | Added strict `route_applied` precondition per episode (must exist, type exactly `bool`, value `True`; missing/False/None/malformed -> `infrastructure_failure`), removed permissive equality-only route handling, added cleanup/persistence failure checks, added required snapshot schema checks and explicit policy enforcement (world/map/settings/sync/fixed delta, controlled-agent descriptors/route status, background+traffic-light sections required; missing required invalidates pair including identical-missing cases; CARLA actor IDs/background details treated as diagnostic/not identity criteria). | **Yes, untracked M9A file** | **Yes**. |
| `scripts/milestone9/logging.py` | Manifest lifecycle states | `ExperimentLayout` initializes status as `running`; added status validation for `running/incomplete/failed/completed`; exposed manifest accessor. Atomic writes/checksums/no-overwrite behavior preserved. | **Yes, untracked M9A file** | **Yes**. |
| `tests/test_milestone9_validation.py` | Route-validity and fixture alignment | Updated base fixture with required `route_applied` and required snapshot schema; added 7 required route-validity cases (both applied pass, first false, second false, both false, missing, None, malformed string). | **Yes, untracked M9A test file** | **Yes** (existing cases preserved/updated). |
| `tests/test_milestone9_snapshot_policy.py` | Snapshot/lifecycle validation policy tests | New CARLA-independent tests for required snapshot sections, missing-required (including both missing), world-setting mismatch, cleanup failure invalidation, persistence failure invalidation. | **New file** | N/A |
| `tests/test_milestone9_lifecycle.py` | Lifecycle orchestration, route gate, persistence/incomplete-run, dry-run | New CARLA-independent tests using fake env/controller: cleanup failure invalidates episode/pair, reset failure invalidates, route invalid in reset info invalidates, successful mocked pair completes with lifecycle/snapshot metadata and preserved artifacts, persistence failure invalidates, dry-run config works without CARLA. | **New file** | N/A |
| `docs/audit/milestone9a_fix.md` | Documentation | This report. | **New file** | N/A |

**Not modified (preserved):** `envs/v2x_channel.py`, `tests/test_v2x_channel.py`, `docs/v2x_channel.md`, `envs/obs_builder.py` (protected), `results/milestone7/`, `results/milestone8/`, `results/milestone8a/` and other existing result artifacts, rewards, termination semantics, MARL algorithms, 56-D observation contract/actions/candidates/action masks, V2X behavior, selection definitions, research question/configuration. `grouping/__init__.py` remains pre-existing modified/uncommitted and was **not** changed in this fix.

## 4. Route validity behavior

- **Environment reset (`envs/carla_env.py:reset`):** each agent route is applied via Traffic Manager with one retry; per-agent `route_applied` stored as bool. If any agent remains unapplied, reset raises `RuntimeError` with failed-agent list and does **not** return success (partial reset cleaned up by existing `finally`). No silent acceptance and no unintended route substitution.
- **Runner precondition (`scripts/milestone9/run_paired.py:run_episode`):** after reset/scenario capture, episode fails with explicit reason if overall `initial_scenario.route_applied`/`reset_info.route_applied` is not exactly boolean `True` or if any per-agent reset `route_applied` is not `True`. Failed episode artifacts (including reset/scenario info) are preserved.
- **Validator precondition (`scripts/milestone9/validate_pair.py`):** for each episode, `route_applied` must exist, be type `bool`, and be `True`; missing/`False`/`None`/malformed (e.g. string `"true"`) -> `infrastructure_failure` issue; pair cannot be valid. Per-agent controlled-agent `route_status.route_applied` is also required/validated as strict boolean `True` when present in required snapshot.
- **Result:** invalid route cannot be classified as a valid pair; failed route episodes remain recorded for diagnosis.

## 5. CARLA lifecycle/isolation protocol

**Implemented (narrow, architecture-compatible):**
- **Per-episode lifecycle:** runner creates a fresh env per episode, performs reset, captures initial snapshot before stepping, runs to env termination/truncation, and always attempts cleanup in `finally`.
- **Cleanup handling:** `env.close()` outcome recorded explicitly (`cleanup_attempted`, `cleanup_status=success|failed|skipped`, `cleanup_error`, `close_called`). Cleanup exceptions are **not** swallowed silently; cleanup failure marks the episode failed and invalidates the pair.
- **Reset/lifecycle failures:** reset exceptions fail the episode with recorded reason; lifecycle/persistence failures are treated as infrastructure failures in validation/aggregation.
- **Metadata/readiness gate:** episodes record `isolation_protocol="fresh_env_per_episode_create_reset_close"`, `live_carla_isolation_verified=False`, and `readiness_gate="live_carla_verification_required"`. Manifest/summary statuses distinguish processing outcomes (`running/incomplete/failed/completed`) and pair validity.
- **No broad rewrite:** no automatic Town03 reload between episodes introduced (impact on client/TM/world/sensors/refs unverified without live CARLA); no destruction of arbitrary unmanaged/external actors.

**Source-verified limits:** Python env close/reset cleanup manages project-owned actors/sensors/sync transitions as implemented in `envs/carla_env.py`, but this does **not** prove CARLA server/world/TM/traffic-light/weather/unmanaged-actor state is fully equivalent across episodes or pair members. Reliable simulator isolation/reset cannot be claimed without live CARLA verification.

## 6. Initial snapshot schema and validation rules

**Snapshot (captured pre-step in `scripts/milestone9/run_paired.py:collect_initial_scenario`, best-effort/serializable):**
- **World:** `map_name`, `world_settings` (`synchronous_mode`, `fixed_delta_seconds`, `no_rendering_mode` when available), `weather` (available fields or error marker).
- **Controlled agents:** per experiment agent (`cav_*`) descriptor with managed flag, `experiment_agent_id`, vehicle `type_id`/attributes when available, transform, velocity/speed, destination, `route_status` (`route_applied`, `route_length_points`), plus CARLA actor ID recorded as **diagnostic**.
- **Initial state (physical):** per-agent position/orientation, speed, destination/orientation, route points/length, per-agent route-applied indicator.
- **Background/unmanaged actors:** `background_and_traffic_lights.vehicles` (relevant vehicles with managed/unmanaged classification, stable descriptors where possible: type, transform, velocity, attributes subset), `managed_vehicle_actor_ids_diagnostic`, capture note.
- **Traffic lights:** descriptors with transform/state and best-effort timing/phase fields; explicit limitation recorded when unavailable/unreliable.
- **Policy metadata:** recorded in snapshot (`snapshot_policy`) for auditability.

**Validation policy (`scripts/milestone9/validate_pair.py`):**
- **Must match / required:** pair identity and scenario seeds; config equivalence (town/n_agents/max_steps/spawn/candidate source/V2X/controller); overall `route_applied` exists, type exactly `bool`, value `True` for both members; required snapshot sections present (`world` with `map_name`, `world_settings.synchronous_mode`, `world_settings.fixed_delta_seconds`, `weather`; `controlled_agents` for all agents with `experiment_agent_id` and `route_status`; `background_and_traffic_lights` with `vehicles` and `traffic_lights` lists). Map name and sync/fixed-delta compared when present as required. Controlled-agent physical initial states (position/orientation/speed/destination/route) compared within tolerances. Per-agent `route_status.route_applied` must be strict boolean `True` when provided in required snapshot.
- **Allowed to differ:** CARLA actor IDs/diagnostic IDs (unstable, especially across reloads), strategy and strategy-specific selection/V2X seeds.
- **Diagnostic only (not required to match identically):** background/unmanaged actor full-set details and best-effort traffic-light timing/phase details; weather details unless future protocol requires.
- **Missing required invalidates pair:** yes; identical missing required information does **not** pass.
- **Tolerances:** position `1e-3 m`, orientation `1e-2 deg` (cyclic yaw), speed `1e-3 m/s`, route point `1e-3 m`.

## 7. Incomplete-run handling

- **Manifest states:** `running`, `incomplete`, `failed`, `completed` (`ExperimentLayout.update_status` validates states).
- **Experiment lifecycle:** marked `running` at start; pair/episode progress persisted incrementally. On uncaught interruption/exception, manifest finalized as `incomplete` (or `failed` when appropriate) with fatal error recorded. On normal completion, status is `completed` **only** if all requested pairs processed and all episodes/pair validations succeed (`all_pairs_valid_for_analysis`); otherwise `failed` with reasons/counts. Final summary/manifest written in `finally` (best-effort).
- **Episode/pair safety:** failed execution, cleanup failure, persistence/artifact failure, missing required snapshot/route validity, or missing episode records invalidate the pair for analysis. Raw/episode artifacts preserved where possible.
- **Persistence:** atomic writes used; file existence checked; checksums/registration reflect final written content; persistence failures mark episode failed and recorded best-effort (including attempted raw record updates). Missing artifacts cannot produce a valid persisted success.
- **No-overwrite:** existing `ExperimentLayout` non-empty directory guard and atomic replace behavior preserved.

## 8. New tests and exact results

**New/updated targeted tests (CARLA-independent):**
- `tests/test_milestone9_validation.py` — updated fixtures + 7 route-validity cases.
- `tests/test_milestone9_snapshot_policy.py` — required snapshot schema, missing-required (including both missing), world-setting mismatch, cleanup/persistence invalidation.
- `tests/test_milestone9_lifecycle.py` — cleanup failure, reset failure, route-invalid reset, successful mocked pair lifecycle/snapshot/artifact recording, persistence failure invalidation, dry-run config.

**Commands and results:**
1. `C:\Users\souro\venvs\v2x-carla312\Scripts\python.exe -m pytest tests/test_milestone9_validation.py tests/test_milestone9_snapshot_policy.py tests/test_milestone9_lifecycle.py tests/test_milestone9_logging.py -q`  
   **Result:** `30 passed in 1.57s`
2. `C:\Users\souro\venvs\v2x-carla312\Scripts\python.exe -m pytest tests/test_milestone9_config.py tests/test_milestone9_reproducibility.py tests/test_milestone9_logging.py tests/test_milestone9_validation.py tests/test_milestone9_integration.py tests/test_milestone9_snapshot_policy.py tests/test_milestone9_lifecycle.py -q`  
   **Result:** `57 passed in 1.62s`
3. `C:\Users\souro\venvs\v2x-carla312\Scripts\python.exe scripts\milestone9\run_paired.py --dry-run-config` (output redirected; no CARLA)  
   **Result:** exit code `0`
4. `C:\Users\souro\venvs\v2x-carla312\Scripts\python.exe -m pytest tests -k "not carla" -q`  
   **Result:** `370 passed, 8 skipped, 18 deselected in 62.92s (0:01:02)`

## 9. Remaining limitations

- **Live CARLA isolation not proven:** fresh env/close/reset does not guarantee equivalent CARLA server world/TM/traffic-light phase/weather/unmanaged-actor state across pair members or episodes. No world reload was added to avoid unverified side effects.
- **Background/traffic-light equivalence:** snapshots capture best-effort descriptors/states; unmanaged actor populations and traffic-light timing/phase may vary or be incompletely observable. Validator treats unstable/best-effort details as diagnostic per policy, so physical/behavioral equivalence is not guaranteed.
- **CARLA actor IDs:** diagnostic only; not used as sole identity criterion.
- **Env route-application behavior changed:** `envs/carla_env.py:reset` now fails hard on unapplied routes (correct for controlled experiments) instead of warning-and-continuing; this may affect callers relying on previous permissive behavior (not exercised by CARLA-free suite).
- **Cleanup semantics:** runner records/invalidates on `close()` exceptions and uses explicit lifecycle gates, but cannot verify internal partial CARLA-side cleanup without live simulator introspection.
- **Atomic persistence bounds:** same-filesystem atomic replace/fsync used; catastrophic interruption during multi-file/manifest updates can still leave partially advanced but explicitly non-completed manifests/artifacts (not marked completed/valid).

## 10. What still requires live CARLA verification

- Actual CARLA server/world/TM state cleanliness and reproducibility across sequential paired episodes (including traffic-light phase, weather, unmanaged actors, sync/TM behavior).
- End-to-end route application success, spawn/cluster feasibility, and `route_applied` correctness under real CARLA 0.9.16/Town03 conditions.
- Correctness and completeness of snapshot fields collected from live CARLA APIs (world settings/weather/traffic-light timing availability, vehicle attributes, actor filtering/classification).
- That cleanup (`env.close()`/actor/sensor destruction) fully releases simulator resources and restores safe state between runs.
- One-pair pilot execution, paired scenario equivalence in practice, and runtime stability/performance (explicitly not performed). 

**Protected files unchanged:** `envs/v2x_channel.py`, `tests/test_v2x_channel.py`, `docs/v2x_channel.md`, `envs/obs_builder.py`. **No CARLA run, no experiments, no results modified.**