# Milestone 9A — Reproducible Experimental Infrastructure Implementation

**Repository:** `C:\Users\souro\Desktop\V2X-MARL`  
**Branch/HEAD (inspected):** `feat/teammate-selection` @ `a82dc6d`  
**Scope:** Additive, backward-compatible infrastructure for reproducible paired `nearest_k` vs `random_k` experiments using the identical deterministic observation-conditioned controller. **No large experiments launched. No CARLA pilot executed (explicit authorization not provided).**

## 1. Architecture

The Milestone 9A implementation is organized as additive modules under `scripts/milestone9/` plus CARLA-free tests:

- **`scripts/milestone9/config.py`**: Validated, serializable experiment configuration and explicit seed derivation helpers. Enforces baseline protocol values (Town03, 4 CAVs, cluster spawn 45 m, V2X candidate source, 50 m comm/candidate range, `k=2`, horizon 300 decision steps, `loss=0`, `delay=0`, `cavs_and_background`) and requires separate `scenario_seed` / `selection_seed` / `v2x_seed` identities.
- **`scripts/milestone9/logging.py`**: CARLA-free artifact management for `results/milestone9/` (layout, atomic JSON/CSV/text writes, checksums, manifest, file/episode registration, ending-reason derivation, compact observation/selection/V2X helpers). Prevents overwriting existing non-empty experiment directories.
- **`scripts/milestone9/run_paired.py`**: Paired runner (pilot-only when explicitly requested). For each pair `(scenario_seed, episode_index)` it selects/counterbalances strategy order, builds one identical controller instance, creates a **fresh CARLA environment per episode** with explicit per-episode seeds, runs to environment termination/truncation at configured horizon (no conflicting independent loop cap), records standardized episode/step artifacts, validates pair equivalence, marks failed members explicitly, retains raw failed data, and writes incremental summaries/manifest.
- **`scripts/milestone9/validate_pair.py`**: CARLA-free pair equivalence checker with configurable float tolerances. Compares required initial/config fields, reports explicit issues, never silently marks a mismatched pair valid, and distinguishes categories (`initial_scenario_mismatch`, `infrastructure_failure`; later trajectory divergence intentionally not required to match).
- **`envs/carla_env.py` (minimal additive change in this phase)**: added optional `max_steps=None` constructor parameter (default preserves `MAX_STEPS`) with validation. This supports clean, explicit horizon configuration without changing global `MAX_STEPS` or default behavior.

Design constraints respected: no modification of protected files, 56-D observation contract/normalization, rewards, termination semantics, existing MARL algorithms, or original V2X channel implementation.

## 2. Files created / modified

### Created (additive)
- `scripts/milestone9/__init__.py`
- `scripts/milestone9/config.py`
- `scripts/milestone9/logging.py`
- `scripts/milestone9/run_paired.py`
- `scripts/milestone9/validate_pair.py`
- `tests/test_milestone9_config.py`
- `tests/test_milestone9_reproducibility.py`
- `tests/test_milestone9_logging.py`
- `tests/test_milestone9_validation.py`
- `tests/test_milestone9_integration.py`
- `docs/audit/milestone9a_implementation.md` (this report)

### Modified (additive, minimal, this phase)
- `envs/carla_env.py`: `CarlaMultiAgentEnv.__init__` now accepts optional `max_steps=None` and sets `self.max_steps` to `MAX_STEPS` when `None` (backward compatible) or to a validated positive integer when provided. No other runtime logic/semantics changed in this M9A edit.

### Pre-existing modified/untracked work preserved (not changed by M9A)
- Pre-existing uncommitted modifications remain in `envs/carla_env.py` and `grouping/__init__.py` (from prior milestones/work). M9A only added the minimal horizon parameter handling in `envs/carla_env.py`.
- Existing untracked artifacts/directories remain untouched, including `controllers/`, `envs/v2x_adapter.py`, `marl/`, `policies/`, `results/milestone7/`, `results/milestone8/`, `results/milestone8a/`, prior scripts/tests/docs, etc.
- Protected files verified unchanged by this audit/implementation phase: `envs/v2x_channel.py`, `tests/test_v2x_channel.py`, `docs/v2x_channel.md`, `envs/obs_builder.py`.

No files were staged, committed, or pushed.

## 3. Seed-management strategy

### Explicit seed domains
- **Scenario seed (`scenario_seed`)**: identical for both members of a pair; passed to env construction (`seed=scenario_seed`) and `env.reset(seed=scenario_seed)` to drive scenario/spawn/route/TM setup as currently implemented by env.
- **Selection seed (`selection_seed`)**: explicit per episode and per strategy, derived deterministically by `derive_selection_seed()` (SHA-256 stable derivation from `selection_seed_base`, scenario seed, episode index, strategy) unless explicitly overridden in config. Logged as realized seed.
- **V2X seed (`v2x_seed`)**: explicit per episode and per strategy, derived independently by `derive_v2x_seed()` (separate base/domain tag) unless overridden. Logged as realized seed. V2X stochastic effects are disabled in baseline config (`loss_prob=0`, `delay_steps=0`), but seed control remains explicit.

### Random-k / nearest-k behavior
- **Nearest-k** remains deterministic (no RNG consumption required); a selection seed is still derived/logged for completeness.
- **Random-k** must use an explicit `numpy.random.Generator` (existing selector contract in `grouping/random_sel.py`). The runner provides a reproducible per-episode selector stream by creating a **fresh env per episode** with `selection_seed=<episode-specific seed>` (`envs/carla_env.py:_configure_selection` initializes `self._selection_rng = np.random.default_rng(selection_seed)` at env construction). This initializes the episode selection stream without modifying selector semantics or the protected V2X channel.
- **V2X channel** reset behavior is unchanged (`V2XChannel.reset()` replays stored seed; protected implementation untouched). Runner supplies explicit `v2x_seed` per episode at env construction.

### Evidence (CARLA-free)
- `tests/test_milestone9_config.py`: deterministic derivation, explicit overrides, separation across scenario/selection/V2X/strategy/episode identities, pair/episode IDs.
- `tests/test_milestone9_reproducibility.py`: same seeded `np.random.default_rng` reproduces random-k selections; different seeds can differ when candidates permit; nearest-k deterministic/unchanged; legacy factory modes/backward-compatible behavior preserved; explicit `RandomSelector` does not consume global NumPy RNG.
- "Per-episode selection stream initialization" is validated at the RNG/selector level consistent with the runner design (fresh seeded generator per episode). End-to-end CARLA env-reset reseed behavior is **not CARLA-free verifiable** because `CarlaMultiAgentEnv` requires a live CARLA connection at construction.

### Limitation
- `envs/carla_env.py` does not currently reseed the selection RNG on `reset()` (pre-existing). M9A avoids relying on in-process reseed by using fresh env instances per episode. If a future approved change requires reusing one env across episodes, a minimal additive reseed hook should be introduced and tested.
- CARLA/TM/world determinism is not guaranteed; runner logs realized seeds/config and validates initial scenario metadata to detect mismatches.

## 4. Backward-compatibility evidence

- **Config/runner/logging/validator** are additive modules and do not alter existing runtime modules/contracts.
- **`envs/carla_env.py`**: new `max_steps` argument defaults to `None` → `MAX_STEPS` (existing behavior preserved); invalid values raise `ValueError` (explicit, no silent fallback).
- **Selectors/grouping**: existing API/behavior unchanged; tests confirm `nearest_k`/`random_k` selection behavior and legacy mode handling via `make_selector` remain consistent.
- **Controller**: `controllers.observation_heuristic.ObservationHeuristicController` used as-is with identical parameters for both strategies (`tests/test_milestone9_integration.py`).
- **Protected/contracted components untouched:** `envs/v2x_channel.py`, `tests/test_v2x_channel.py`, `docs/v2x_channel.md`, `envs/obs_builder.py`; no changes to 56-D obs/normalization, reward definitions, termination semantics, or MARL algorithms.
- **CARLA-free regression suite:** `350 passed, 8 skipped, 17 deselected` (`pytest tests -k "not carla" -q`).

## 5. Logging schema

Results are stored only under `results/milestone9/<experiment_id>/` with subdirectories:
`raw/`, `episodes/`, `summaries/`, `statistics/`, `figures/`, `tables/`, `metadata/`.
Existing experiment directories are never overwritten (`ExperimentLayout` refuses non-empty/existing manifests).

### Episode record (`episodes/<episode_id>.json`, compact; raw in `raw/<episode_id>_raw.json`)
- Identity: `experiment_id` (context), `episode_id`, `pair_id`, `strategy`, `episode_index`
- Seeds: `scenario_seed`, `selection_seed`, `v2x_seed`, `realized_seeds`
- Execution: `execution_status`, `failure_reason` (and traceback in raw), timestamps
- Config/controller: serialized `config`, controller identity/parameters, CARLA version/map, environment/spawn/V2X/horizon
- Initial scenario: `initial_scenario` with agents, per-agent initial position/orientation/speed, destination, route points, spawn/reset metadata
- Termination: `ending_reason` (derived), `success`, `collision`, `off_road`, `stuck`, `timeout`, `steps`
- Outcomes: `arrival_steps`, per-agent final info, `team_metrics` (team reward total/mean-step, min TTC, jerk when available, duration), `communication_metrics` (V2X episode stats when present), candidate source
- Artifacts: paths to episode JSON/steps CSV/raw JSON

### Per-step log (`episodes/<episode_id>_steps.csv`)
Per agent, per step: `step`, `agent_id`, `action`, `action_mask`, `selected_candidate_ids`, `candidate_pool_size`, `selected_count`, `missing_slots`, `valid_candidate_count`, `v2x_received_count`, `v2x_received_ids`, `v2x_received_ages`, `reward`, termination flags (`terminated/truncated/goal_reached/collision/off_road/stuck`), `stop_reason`, `arrival_step`, and compact `observation_summary` (speed, distance/progress, valid candidate count, selected relative candidate features). Complex fields serialized as JSON strings in CSV.

### Pair summary (`summaries/<pair_id>.json`) and experiment summary (`summaries/experiment_summary.json`)
- Pair: `pair_id`, scenario seed/episode index, strategy execution order, per-strategy episode status/artifacts, validation report, `valid_for_analysis`
- Experiment: config snapshot, requested pairs, pair results, counts of episodes and valid pairs

### Manifest (`metadata/manifest.json`)
- Experiment metadata (id, timestamps, git SHA, platform/python, status)
- Registered files with relative path, kind, size, SHA-256 checksum, registration time, and extra metadata
- Registered episode records and incremental updates

### Atomicity and integrity
- `atomic_write_text/json/csv` write to temp files in target directory then `os.replace`.
- Checksums recorded via SHA-256 per registered file; raw episode data preserved separately from summaries.

## 6. Pair-validation logic

`scripts/milestone9/validate_pair.py` compares two episode records and returns a structured report (`valid`, issues, checked fields). It **never silently marks a mismatched pair as valid**.

### Checked equivalence (initial/config)
- Pair identity: `pair_id`, `scenario_seed`, `episode_index`
- Config: town, vehicle count, max steps/horizon, spawn mode/max distance, candidate source, V2X config, controller identity/parameters
- Initial scenario: agent IDs/count/order, per-agent initial position, orientation (with cyclic yaw comparison), initial speed, destination, route points/length, spawn metadata (when present)

### Tolerances
- Configurable float tolerances (defaults): position `1e-3 m`, orientation `1e-2 deg`, speed `1e-3 m/s`, route point `1e-3 m`.

### Classification and handling
- **`initial_scenario_mismatch`**: identity/config/initial-state differences beyond tolerance; details include compared values and plausible causes (e.g. CARLA/TM nondeterminism, reset-order/infrastructure differences) without claiming proof.
- **`infrastructure_failure`**: episode execution not successful/complete or missing records.
- **Later trajectory divergence**: explicitly **not compared/required**; different actions may legitimately produce different trajectories.
- **CARLA nondeterminism**: recognized as a plausible cause for initial mismatches; cannot be proven from logs alone.
- Pair is valid for paired analysis only when no issues/failures are detected; failed members invalidate the pair for statistical pairing while raw data is retained.

## 7. Test results

### New M9A CARLA-free tests (run first)
Command:
```powershell
C:\Users\souro\venvs\v2x-carla312\Scripts\python.exe -m pytest tests/test_milestone9_config.py tests/test_milestone9_reproducibility.py tests/test_milestone9_logging.py tests/test_milestone9_validation.py tests/test_milestone9_integration.py -q
```
Result: **36 passed**.

Coverage (CARLA-free):
- Configuration validation and serializable round-trip
- Seed derivation (deterministic, separated domains, explicit overrides), pair/episode identity
- Random-k reproducibility (same seed), different-seed variability when candidates permit, per-episode fresh seeded stream initialization, nearest-k unchanged, selector backward compatibility/legacy modes, no global NumPy RNG consumption
- Atomic output, experiment layout/manifest generation, checksum validation, no-overwrite protection, ending-reason derivation
- Pair equivalence and mismatch detection, tolerances, never-silent mismatch, failed-episode/infrastructure handling, trajectory divergence not required
- Existing controller integration and deterministic/counterbalanced strategy ordering helpers

### Full CARLA-free regression suite
Command:
```powershell
C:\Users\souro\venvs\v2x-carla312\Scripts\python.exe -m pytest tests -k "not carla" -q
```
Result: **350 passed, 8 skipped, 17 deselected** in ~67.6s.

### Not run
- **CARLA-dependent tests/episodes/pilot:** not run (CARLA server not launched; no explicit authorization for pilot execution).
- No runtime behavior experiments executed.

## 8. Remaining limitations

- **CARLA nondeterminism/uncontrollable state:** exact scenario equivalence across runs is not guaranteed despite seeding/TM controls; validation detects initial mismatches but cannot eliminate TM/CARLA/world stochasticity.
- **Selection RNG lifecycle in env:** pre-existing `CarlaMultiAgentEnv` initializes selection RNG once at construction and does not reseed on `reset()`. M9A uses fresh env per episode to obtain explicit per-episode streams; no in-env reset reseed hook added (to avoid runtime behavior changes).
- **V2X seed/reset coupling:** protected `V2XChannel.reset()` replays constructor seed (unchanged). Baseline uses loss/delay 0, minimizing impact; explicit per-episode `v2x_seed` is passed at env construction.
- **Env-level `ending_reason`:** `envs/termination.py`/env info do not emit a canonical `ending_reason` string; runner/logging derive it from flags/steps (`derive_ending_reason`) without changing env semantics.
- **End-to-end CARLA verification untested:** actual pilot pair execution, CARLA cleanup behavior under failures, and live scenario equivalence remain unverified until an authorized CARLA pilot is run.
- **Metrics/analysis:** M9A provides logging/infrastructure only; no confirmatory statistical results. Five-seed runs remain exploratory per audit.
- **Episode-horizon configurability in env:** minimal `max_steps` init support added; runner enforces baseline 300 and relies on env termination/truncation. Any broader horizon study must keep values identical across paired modes and pre-registered.

## 9. Exact commands to run one pilot pair later

**Prerequisites (not executed here):** CARLA 0.9.16 running and reachable (Town03), venv `C:\Users\souro\venvs\v2x-carla312`, all relevant CARLA-free tests passing, and **explicit authorization** to run the pilot.

From repository root `C:\Users\souro\Desktop\V2X-MARL`:

```powershell
# 1) (Recommended) verify CARLA-free suite first
C:\Users\souro\venvs\v2x-carla312\Scripts\python.exe -m pytest tests -k "not carla" -q

# 2) Inspect resolved baseline config/plan without launching CARLA
C:\Users\souro\venvs\v2x-carla312\Scripts\python.exe scripts\milestone9\run_paired.py --dry-run-config

# 3) Run exactly one explicitly requested pilot pair (e.g. scenario seed 7, episode 0)
#    Counterbalanced strategy order by default; fresh env per episode; outputs under results/milestone9/<new_experiment_id>/
C:\Users\souro\venvs\v2x-carla312\Scripts\python.exe scripts\milestone9\run_paired.py --pilot --scenario-seed 7 --episode-index 0

# Optional: randomize (instead of counterbalance) strategy order with a fixed seed
C:\Users\souro\venvs\v2x-carla312\Scripts\python.exe scripts\milestone9\run_paired.py --pilot --scenario-seed 7 --episode-index 0 --randomize-order --strategy-order-seed 0
```

Notes:
- The runner refuses execution unless `--pilot` (or `--dry-run-config`) is provided.
- A new experiment ID/directory is created; existing result directories are not overwritten.
- Failed episode members invalidate the pair for paired analysis, but raw artifacts/manifests are retained for inspection.
- Do not claim statistical or performance conclusions from a single pilot pair.

## 10. Quality and safety checks

- **`git diff --stat`:** `envs/carla_env.py` and `grouping/__init__.py` show pre-existing uncommitted modifications (M9A added only the minimal `max_steps` handling in `envs/carla_env.py`; `grouping/__init__.py` not modified in this phase).
- **`git diff --cached --stat`:** empty (nothing staged).
- **Protected files unchanged:** no diffs for `envs/v2x_channel.py`, `tests/test_v2x_channel.py`, `docs/v2x_channel.md`, `envs/obs_builder.py`.
- **Previous results untouched:** `results/milestone7/` (8 files), `results/milestone8/` (4 files), `results/milestone8a/` (15 files) present.
- **No Git mutations:** no reset, clean, add/stage, commit, or push performed.
- **CARLA pilot:** not launched; no performance/statistical claims made.
