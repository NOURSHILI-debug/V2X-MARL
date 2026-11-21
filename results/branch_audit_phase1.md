# Phase 1 branch audit — `feat/phase1-core`

**Scope:** Read-only audit of the checked-out branch and existing evidence. No CARLA server was started and no packages were installed. The only write is this report. The permitted non-CARLA pytest command was run; pyflakes was attempted and found unavailable.

**Evidence convention:** line references refer to the current checkout unless explicitly described as Git/artifact evidence. Artifact provenance is **UNVERIFIED** where no embedded code revision/run metadata exists. User beliefs were treated as hypotheses and verified against files/commands where possible.

## 1. Branch state

- `git status`: branch `feat/phase1-core`, tracking `origin/feat/phase1-core`, up to date. Two untracked files: `results/check_obs.txt`, `results/check_policies.txt`. `git diff --stat` is empty; no modified tracked files. `git status --short` output: `?? results/check_obs.txt`, `?? results/check_policies.txt`.
- `git log --oneline --graph -n 30`: branch tip `d1497b1 feat: implement control and observation builders`; preceding commits include `fbdff63`, `3d8a8e2`, `5c4a9f5`, `8132f06`, `93bf9f0`, `ca587fd`, `61513d3`, `d703462`, then merges and earlier work. The displayed history contains 21 commits (fewer than 30 available).
- `git rev-list --left-right --count origin/main...HEAD` → `0 12`: 12 commits ahead of the locally known `origin/main`, zero behind. `git rev-list --left-right --count origin/feat/phase1-core...HEAD` → `0 0`: matches the local remote-tracking ref. Remote freshness beyond local refs is UNVERIFIED; no fetch was done.
- Root untracked-looking clutter: `desktop.ini` (54 bytes) is present at root and not listed among tracked files; `t tests -q -k not carla and not highway` (866 bytes) is also a root file, likely accidental command-output debris. `README.md` and `requirements.txt` are zero bytes. Confirmed through `Get-ChildItem -Force -File`; `git status` did not list desktop.ini or README/requirements changes, likely ignore/global-ignore behavior. Do not delete without owner direction.
- `.gitignore` contains `.venv/`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.coverage`, `results/raw/*` except `.gitkeep`, and `*.pt`. Thus `.pt` is ignored globally; `results/` is not globally ignored, only `results/raw/` contents are. No explicit image/log ignore rule.
- Largest tracked blobs from `git ls-tree -rl HEAD`: `results/vdn_toy_curve.png` 134,850 B; `results/vdn_toy_log.csv` 69,412 B; `results/repo_audit.md` 34,027 B; `envs/carla_env.py` 21,781 B; none is a large Git blob by ordinary repository standards. `results/vdn_toy.pt` (73,480 B) exists locally but is ignored, not in tracked tree.

## 2. Work inventory (Person A)

Attribution is based on branch history/commit subjects and the ownership split supplied in the request. `git diff origin/main...HEAD --name-status` shows these branch changes; “authored” is an attribution estimate, not cryptographic proof. Line counts are current checkout counts.

### Env code

| File | Purpose / status / lines |
|---|---|
| `envs/carla_env.py` | CARLA adapter, routes, control, observations and shared termination; changed on branch, committed, 510 lines. |
| `envs/base.py` | Shared multi-agent API/actions/required info constants; added, committed, 65 lines. |
| `envs/control.py` | CARLA-independent target-speed calculations; added in `d1497b1`, committed, 30 lines. |
| `envs/termination.py` | Pure termination decision and info-building helpers; added, committed, 111 lines. |
| `envs/obs_builder.py` | Pure NumPy 56-D layout, mask and normalization; added in `d1497b1`, committed, 166 lines. |
| `envs/reward.py` | Reward calculator exists but is teammate-authored/shared and not modified in this branch diff; 140 lines. |
| `envs/safety.py` | Collision sensor helper exists; shared/teammate history, not modified in branch diff; 125 lines. |

### Tests

| File | Purpose / status / lines |
|---|---|
| `tests/test_contract.py` | Contract tests, added, committed, 135 lines. |
| `tests/test_action_dispatch.py` | Action/control dispatch pure test, added, committed, 104 lines. |
| `tests/test_control.py` | Speed arithmetic, added, committed, 22 lines. |
| `tests/test_obs_builder.py` | Layout, clipping, frame and mask tests, added, committed, 163 lines. |
| `tests/test_termination_logic.py` | Termination priority/counters, added, committed, 99 lines. |
| `tests/test_env_info.py` | Off-road/info helper tests, added, committed, 63 lines. |
| `tests/test_collision_monitor.py` | Fake-sensor unit tests, shared/teammate file, unchanged in branch diff, 85 lines. |
| `tests/test_reward.py` | Reward-unit tests, shared/teammate file, unchanged in branch diff, 58 lines. |
| `tests/test_toy_env.py` | Toy tests, teammate-owned, unchanged in branch diff, 70 lines. |
| `tests/test_grouping.py` | Empty placeholder, teammate-owned, unchanged, 0 lines. |

### Scripts and docs

| File | Purpose / status / lines |
|---|---|
| `scripts/check_obs.py` | CARLA observation sampling/range diagnostic, added, committed, 127 lines. |
| `scripts/check_policies.py` | 20-seed simple policy comparison, added, committed, 102 lines. |
| `scripts/test_termination.py` | CARLA manual termination integration script, added, committed, 124 lines. |
| `scripts/test_env.py` | CARLA smoke script, modified on branch, committed, 31 lines. |
| `docs/env_spec.md` | Proposal plus Contract v1; changed, committed, 117 physical lines (formatting/encoding causes wrapped bold contract content). |
| `docs/env_spec_section12.md` | Contract extract, added, committed, 27 lines. |
| `pytest.ini` | Pytest configuration, added, committed, 2 lines. |

### Results evidence

Branch diff added `results/checkpoint_A.md` (committed, 224 lines), `results/repo_audit.md` (committed, 516 lines), `results/test_termination.txt` (committed, 10 lines), `results/train_log2.txt` (committed, 39 lines), `results/train_log3.txt` (committed, 6 lines). `results/check_obs.txt` (122 lines) and `results/check_policies.txt` (14 lines) are currently untracked local evidence, so not authored/committed as branch content. `results/train_log.txt`, `results/test_env.txt`, toy CSV/PNG are pre-existing tracked artifacts, not added in this branch diff.

## 3. Technical read of CARLA adapter

Evidence: `envs/carla_env.py` lines 38–58 (constants), 63–106 (initialization/sync), 109–137 (cleanup), 140–159 (route), 162–231 (reset/spawn), 233–298 (step), 300–342 (goal/control), 344–408 (mask/candidates), 410–510 (features/state/masks); `envs/termination.py:1–111`; `envs/obs_builder.py:1–166`.

- **Spawn/setup:** client timeout 30 s; target town loaded if necessary; world sync with `fixed_delta_seconds=.05`, TM sync, TM random seed; shuffled map spawn points with local seeded RNG; Tesla Model 3 vehicles spawned until N; autopilot enabled; auto lane changes disabled; lights respected. Four agents default, configurable N. Five settling ticks, then collision sensors. Route is seeded random choice among `Waypoint.next(5 m)` until nominal 150 m, then TM `set_path` receives sparse waypoints and final point. `route_applied` records success, but no failure causes reset rejection.
- **Tick cadence:** four `world.tick()` calls per `step`, then step count increments: 20 Hz simulation / 5 Hz action as intended. Initial settling and teardown ticks also occur outside decisions.
- **All six actions:** KEEP=0 leaves speed target unchanged and re-applies it; ACCEL=1 adds 2 m/s; DECEL=2 subtracts 2 m/s; BRAKE=3 sets TM percentage difference 100% and returns without changing stored target; LANE_LEFT=4 calls `force_lane_change(v, False)`, right=5 uses `True`, then both reapply current target. Speed arithmetic clamps [0, V_MAX], `V_MAX=30`. Lane action masks are exposed but `_apply_action` does not consult them, so direct invalid lane actions still issue TM commands. BRAKE’s next non-brake action restores stored target; its duration is one decision interval, but it is not an explicit vehicle brake command.
- **Collision:** one collision sensor per controlled vehicle (`envs/safety.py:18–60`); callback flags collision and stores event frame/actor id. Sensor attached after settling, destroyed before vehicles.
- **Termination order:** collision > all goals > off-road (10 consecutive decisions) > stuck (>50 slow decisions after 25-step grace) > timeout (300). Off-road is therefore 0.5 s continuous, despite user-specified report requirement “off-road or stuck > 10 s”; Contract §12.2 specifies stuck and timeout, and §7 says off-road or stuck >10 s, so this is stricter and needs agreement. Stuck is actually >50 count and grace gates termination (counter reset during grace); non-traffic-light speed <.5 m/s. All-agent flags broadcast. `finished` marks local/global terminal actors, and the state slot is zero-filled afterward.
- **Goal/route caveats:** goal uses Euclidean distance <6 m to last route waypoint; route length may be shorter than 150 m if `next()` ends. `route_applied=False` does not stop evaluation. Goal handling sets TM speed difference 100, marks reached, ignores subsequent actions, but does not disable autopilot/destroy/freeze physics explicitly. “Frozen” may be approximate, not verified. `scenario_id` is not implemented.
- **Candidates:** `get_candidates` directly enumerates all `vehicle.*` actors in world within horizontal XY range 50 m, including other CAVs/background vehicles, sorts distance/id. It is raw simulator state, not V2X messages; no channel, delay or loss is used. It returns all neighbors then builder truncates to six.
- **Observation/masks:** declared 56-D; builder normalizes/clips to [-1,1], replacing non-finite input. CARLA adapter imports `carla` at module top. Route x/y supplied are world coordinates but builder rotates route points relative to ego (check builder implementation); ego block’s 8 inputs are speed, longitudinal acceleration, heading error, lateral offset, lane index/count, goal distance, progress fraction. Masks use same-road, same-sign lane ID plus Driving lane type; action masks recompute live lanes. Lane context and masks are in adapter. Traffic-light state is red/yellow/green/none. Heading/y-frame sign convention has unit math tests but no CARLA-left-handed integration test.
- **State:** flattened `(n_agents,6)`: world x/y, vx/vy, yaw radians, goal distance. Contract only specifies global state, no exact dimension, so shape is acceptable.
- **Cleanup:** close disables world and TM sync then destroys sensors/vehicles and ticks. Reset calls stop-sync/destroy then re-enables sync. Nested `finally` attempts cleanup on errors; failure before `self.tm` initialization (constructor setup exception) is not specially guarded. Calling close twice is not demonstrably safe because teardown ticks a world outside sync; no CARLA repeat-close test is present.
- **Stubs/duplication/magic:** reward is literal zeros (`carla_env.py:282`); `min_ttc`, `jerk`, `team_reward`, scenario and actual per-term reward absent. Speed/route/stuck values are in CARLA module while observation scales sit centrally in `obs_builder.py`; `M_CANDIDATES` duplicated in CARLA/base/obs modules. `lane_index`/`lane_count` returned as lane offset/count and scaled in builder, not actual raw CARLA lane id. `MAX_STEPS` is 300 but smoke script may override shorter episodes. `speed_limit_difference` can be negative if target exceeds limit, accepted by TM but not tested against simulator.
- **CARLA import boundary:** `envs/carla_env.py` imports CARLA appropriately as adapter. `envs/safety.py` dynamically imports CARLA only inside `_spawn_sensor` (a CARLA-facing helper); `envs/base.py`, `control.py`, `termination.py`, `obs_builder.py`, `reward.py` are CARLA-independent. `rg` audit should be rerun to prove entire repository boundary; current focused source check found no CARLA import in generic helpers.
- **TODOs:** `carla_env.py:9–10` explicitly lists reward/min TTC/jerk (T1.3); scenario ids and V2X are absent; metrics/baseline runner and CSV absent.

## 4. Contract compliance: CARLA vs §12

| Contract item | Status | Evidence |
|---|---|---|
| API tuple/dicts, attrs and methods | **Pass (static)** | `carla_env.py:63–74,162,233,498`; `MultiAgentEnv` declaration `base.py:37–65`. Runtime API artifact `api_carla.txt` shows reset tuple, `(56,)`, mask `(6,)`, five-return step/dicts, state `(10,)` for its two-agent run. |
| Shared collision termination | **Pass (code + historical server artifact)** | `termination.py:54–80`; `results/test_termination.txt` says forced collision terminates all agents, ALL OK. Provenance relative to current head is not recorded, so runtime confirmation of exact latest revision is UNVERIFIED. |
| All goals / timeout / stuck | **Partial** | Logic exists, timeout 300, stuck >50 plus 25-step grace (`carla_env.py:51–53,258–269`, termination.py). Test artifact reports goal/stuck/timeout; does not explicitly prove 300 steps because script uses short timeout. Off-road behavior adds termination after 10 steps. |
| Step info keys | **Partial** | `collision`, `goal_reached`, `steps` always (`termination.py:72–81`). Adds `stuck`, `off_road`, `route_applied`, `per_agent`. |
| End info | **Fail** | `termination.py:82–83` adds `success`, `town`, `seed`, but not required `min_ttc`, `jerk`, or `scenario_id`; no `team_reward`. `envs/base.py:27` requires `min_ttc`,`jerk`. |
| Reward formula/team reward | **Fail** | CARLA gives all-zero rewards (`carla_env.py:282`); team reward not reported; RewardCalculator defaults independently mismatch (§6, see §10). |
| CARLA obs 56-D | **Pass (static/unit)** | `obs_builder.py:15–22`, `carla_env.py:71`, tests and `check_obs` artifact. Feature semantics/scaling still need empirical acceptance (see §7). |
| Reproducibility | **Partial** | local RNG and TM seed (`carla_env.py:176–177`) control spawn shuffle/routes. Stable route/spawn code, but background traffic isn’t configured here; exact repeatability and different-seed difference require live paired check. |
| Action mask honored | **Fail** | Mask returned (`carla_env.py:487–509`) but `_apply_action` has no mask guard (`326–342`). |

## 5. Plan conformance

| Deliverable | Status | Exact completion criterion | Evidence / missing |
|---|---|---|---|
| Phase 0 contract/base/spec | **Partial** | contract passes on toy and CARLA | Code exists; current permitted test selection passed, but includes no strict CARLA live run. CARLA static info fails end metrics; historical `results/checkpoint_A.md` reports toy strict gaps but predates current branch. Current strict toy run not run here. |
| T1.1 collision/termination | **Partial** | forced collision ends episode with collision=True | `results/test_termination.txt`: forced collision/shared ends, stuck or timeout, short-route goal, 10 resets ALL OK. It is committed in `fbdff63`/3d8a8e2 era; new stuck grace/off-road/sync edits are not proven server-side at current head. |
| T1.2 observation/masks | **Partial** | `(56,)`, zero NaN over 100 episodes, lane changes masked without adjacent lane | Builder unit tests and 16 partial episodes (`check_obs.txt`) show vector diagnostics; no 100 episodes, artifact is 16 episodes (15 timeout + episode 16 stuck), and no live assertion that masked commands are rejected. Current output provenance pre latest speed fix is supported by low speeds and comment in request, but exact commit is UNVERIFIED. |
| T1.3 reward integration | **Not started** | logged per-term rewards, scripted “drive on” beats “crash” | `carla_env.py:282` TODO zero reward; no reward terms/info/team reward, no baseline comparison. `RewardCalculator` not used in adapter. |
| T1.4 CSV + random/scripted baseline | **Partial** | episode CSV and baselines report success/collision/travel time/jerk | `scripts/check_policies.py` has keep/accelerate/random-valid but interrupted in `world.tick` per output; no completed paired baseline or metrics CSV. `scripts/evaluate.py` exists but not shown to implement this (inspect before claiming). |
| T1.5 SOTA rows | **Not started** | VDN/QMIX/MAPPO rows checked against papers | `docs/sota_A.md` and `docs/sota_B.md` exist, but request names `docs/sota_rows_1.md`; that exact file is absent from `rg --files docs`. Literature verification not performed in this audit. |

## 6. Results audit

Artifact files have no embedded commit IDs. Production version is UNVERIFIED except inferences from their content and request-provided run notes. Validity here means evidence quality for the stated claim, not endorsing performance.

| Artifact | What it shows | Version relation | Judgment |
|---|---|---|---|
| `results/api_carla.txt` | Historical API shape, 2 agents, obs 56, mask 6, state 10; Town03 cache messages | Unknown; API shape agrees with current static code | **STALE** for current runtime behavior (no revision); limited historical API smoke evidence only. |
| `results/api_toy.txt` | Toy reset/step shape, obs 4, mask 3; records scalar terminated/truncated and state 6 | Historical | **STALE**; conflicts with contract’s dict requirement as currently printed. |
| `results/test_env.txt` | 100 ticks, 10 resets leave two vehicles, five episodes end at 50 steps; “ALL OK” | CARLA smoke, no revision; likely older | **STALE** as current validation. |
| `results/test_termination.txt` | Forced collision shared termination; stuck/timeout; short-route goal; 10 resets cleanup; ALL OK | Evidence from earlier pushed T1.1 commit; no exact executed hash. New speed/grace/reset code not shown validated | **VALID** only for the narrow historical T1.1 scenarios; **STALE** for current HEAD/full contract. |
| `results/check_obs.txt` | 16 episodes; first 15 time out at 100 steps, episode 16 stuck at 84; mean speed roughly .46–1.10 m/s in visible records; masks include invalid lane action fractions | Low speed indicates pre-speed-fix behavior; exact revision absent | **STALE** for latest speed fix; valid diagnostic that this recorded controller did not progress adequately. Not 100 episodes, not success evidence. |
| `results/check_policies.txt` | Interrupted traceback at `env.step` → `world.tick()` with KeyboardInterrupt | Current local script’s attempt, no completed run | **INVALID** as policy comparison/performance result; valid evidence of non-completion/hang/slow tick in that attempt. |
| `results/train_log.txt` | Toy VDN training log ending in Torch stack and KeyboardInterrupt | Older attempt | **INVALID** as completed training; interruption evidence only. |
| `results/train_log2.txt` | Another VDN Torch stack then `^C` | Older attempt | **INVALID** as completed training. |
| `results/train_log3.txt` | 100 toy episodes logged, then `^C` | Later/repair attempt inferred, exact code version absent | **INVALID** as completed/converged training. |
| `results/vdn_toy_log.csv` | 2,000 episode rows of returns/epsilon; values are highly variable with many near -91 and positive bursts | No revision/seed metadata; toy configuration may predate current branch | **STALE** for reproducible claim; rows alone do not show completion mechanism or held-out performance. |
| `results/vdn_toy_curve.png` | Plot corresponding to toy run | No code/version metadata | **STALE**. |
| `results/vdn_toy.pt` | Ignored local checkpoint, 73 KB; no metadata inspected | Unknown | **UNVERIFIED**; not tracked and not claimable as evidence. |
| `results/repo_audit.md` | Earlier audit on different branch/date, historical assertions | Explicitly reports `feat/env-and-vdn` / older tip | **STALE** for current branch; useful history only. |
| `results/checkpoint_A.md` | Handoff audit from 2026-11-05 on old branch `feat/env-and-vdn` at `93bf9f0` | Explicitly stale branch and earlier code | **STALE**. |

No other `.txt/.csv/.json/.png` were found under `results/`. Markdown audit/checkpoint reports are above; binary `.pt` noted. Existing `results/check_obs.txt` and `check_policies.txt` are untracked local outputs.

## 7. Fine-tuning / enhancement gate

| Component | Gate before next task | Evidence and measurable acceptance |
|---|---|---|
| Speed/controller behavior | **Fix/validate before reward experiments.** | Recorded pre-fix mean speed .46–1.10 m/s, no goal in 16 episodes (`check_obs.txt`). Current target 9 m/s, +/−2, V_MAX 30 code (`carla_env.py:54–55,315–342`; `obs_builder.py:18`) is not server-validated. Run at least 20 fixed seeds, keep policy mean speed 7–10 m/s on moving episodes and reach all goals in a clear-road scripted run; explain congestion separately. |
| Stuck rule | **Confirm semantics and test.** | Current grace 25 + `>50` (post-grace means actual earliest >75 decisions) and threshold .5 m/s (`carla_env.py:51–53`, `termination.py:49–50`). That is ~15 s total from reset in worst case, vs section 7 says >10 s. Choose/specify whether grace counts in the 10 s; then acceptance at exact boundary tests. |
| VDN toy | **Not a Phase 1 gate for CARLA, but do not claim trained baseline.** | Logs interrupted; no held-out evaluation. Require successful process completion, deterministic saved config/checkpoint, and a separate fixed-seed evaluation with mean return better than random by confidence interval / predeclared threshold. |
| Reward weights | **Must fix before T1.3.** | `RewardCalculator.DEFAULT_WEIGHTS`: collision 100, safety 15 / 2.5 s, comfort accel/jerk 1 over thresholds 2, goal 50; no idle. Differs from frozen contract (`reward.py:24–35`; `docs/env_spec.md` §12.4). Acceptance: explicit per-term unit tests exactly match +1, -10, TTC formula -0.5*(1-TTC/3) under 3 s, comfort -0.05 signals, +10 one-time goal, idle -0.1. |
| Masks | **Fix before policies.** | Current masks mark adjacent lanes, but env executes supplied lane action regardless (`carla_env.py:326–342`). Threshold: all invalid lane action commands are ignored/rejected and no forced lane-change API call occurs; test all 6 mapping paths. |
| Observation scales | **Review before training.** | `V_MAX=30`, `A_MAX=8`, `D_MAX=150`, lane width 6, max lanes 6, route-frame 20, candidate range 50 (`obs_builder.py:18–24`). `D_MAX=150` equals nominal route length; near route start, goal distance can exceed 150 and saturate. `ROUTE_FRAME_MAX=20` with next waypoint 5 m usually safe. V_MAX 30 m/s can saturate candidate relative velocities above 30; A_MAX 8 may saturate harsh braking; no distributions captured by check_obs excerpt. Require per-feature fraction clipped to ±1 <1% for ordinary driving, except masks/valid bits; adjust scales or document outliers. |
| Candidate range | **Clarify information model before selection research.** | `CANDIDATE_RANGE=50` and candidates are simulator actors, no V2X channel (`carla_env.py:376–408`). Contract says generic candidates; research spec says received V2X messages. Require controlled candidate list match simulated range/delay/loss and deterministic ID order. |
| Episode length/timeouts | **Fix script/spec mismatch.** | env 300-step / 60 s; check_obs first 15 episodes finish 100 steps, likely script short diagnostic cap; provenance and exact short cap absent. Require policy evaluation to run to true 300 timeout or terminal event and log actual sim duration. |

## 8. Tests

The exact permitted command was run in the active Python 3.13 interpreter:

```powershell
python -m pytest tests -q -k 'not carla and not highway'
```

Result: **52 passed, 4 skipped, 24 deselected** in 1.94 s. Two `PytestCacheWarning`s reported permission denial writing `.pytest_cache`; they did not fail the tests. This is not the requested Python 3.12 CARLA venv. No packages installed and no server tests run.

`python -m pyflakes envs scripts tests marl` → `No module named pyflakes` under Python 3.13. Do not infer clean lint.

| Test file | Coverage | CARLA server? |
|---|---|---|
| `tests/test_contract.py` | Generic API/obs/dict/termination; opt-in strict attrs, seeding and info; parametric env factories | CARLA-marked case needs server; toy/highway can run without CARLA. |
| `tests/test_action_dispatch.py` | CARLA action dispatch via fakes, including mapping behavior | No live server; no real TM behavior. |
| `tests/test_control.py` | Target-speed arithmetic and clamping | No. |
| `tests/test_obs_builder.py` | 56-D shape, finite/clipped features, ego-frame math, masks | No. Does not validate CARLA’s actual left-handed coordinate/sign convention. |
| `tests/test_termination_logic.py` | Pure priority, boundaries, counters, grace | No. |
| `tests/test_env_info.py` | `None` waypoint/off-road and nested info helper | No. Does not validate actual CARLA lookup. |
| `tests/test_collision_monitor.py` | Fake world/sensor callbacks, flags and cleanup | No. Does not prove actual sensor event delivery. |
| `tests/test_reward.py` | Calculator behavior under current defaults | No; currently asserts stale -100/+50 expectations and does not enforce spec. |
| `tests/test_toy_env.py` | Toy reset, movement, invalid action, state, toy collision | No. |
| `tests/test_grouping.py` | Empty file | No coverage. |
| `scripts/test_termination.py` | Manual live collision/stuck/goal/reset/cleanup scenarios | Yes, server required; artifact exists but post-current-head rerun not proven. |
| `scripts/check_obs.py` | 100-episode CARLA obs diagnostics (as configured in script), speed and masks | Yes; output artifact only 16 episodes, so did not complete its configured run. |
| `scripts/check_policies.py` | 20 seeds × keep/accelerate/random-valid, records terminal counts/speeds | Yes; output shows interruption at `world.tick()`. |
| `scripts/test_env.py` | Manual CARLA tick/reset/action smoke | Yes. |

Coverage gaps: no live all-six-action CARLA mapping test; no live assertion that invalid masks suppress lane-change execution; no ego-frame sign test against actual CARLA orientation; no explicit route_applied failure test; no server test for seed same/different starts/routes/goals; no test for 56-D real observation across 100 completed episodes; no reward integration/per-term/team reward test; no metric CSV test; no 300-step timeout live test in current artifacts.

## 9. Code quality

- Pyflakes result is unavailable, so undefined/unused import report is **UNVERIFIED**. `python -m pyflakes envs scripts tests marl` could not import module. The permitted pytest run found no failures.
- Static notable issue: `scripts/check_policies.py` output shows interruption inside `world.tick`, but does not distinguish deadlock from slow synchronous server, and script runs 60 episodes total; log lacks progress timing/heartbeat per tick.
- Names match singular `collision`, `min_ttc` desired, but `min_ttc` is missing; current keys include extra `stuck`, `off_road`, `route_applied`, `per_agent`. `scenario_id` is missing. Reward helper uses `safety` and `comfort_accel` terms rather than spec’s TTC/comfort aggregate and has different weights.
- Typing is sparse in CARLA adapter method signatures (`__init__`, reset, step, helpers); `obs_builder`, control, termination have more explicit typing. Several public/helper methods are undocumented (`step`, reset details, `get_candidates` only documented). `state()` is documented.
- Refactor before Phase 2: move all env constants into one spec/config module; separate route planner, CARLA lifecycle, reward/metrics and observation extraction; avoid duplicate candidate/action dimensions. Add explicit info schema typed object or constants. Do this after behavior gate so refactor does not obscure validation.
- README, requirements are empty; current user docs cannot be runbook. `docs/env_spec.md` contract is encoded/rendered with mojibake in terminal (`4Ã—`, `âˆ’`); verify file encoding/editor display and remove duplicate `docs/env_spec_section12.md` drift risk. Results contain old branch audits presented beside current outputs; clearly label archival evidence.
- Code-level concern: action dictionaries are not validated for missing/extra agents and invalid action IDs: current `step` iterates the supplied mapping; unknown IDs can KeyError and omitted IDs are silently uncontrolled for a decision. Add contract validation.

## 10. Cross-team impact

- **B needs from A:** stable CARLA `obs_dim=56`, feature ordering/scaling, action ids/mask semantics, candidate row meaning/ordering, global state shape, consistent info schema, and one frozen reward formula with episode/team fields. Current adapter satisfies shape/action numbering but mask is advisory only, candidate source is raw world state without V2X delay/loss, and info/rewards lack fields.
- **A needs from B:** RewardCalculator defaults and API aligned to Contract §12.4, with per-term outputs if “logged per-term rewards” is required; teammate clarification for candidate/V2X boundary; Toy/Highway strict API/obs key convergence; any VDN/QMIX expected `team_reward` and terminal info consumers.
- **Interface mismatches:** `base.py` requires `success,min_ttc,jerk`; CARLA only returns success/town/seed at terminal. `RewardCalculator` uses collision 100, goal 50 and TTC 15/2.5s, lacks idle. Toy/Highway mismatch details in old `checkpoint_A.md` are historical, not re-audited strictly here. Current selected pytest excludes highway and CARLA; strict toy status therefore **UNVERIFIED this run**.
- **T1.3 blocked:** CARLA has no reward integration, all rewards fixed 0 (`carla_env.py:282`); team reward absent; min TTC and jerk cannot be logged; RewardCalculator contract mismatch means integration now would bake incorrect weights. Need agreed reward API + metric extraction first.

## 11. Ranked risks (top 10)

| Rank | Severity | Risk | Fix |
|---:|---|---|---|
| 1 | Critical | CARLA training signal is all zero; T1.3 not begun. | Implement and unit-test exact §12.4 terms and team reward; log term breakdown. |
| 2 | High | Speed/controller behavior only has pre-fix low-speed artifact; new speed fix has no current server evidence. | Run paired 20-seed controller validation on server with timing/progress and goal rates. |
| 3 | High | Invalid lane-change actions are executed despite masks. | Enforce mask in `step`/`_apply_action`, choose reject-or-no-op semantics, test all cases. |
| 4 | High | Required terminal metrics/info absent; strict CARLA contract fails. | Populate `min_ttc`, `jerk`, `scenario_id` and team reward before end-of-episode contract gate. |
| 5 | High | 20-seed policy checker did not complete; synchronous tick may be stalled or just slow. | Diagnose server/tick timeout and add episode/tick heartbeat, then rerun to completion. |
| 6 | High | V2X grouping candidates bypass channel delay/loss and use simulator ground truth. | Define adapter boundary and feed candidates from channel before selection experiments. |
| 7 | Medium-high | Goals/routes may not be honored by TM; route application failure only logged, not enforced. | Make route failure explicit; live-test route_applied and end-goal behavior on fixed seeds. |
| 8 | Medium-high | Stuck grace changes effective threshold from >10 s to at least ~15 s, unclear vs spec; off-road ends after only 0.5 s. | Agree exact semantics; align constants and test boundary transitions. |
| 9 | Medium | Observation normalization may saturate goal distance and extreme relative speeds/accels. | Log per-feature clipping fractions over representative runs; acceptance <1% routine saturation. |
| 10 | Medium | Artifacts lack code revision/run metadata; stale logs risk slide claims. | Every runner records git SHA, config, seeds, CARLA version, date, complete/incomplete status. |

## 12. Recommended next actions in strict order

### (a) Validation before commit

1. In the CARLA 0.9.16 / Python 3.12 venv, start/use an already-running server and run `python scripts/test_termination.py`; **done when** collision, all-goal, timeout, stuck, off-road, repeated reset and cleanup each pass at current `git rev-parse HEAD` (script currently must be checked for timeout/off-road assertions).
2. Run `python scripts/check_obs.py`; **done when** it completes 100 full episodes, records revision and per-feature clip fractions, has no NaN, and lane masks match actual no-adjacent-lane state.
3. Fix invalid-mask dispatch, then run focused pure tests `python -m pytest tests/test_action_dispatch.py tests/test_control.py tests/test_obs_builder.py tests/test_termination_logic.py -q`; **done when** unavailable lane changes are ignored/rejected and existing pure tests pass.
4. Run strict integration ` $env:CONTRACT_STRICT='1'; python -m pytest tests/test_contract.py -q -k carla` in PowerShell, CARLA venv/server; **done when** all required end info keys and shared termination pass on current code.
5. Run `python scripts/check_policies.py`; **done when** all 20 seeds for each policy complete and means/counts are emitted with no KeyboardInterrupt, using same paired seed set.
6. Run permitted suite in CARLA venv: `python -m pytest tests -q -k 'not carla and not highway'`; **done when** green without cache permission warnings or warnings are documented.
7. If pyflakes is already available, `python -m pyflakes envs scripts tests marl`; **done when** no diagnostics. Do not install solely for this audit.

### (b) Commit/push plan

1. Review `git status --short` and `git diff --check`; decide whether local `check_obs.txt`/`check_policies.txt` belong in results; **done when** only intended code/evidence staged.
2. Commit implementation separately, suggested message: `fix: enforce CARLA action masks and controller semantics`; **done when** focused tests pass and commit contains only source/tests.
3. After successful current-head server validation, commit its metadata-stamped logs separately: `test: record Phase 1 CARLA validation`; **done when** logs include SHA/config/seed and completed status.
4. Push `feat/phase1-core`; **done when** `git rev-list --left-right --count origin/feat/phase1-core...HEAD` is `0 0` and the remote commit is visible. No push performed in this audit.

### (c) T1.3 prerequisites

1. Align `envs/reward.py` defaults and semantics with frozen §12.4 and agree whether lateral acceleration or longitudinal acceleration is comfort input; **done when** exact per-term unit tests match spec.
2. Add CARLA measurements for route progress delta, TTC minimum and jerk; **done when** deterministic fixtures prove units/signs and terminal info includes all mandated keys.
3. Wire per-agent calculator, one-time goal, freeze-after-goal, idle exception for red/yield, `team_reward=mean(r_i)` plus collision penalty; **done when** no zero placeholder remains and `scripted drive-on` beats `crash` across paired seeds.

## 13. Presentation-ready evidence pack

Only narrow **VALID** evidence is listed; historical runtime provenance still limits claiming it proves current HEAD.

| Claim | Number | Source | Validity |
|---|---:|---|---|
| Current branch matches its local remote-tracking ref | 0 ahead / 0 behind | `git rev-list --left-right --count origin/feat/phase1-core...HEAD` | VALID for local refs at audit time |
| Current branch is ahead of locally known main | 12 commits | `git rev-list --left-right --count origin/main...HEAD` | VALID for local refs at audit time |
| Pure selected suite passes | 52 passed, 4 skipped, 24 deselected | pytest command above | VALID (Python 3.13 runtime; non-CARLA selection only) |
| Historical forced-collision shared termination passed | 1 reported scenario | `results/test_termination.txt` | VALID as historical narrow evidence, not latest-HEAD certification |
| CARLA observation layout declared | 56 floats | `envs/obs_builder.py:15–22` and shape tests | VALID as implementation/tested builder shape, not live 100-episode claim |

Do **not** put speed, goal-success, policy ranking, converged VDN, reward, or robustness claims on results slides: current evidence does not establish them.

Suggested slide sequence:

1. **Question and fixed contract** — §12 API/56-D/team termination; diagram/table from spec.
2. **Environment architecture** — CARLA adapter plus pure observation/control/termination helpers; source-level architecture only.
3. **Meta-action controller** — six actions and 5 Hz / 20 Hz code constants; label as implementation, pending live validation.
4. **Observation and action-mask layout** — 8+6+36+6=56; shape tests support builder only.
5. **Termination evidence** — historical forced collision/all-agent ending and reset cleanup (`test_termination.txt`), date/revision unknown.
6. **Software check status** — 52 pass / 4 skip / 24 deselect in Python 3.13.
7. **Controller diagnostic** — show low-speed historical `check_obs` only as a failure diagnosis, explicitly pre-speed-fix/stale; avoid performance implication.
8. **Reward and metrics gap** — zero reward placeholder and absent TTC/jerk/team reward; source code evidence.
9. **Next validation gates** — speed, masks, 100 episodes, paired baselines.
10. **What is not done / risks** — no current server validation of speed fix, no T1.3 reward, no completed policy run, no valid VDN convergence claim.

## 14. Handoff checkpoint skeleton for colleague

**Done**
- Contract/base and CARLA 56-D helper implementation are present on `feat/phase1-core`; selected pure test run: 52 passed, 4 skipped, 24 deselected.
- Historical T1.1 server log reports forced collision shared termination, stuck/timeout, goal and cleanup scenarios.

**In progress**
- Current controller values and stuck grace implementation are committed but require live server validation.
- Observation script exists; only 16 episodes are captured in current untracked output, not its stated 100-episode run.

**Not started**
- CARLA reward integration, TTC/jerk/team reward and per-term logging; T1.4 complete CSV baselines; T1.5 requested `sota_rows_1.md` verification.

**Known limitations**
- `step` reward is 0; terminal metrics missing; mask is not enforced; candidates are direct world actor queries without V2X delay/loss; current policy output interrupted at `world.tick`; audit ran under Python 3.13, not CARLA venv Python 3.12.

**What I need from B**
- Frozen reward API/defaults aligned to §12.4; candidate/V2X interface decision; Toy/Highway strict contract status and expected consumers of info/team reward.

**How to run**
- Pure tests: `python -m pytest tests -q -k 'not carla and not highway'`.
- Live CARLA checks require CARLA 0.9.16 already running and Python 3.12 venv: `python scripts/test_termination.py`, `python scripts/check_obs.py`, `python scripts/check_policies.py`.
- Record `git rev-parse HEAD`, CARLA/Python versions, seeds, completion status with each result. This audit did not start CARLA or install packages.

## 15. Questions the owner must answer

1. Does the 25-decision stuck grace count in the spec’s 10-second stuck threshold, or should termination be exactly after 50 continuous slow decisions from reset?
2. Should off-road termination be immediate/short (current 10 decisions) or share the 10-second grace/threshold stated in §7?
3. Is CARLA allowed to use raw world actor states as candidates, or must candidates be actual V2X-delivered messages so delay/loss robustness is represented?
4. Should a masked lane-change action be a no-op or raise an error? Should all missing/extra action keys be rejected?
5. For comfort, which physical signals are required by the frozen formula—jerk and lateral acceleration as §6 says, or acceleration thresholds currently implemented in `RewardCalculator`?
6. Which exact route-progress normalization and TTC/gap computation should `RewardCalculator` receive, and what does “logged per-term rewards” mean in the output schema?
7. Does “frozen at goal” require stopping physics/autopilot or only ignoring actions and setting 100% speed difference?
8. Should route planning failure (`route_applied=False`) abort/reset the episode instead of continuing with an uncontrolled route?
9. Is the intended Phase 1 proof 20 seeds or 100 episodes per controller/policy, and what exact goal/speed acceptance threshold should gate Phase 2?
10. Which current CARLA server/venv is authoritative, and can the owner attach revision-stamped reruns after the speed change?
11. Are root `desktop.ini` and `t tests -q -k not carla and not highway` known files, and should the owner clean them up separately?
12. Which teammate owns the canonical reward/contract update and the `docs/sota_rows_1.md` deliverable?
