# Milestone 9 — Experimental Readiness Audit

**Repository:** `C:\Users\souro\Desktop\V2X-MARL`  
**Branch/HEAD (inspected):** `feat/teammate-selection` @ `a82dc6d` (`git log --oneline -10` captured)  
**Audit date:** 2026-11-20 (workspace context)  
**Scope:** Paired `nearest_k` vs `random_k` driving-performance comparison using the same deterministic observation-conditioned controller (`controllers/observation_heuristic.py`). **Audit-only; no runtime code modified, no experiments launched.**

## 1. Executive summary

- **Overall readiness:** **NOT READY** for scientifically defensible Milestone 9 paired experiments without additive protocol/tooling and explicit controls. Core building blocks exist (V2X integration, selection baselines, 56-D obs, deterministic controller, live validation scripts/artifacts), but **episode-horizon control, seed/reproducibility semantics, scenario-equivalence guarantees, and metric/logging completeness** are only partially verified or not verified.
- **Key blockers/risks:** (1) `max_steps` is hardcoded in env and not a clean experiment parameter (`envs/carla_env.py:94`), while scripts sometimes enforce their own caps, creating ambiguity for timeout/success measurement; (2) **selection RNG is fixed at env construction and not reset per episode** (`envs/carla_env.py:179`, `grouping/random_sel.py`), and V2X channel resets replay its constructor seed (`envs/carla_env.py:241`, `envs/v2x_channel.py:100-110`) — current pairing scripts do not establish a rigorously documented, reproducible per-episode seed architecture for random-k; (3) CARLA/TM/background/traffic-light dynamics and cross-run world state introduce uncontrolled variability not fully characterized; (4) env `info` omits some fields needed for analysis (e.g. explicit `ending_reason` string, per-step actions/masks/obs snapshots, valid-candidate counts) — runners must log them consistently; (5) five seeds are **exploratory only**, not confirmatory.
- **Protected files untouched (verified):** `envs/v2x_channel.py`, `tests/test_v2x_channel.py`, `docs/v2x_channel.md`, `envs/obs_builder.py` (no runtime modifications made). Milestone 7/8A results preserved.

### Requirement classification summary

| Area | Requirement | Status | Evidence (file/function/loc) |
|---|---|---|---|
| Protection | Git inspected; no reset/clean/overwrite/commit/push; protected files untouched | **VERIFIED** | `git status`, `git log --oneline -10`, `git diff --stat`; protected paths unchanged in this audit |
| Horizon/termination | Max horizon configurable & understood | **PARTIALLY VERIFIED** | `envs/carla_env.py:58,94,824` (`MAX_STEPS=300`, `self.max_steps`, passed to `evaluate_termination`) |
| Horizon/termination | Timeout vs success distinguished | **VERIFIED** | `envs/termination.py:38-101` (`timeout`→truncated, `goal`→terminated+success) |
| Horizon/termination | Team success requires all 4 to reach | **VERIFIED** | `envs/termination.py:82-83,98`; `envs/carla_env.py:946-951` |
| Horizon/termination | Individual arrivals recorded | **VERIFIED** | `envs/carla_env.py:133,670,950-951`; `envs/termination.py:125-139` |
| Horizon/termination | Collision/off-road/stuck detection defined | **VERIFIED** | `envs/carla_env.py:773-776,800-815,942-968`; `envs/termination.py:19-29,71-90` |
| Horizon/termination | Individual vs team-wide termination clear | **PARTIALLY VERIFIED** | Team episode end in `evaluate_termination`; per-agent finished/reached handling in `envs/carla_env.py:748,767,830,837,942-983` |
| Horizon/termination | Rewards available after termination | **PARTIALLY VERIFIED** | Rewards returned on terminating step (`envs/carla_env.py:833-940`); no env-level post-episode accumulation |
| Horizon/termination | Completed vehicle can affect remaining agents | **PARTIALLY VERIFIED** | Finished agents removed from physics/moved underground and excluded in many paths (`envs/carla_env.py:970-983,767,774,802,1055,1092,1229`) — residual effects not fully excluded |
| Horizon/termination | Metrics distinguish success/collision/off-road/stuck/timeout | **PARTIALLY VERIFIED** | `envs/termination.py:80-101,153-178`; env `info` flags set but explicit `ending_reason` string not stored (`envs/termination.py:154-171`, `envs/carla_env.py:898-899`) |
| Reproducibility | All randomness sources identified | **PARTIALLY VERIFIED** | Seeds/RNGs in `envs/carla_env.py`, `envs/v2x_channel.py`, `grouping/random_sel.py`; CARLA/TM internals not fully controllable/verified |
| Reproducibility | Same seed reproduces same initial scenario | **NOT VERIFIED** | TM/CARLA/world nondeterminism possible despite seeding (`envs/carla_env.py:605-610,352-359,485-486`); no determinism test evidence |
| Reproducibility | Separate documented seed architecture for random-k | **NOT VERIFIED** | Selection RNG fixed at init, not reset per episode (`envs/carla_env.py:164-182`); current scripts pair modes but do not formalize per-episode selector streams |
| Seeds | Seeds 7,17,27,37,47 sufficient for initial exploratory comparison | **PARTIALLY VERIFIED** (exploratory only) | Small N, single town/config; cannot support confirmatory claims (recommendation) |
| Equivalence | Two strategies can use equivalent scenario/config | **PARTIALLY VERIFIED** | Same env/config possible in code; sequential-run state and stochastic components limit guarantees |
| Equivalence | Paired execution protocol defined | **NOT VERIFIED** | No canonical Milestone 9 runner/protocol artifact; existing scripts differ (`scripts/m8a_run.py`, `scripts/m8a_validate_controller.py`) |
| Logging | Same controller/params for both strategies | **VERIFIED** | `controllers/observation_heuristic.py` (no mode dependency); usage in `scripts/m8a_run.py:253`, `scripts/m8a_validate_controller.py:205` |
| Logging | Current logs sufficient for required measures | **PARTIALLY VERIFIED** | Env provides selection/V2X/per-agent fields (`envs/carla_env.py:900-939`); actions/masks/obs/valid-candidate details require runner logging |
| Metrics | Feasible definitions for all listed metrics | **PARTIALLY VERIFIED** | Core fields exist (`envs/termination.py`, `envs/carla_env.py`, `envs/metrics.py`); some ambiguous/misleading without careful aggregation |
| Statistics | Paired protocol with CIs/effect sizes/missing-data handling | **NOT VERIFIED** | Design proposed in this report; no implemented M9 analysis pipeline |
| Results org | `results/milestone9/` structure/schema defined without overwriting | **PARTIALLY VERIFIED** | Proposed below; not yet created/implemented (additive when done) |
| Report | `docs/audit/milestone9_experimental_readiness.md` produced | **VERIFIED** | This document |

---

## 2. Verified existing implementation behavior (facts)

### 2.1 Environment, controller, and artifacts
- **Controller:** `controllers/observation_heuristic.py` — deterministic, stateless decision logic, no CARLA/global RNG; consumes 56-D obs + mask; candidate-dependent (`ObservationHeuristicController.act/validate_observation/parse_candidates`, decision rules lines ~239-296). Used identically in validation (`scripts/m8a_validate_controller.py:25,205,210`).
- **Validation runner:** `scripts/m8a_validate_controller.py` — Town03, 4 CAVs, cluster spawn 45 m, `candidate_source="v2x"`, `V2XConfig(range_m=50, delay_steps=0, loss_prob=0)`, `selection_k=2`, paired modes `nearest_k/random_k`, same controller instance; logs per-step actions/masks/obs metadata/selection IDs/pool/missing/rewards/terminations/V2X episode stats to `results/milestone8a/validation/`.
- **Prior runner:** `scripts/m8a_run.py` — similar paired config but uses `policies.observation_heuristic.ObservationHeuristicController` (equivalent logic family) and different seed/episode derivation (`reset_seed=seed+ep`, `env_seed=seed`, `selection_seed=v2x_seed` bases; fresh env per episode). Demonstrates current (non-canonical) experimentation patterns.
- **Reported tests (context):** new controller tests `tests/test_controllers_observation_heuristic.py` and CARLA-free suite reported as 314 passed, 8 skipped, 17 deselected (not re-run as part of this audit beyond prior evidence).

### 2.2 Core env/termination/reward/metrics behavior
- **Horizon:** `MAX_STEPS=300` (`envs/carla_env.py:58`), stored as `self.max_steps` (`:94`), used in termination check (`:824`), no constructor/config parameter.
- **Decision/sim timing:** `SIM_DT` 20 Hz, `TICKS_PER_STEP` implied by loop, `DECISION_DT_SECONDS=0.2` (`envs/metrics.py:8`); `step()` ticks `TICKS_PER_STEP` then increments `step_count` (`envs/carla_env.py:769-771`).
- **Termination logic:** `evaluate_termination` priority: collision (`any collided`) → goal (`all reached`) → off_road (`any off_road_steps >= 51`) → stuck (all non-reached stuck, with grace) → timeout (`step_count >= max_steps`) (`envs/termination.py:38-101`). Returns `terminated` for collision/goal/off_road/stuck, `truncated` for timeout, `success` only for goal.
- **Off-road:** consecutive count via `update_off_road_steps` (`envs/termination.py:19-24`), off-road when map waypoint `project_to_road=False` is `None` (`:26-28`), checked per non-finished agent each step (`envs/carla_env.py:800-815`).
- **Stuck:** `STUCK_SPEED_MS=0.5`, `STUCK_STEPS=50`, `STUCK_GRACE_STEPS=25` (`envs/carla_env.py:67-69`); red light resets stuck counter/marks `red_light`, slow speed increments, moving clears (`envs/carla_env.py:942-968`); team-stuck per `envs/termination.py:71-89`.
- **Goals/arrivals:** goal when within `GOAL_RADIUS_M=8.0` (`envs/carla_env.py:63-65`, `946-947`); sets `reached/finished/stop_reason/arrival_step`, removes finished agent (`:947-952,970-983`).
- **Episode step outcomes:** env computes rewards/kinematics, sets `terminated={a: bool(result.terminated or self.reached[a])}`, `truncated` all same, builds `info`, returns `(obs,rewards,terminated,truncated,info)` (`envs/carla_env.py:833-940`). Finished agents get zero-filled obs on next obs build (`:1449-1452`) and zero reward if already finished at step start (`:837-842`).
- **V2X/selection in info:** per-agent `candidate_source`, `v2x_received` (`count/sender_ids/ages`), `selection` (`strategy,k,candidate_source,pool_size,selected_count,missing_slots,selected_ids,selected_scores`) when selector active; episode `info["v2x"]` step/episode stats when V2X source (`envs/carla_env.py:901-939`, reset info includes V2X/selection metadata `:713-733`).

---

## 3. Audit A — Episode horizon and termination

### Findings (facts)

| Question | Finding | Classification | Evidence |
|---|---|---|---|
| 1. How max episode horizon is configured | Hardcoded constant `MAX_STEPS=300` assigned to `self.max_steps` in `__init__`; no constructor arg; `reset` does not set it; termination uses `self.max_steps`. External code may mutate attribute or cap loops independently. | **PARTIALLY VERIFIED** | `envs/carla_env.py:58`, `:94`, `:582-741` (no max_steps), `:824` |
| 2. Timeout correctly distinguished from success | Yes at evaluator level: timeout → `truncated=True, terminated=False, success=False, reason="timeout"`; success only on all-reached goal. | **VERIFIED** | `envs/termination.py:82-101` |
| 3. All four CAVs must reach destinations for team success | Yes. `reason=="goal"` iff `all(reached.values())`; `success` true only for goal. | **VERIFIED** | `envs/termination.py:82-83,98`, `:157-158` |
| 4. How individual arrivals recorded | `arrival_step[agent_id]` set once on reach; `reached`/`finished` set; per-agent `goal_reached`, `stop_reason`, `arrival_step` exposed in step info. | **VERIFIED** | `envs/carla_env.py:133,598,670,946-952`; `envs/termination.py:125-139` |
| 5. Collisions/off-road/stuck detection | Collision via `CollisionMonitor.is_collided` (any collision triggers team collision). Off-road via map waypoint None with consecutive-step threshold 51. Stuck via low speed >50 steps after 25-step grace, paused at red lights. | **VERIFIED** | `envs/carla_env.py:773-776,800-815,942-968`; `envs/safety.py` (monitor, not fully re-audited here), `envs/termination.py:19-29,71-90` |
| 6. Termination individual or team-wide | **Episode termination decision is team-level** (single `TerminationResult`), but **agents can individually finish on arrival** before team episode end; per-agent terminated flags include reached state while episode-level truncation/termination applies to all. | **PARTIALLY VERIFIED** | `envs/termination.py:80-101`; `envs/carla_env.py:817-831,877-880,942-952` |
| 7. Rewards available after termination | Per-step rewards/team_reward returned on the step that ends episode; finished agents yield zero reward on steps where they were already finished. No env persistence of cumulative episode reward after `close/reset` except caller-collected data. | **PARTIALLY VERIFIED** | `envs/carla_env.py:833-876,900,940`; `envs/reward_integration.py:26-31` |
| 8. Completed vehicle can affect remaining agents | Finished agent is braked, physics disabled, teleported below map, excluded from actions/collision checks/obs candidates/V2X/TTc in observed paths; however residual CARLA/TM/sensor side effects cannot be fully ruled out from static audit. | **PARTIALLY VERIFIED** | `envs/carla_env.py:767,774,783,802-805,970-983,1055-1061,1092,1111-1114,1229,1449-1452` |
| 9. Metrics distinguish success/collision/off-road/stuck/timeout | Evaluator distinguishes; `build_step_info` overwrites terminal flags consistently but **does not emit `ending_reason` string key**. `scripts/m8a_run.py` expects `ending_reason`, so logging/analysis can be ambiguous without runner-side derivation. | **PARTIALLY VERIFIED** | `envs/termination.py:80-101,153-178`; `envs/carla_env.py:898-899`; `scripts/m8a_run.py:193,236,357` |

### Horizon recommendation (reasoned, not arbitrary)
- **Fact:** 300 decision steps = 60 s at 5 Hz (`MAX_STEPS`, `DECISION_DT_SECONDS`). Prior M8A/M8 short runs used 20–30 step caps and did not reach env termination; `scripts/check_policies.py` evaluates with configurable `--max-steps` (default 300) and reports termination distributions/arrival stats.
- **Recommendation (protocol):** for Milestone 9 paired driving-performance evaluation, run **full episodes to env termination** with `self.max_steps == 300` (60 s) as the baseline horizon, and **also record time-to-termination/arrival distributions**. If pilot data (paired seeds, both modes) shows a material fraction of informative episodes truncated at 300 s with agents still progressing (not stuck/collision/off-road), consider a justified increase (e.g. 450 steps / 90 s) **only after documenting route lengths/spawn distances and keeping the horizon identical across modes and all paired runs**. Do **not** change horizon mid-study or per-mode.
- **Required implementation note:** expose/configure horizon explicitly in the Milestone 9 runner and log it per episode; avoid relying on ad-hoc loop caps that differ from `env.max_steps` (see Audit B/F).

---

## 4. Audit B — Reproducibility and paired seeds

### Randomness sources (investigated)

| Source | Mechanism / seed control | Classification | Evidence |
|---|---|---|---|
| CARLA spawn-point selection | Spawn points from map shuffled by `rng = random.Random(self.seed)` (default mode) or cluster ordering/shuffle (cluster mode). | **PARTIALLY VERIFIED** | `envs/carla_env.py:606,608,620,528-539` |
| Vehicle spawning | `try_spawn_actor` depends on world occupancy; retries scan in seeded order but success can depend on CARLA state. | **NOT VERIFIED** (exact reproducibility) | `envs/carla_env.py:479,624-650` |
| Route generation | Route waypoints follow lanes with `rng.choice` at junctions; per-agent `random.Random(self.seed + agent_index)` in `_try_spawn_at_transform`. | **PARTIALLY VERIFIED** | `envs/carla_env.py:352-362,485-486` |
| Traffic Manager | `tm.set_synchronous_mode(True)` and `tm.set_random_device_seed(self.seed)` on reset; TM path following/behavior still subject to CARLA/TM internals. | **PARTIALLY VERIFIED** | `envs/carla_env.py:602,605,364-370` |
| Background traffic | No explicit NPC spawn routine found; env primarily spawns/controls CAVs, but V2X/candidate code can include other world vehicles if present (`v2x_broadcasters="cavs_and_background"`). | **PARTIALLY VERIFIED** | `envs/carla_env.py:85,1102-1155,1047-1086`; absence of `spawn_npc` search hit |
| V2X packet loss/delay RNG | `V2XChannel` uses `np.random.default_rng(seed)`; loss draw `self._rng.random()` per eligible pair. | **VERIFIED** (mechanism) | `envs/v2x_channel.py:93-97,172-176` |
| V2X seed lifecycle | `v2x_seed` owns channel RNG; if None uses env seed captured at init; `reset()` replays stored channel seed (`channel.reset()` no arg) — **not coupled to `reset(seed=...)`**. | **VERIFIED** (as implemented) / **risk for pairing** | `envs/carla_env.py:198-201,218-224,241-244`; `envs/v2x_channel.py:100-110` |
| Random-k selection RNG | Explicit `np.random.Generator` required; env creates `self._selection_rng = np.random.default_rng(selection_seed)` at init and **does not reset/reseed it on `reset()`**. | **VERIFIED** (as implemented) / **pairing risk** | `envs/carla_env.py:164-182`; `grouping/random_sel.py:45-47,56-85` |
| Environment resets | `reset(seed=...)` updates env seed and reseeds TM/Python RNG for spawn/route; recreates vehicles; resets channel to its stored seed; does not reseed selection RNG. | **PARTIALLY VERIFIED** | `envs/carla_env.py:582-735` |
| Controller behavior | Deterministic function of obs/mask; no RNG. | **VERIFIED** | `controllers/observation_heuristic.py:121-299`, tests |
| Python/NumPy global RNGs | Not set/reset by env; runners may set separately. | **NOT VERIFIED** (controlled) | No env-level `random.seed`/`np.random.seed` calls found; runner-dependent |

### Same-seed reproducibility
- **Status: NOT VERIFIED.** Code seeds TM and Python RNGs and uses deterministic route/cluster logic, but exact reproducibility depends on CARLA/TM determinism, map/traffic-light controllers, actor IDs/timing, and absence of hidden NPCs/state. No automated determinism/regression evidence was produced in this audit (and experiments were not run).

### Recommended reproducible seed-management architecture (recommendation; no runtime changes made)
- **Document and log per episode:** `scenario_seed` (env/spawn/route/TM), `selection_seed` (random-k stream), `v2x_seed` (channel), `episode_index`, `pair_id`, and `reset_seed` actually passed. Keep **scenario_seed identical** for paired nearest_k/random_k episodes within a pair.
- **Nearest-k:** selection deterministic; `selection_seed` irrelevant but log constant/NA.
- **Random-k:** requires an explicit, reproducible selector RNG stream. **Issue:** current env holds one `Generator` for lifetime (`envs/carla_env.py:179`) and never replays per episode; to achieve paired, reproducible stochastic behavior you must either (a) construct a fresh env (or selector RNG) per episode with a documented `selection_seed = f(pair_id, episode_index)` scheme, or (b) add a minimal, approved reseed/reset hook — **do not silently modify existing selection implementation**. Any change must preserve protected files/contracts and be tested.
- **V2X:** with `loss_prob=0, delay_steps=0`, channel RNG effect is null; still log `v2x_seed` and config. If loss>0 in future, define per-episode/per-pair reseeding explicitly (current reset replays constructor seed).
- **CARLA process protocol:** fix map (`Town03`), synchronous mode, fixed delta, TM port, no extra NPC scripts, consistent traffic-light reset settings, clean world state between paired runs (prefer fresh episode resets in a controlled order; consider process restart between seeds if drift observed).
- **Global RNG hygiene:** in runner, set/log `PYTHONHASHSEED` policy if relevant, avoid `random`/`np.random` globals (controllers/selectors already avoid globals where required).

### Suggested initial seeds `7, 17, 27, 37, 47`
- **Sufficiency:** **PARTIALLY VERIFIED for exploratory comparison only.** Five seeds provide low-resolution evidence across scenario realizations in one town/config (4 CAVs, cluster spawn, Town03). **Not sufficient for confirmatory claims or tight confidence intervals**; sensitive to high between-seed variance, rare events (collisions/timeouts), and uncontrolled CARLA stochasticity. Treat as pilot to estimate variance/effect sizes and refine horizon/sample size.

---

## 5. Audit C — Initial scenario equivalence

| Aspect | Assessment | Classification | Evidence |
|---|---|---|---|
| Map/town | Configurable (`town`, default Town03); reset can reload if changed. | **VERIFIED** | `envs/carla_env.py:80-87,151-155,585-589` |
| Initial vehicle positions/orientations | Derived from CARLA spawn points/routes selected via seeded RNG + spawn validity checks; cluster mode constrains distances. | **PARTIALLY VERIFIED** | `envs/carla_env.py:352-362,469-579,606-650` |
| Initial speeds | Vehicles spawned then settled with ticks; target speed set to `CRUISE_SPEED_MPS` and TM configured. | **PARTIALLY VERIFIED** | `envs/carla_env.py:557-560,639-643,682-684` |
| Routes/destinations | Planned per-agent from seeded route generation; validated/rejected/resampled; uniqueness asserted. | **PARTIALLY VERIFIED** | `envs/carla_env.py:352-362,485-516,651-663` |
| Traffic conditions/background | No explicit NPC spawning found; other actors only if pre-existing in world. Traffic lights reset/configurable. | **PARTIALLY VERIFIED** | `envs/carla_env.py:427-466,610`, absence of NPC spawn |
| Spawn configuration | `spawn_mode`, `spawn_cluster_max_distance` configurable and logged in reset info. | **VERIFIED** | `envs/carla_env.py:114-118,248-265,699-703` |
| V2X configuration | `candidate_source`, `V2XConfig`, `v2x_seed`, broadcasters configurable. | **VERIFIED** | `envs/carla_env.py:106-112,184-245` |
| Episode horizon | Env-level fixed at 300 unless attribute mutated; must be controlled/logged. | **PARTIALLY VERIFIED** | `envs/carla_env.py:94,824` |
| Sequential-run CARLA state effects | Same client/world reused; actors destroyed on reset/close, sync/TM toggled; traffic lights reset by env settings but residual TM/world/controller state possible. | **NOT VERIFIED** (equivalence guarantee) | `envs/carla_env.py:320-349,582-605,610,427-466` |

### Paired execution protocol (recommendation)
1. **Per pair `(seed, episode_index)`:** run `nearest_k` and `random_k` with identical `scenario_seed/reset_seed`, town, n_agents, spawn config, horizon, V2X config (`range/loss/delay/broadcasters`), controller module/params, and candidate settings (`k=2`, 50 m). 
2. **Isolation:** create a fresh `CarlaMultiAgentEnv` per episode (or per mode-run) with explicit seeds; destroy/close fully between runs; log CARLA map, actor spawn transforms/routes (available in reset flow/state), and all seeds. Prefer same-process deterministic ordering or, if needed, CARLA server restart between pairs/seeds after empirical drift checks.
3. **Random-k parity:** for each pair, define and log a reproducible `selection_seed_pair`; ensure the selector consumes randomness in a documented, repeatable way (address init-only RNG lifecycle before confirmatory runs).
4. **Abort criteria:** if route/spawn validation fails, or CARLA connection/world state invalid, mark episode `failed` with reason and exclude per pre-registered rule (do not silently substitute).
5. **Artifacts:** save raw per-step logs and per-episode metadata for both modes under the same `pair_id`.

**Uncontrollable/uncertain variables (current setup):** exact TM internal decisions, CARLA physics micro-variations, traffic-light controller progression details beyond configured durations, actor ID assignment, and any latent world actors; therefore **exact scenario equivalence is not guaranteed** without additional validation instrumentation.

---

## 6. Audit D — Controller and action logging

### Controller parity
- **Same controller and parameters for both strategies:** **VERIFIED.** `controllers/observation_heuristic.py` defines fixed thresholds/rules with no selection-mode parameter; `scripts/m8a_validate_controller.py` instantiates one `ObservationHeuristicController()` and reuses it for both modes (`:205,208-213`). `scripts/m8a_run.py` similarly uses one controller instance per run (`:253`). No per-strategy tuning hooks present.

### Current logging capability

| Measure | Available now? | Classification | Evidence |
|---|---|---|---|
| Action distributions | **Via runners** (actions known at decision time; some runners log them) | **PARTIALLY VERIFIED** | `scripts/m8a_validate_controller.py:89-121,120` logs `action`; `scripts/m8a_run.py` does not store actions in step records (`:139-156`) |
| Action disagreement between strategies | **Derivable offline** only if paired per-step actions logged | **NOT VERIFIED** (in env/default m8a_run) | Requires aligned per-step logs |
| Candidate pool size | **Yes** in selection metadata | **VERIFIED** | `envs/carla_env.py:724,924` (`pool_size`) |
| Selected candidate IDs | **Yes** | **VERIFIED** | `envs/carla_env.py:731,927` (`selected_ids`) |
| Valid candidate observation count | **Not directly in env info**; derivable from obs/selection (`selected_count`) or runner computation | **PARTIALLY VERIFIED** | `Selection.selected_count` (`grouping/base.py:42-45`), logged as `selected_count` (`envs/carla_env.py:729,925`) |
| Missing candidate slots | **Yes** | **VERIFIED** | `envs/carla_env.py:730,926`; `grouping/base.py:47-50` |
| V2X delivery ratio | **Per-step counts and episode stats available**; ratio derivable | **PARTIALLY VERIFIED** | `envs/carla_env.py:935-939`; `envs/v2x_channel.py:153-211` |
| Relative velocity/positions of selected candidates | **In observations** (56-D candidate slots) and adapter records pre-obs; not emitted as structured env info fields | **PARTIALLY VERIFIED** | `envs/obs_builder.py` candidate layout; `envs/carla_env.py` passes `candidates_for_obs` to builder but does not log raw rel features |
| Controller response to candidate changes | **Partially inferable** from paired actions/obs/selection diffs if logged; controller exposes `last_reason` but env/runners do not record it | **PARTIALLY VERIFIED** | `controllers/observation_heuristic.py:122-123,254-296` |

### Recommended additional logging (additive, no obs-dim change)
- Per step, per agent: `action`, `action_mask`, `ego_speed`, `controller_reason` (optional), `selected_ids`, `selected_count/pool_size/missing_slots`, `received_count/sender_ids/ages`, `nearest/selected candidate rel features` (from obs slots or pre-selection records), and termination flags.
- Per episode: all seeds/config hashes, spawn/route metadata (positions, goals, route lengths), per-agent arrival steps, cumulative rewards, min TTC/jerk, V2X episode counters/ratios, and final `ending_reason` (explicit string).
- Per pair: aligned `pair_id`, scenario identifiers, and stepwise action agreement/disagreement summaries.

---

## 7. Audit E — Metrics and scientific evaluation

> **Note:** "higher/lower is better" is interpreted for driving performance unless stated; reward sign/scale follows env design and should be reported transparently.

| Metric | Definition (precise) | Unit | Source | Aggregation | Better | Limitations / caveats | Status |
|---|---|---|---|---|---|---|---|
| Team success rate | Fraction of episodes with team goal success (`success=True`, all CAVs reached; timeout/collision/off-road/stuck are not success). | proportion [0,1] | `TerminationResult.success` / `info['success']` at episode end | episode → mean over episodes/seeds (paired deltas) | higher | Sensitive to horizon; success impossible after collision/off-road/stuck. | **PARTIALLY VERIFIED** |
| Collision rate | Fraction of episodes ending/flagged as collision (team-level `any collided` priority). | proportion | `termination.py` reason; `info['collision']` | episode-level | lower | A single agent collision ends episode (team cost); may mask which agent/when without per-agent analysis. | **VERIFIED** (definition) |
| Off-road rate | Fraction of episodes ending as off-road (any agent reaches off-road step limit). | proportion | `info['off_road']`, `off_road_steps` | episode-level | lower | Threshold-based (51 consecutive steps); depends on map waypoint queries. | **VERIFIED** (definition) |
| Stuck rate | Fraction of episodes ending as stuck (all non-reached agents stuck per evaluator). | proportion | `info['stuck']`, `stuck_by_agent` | episode-level | lower | Red-light grace logic and grace steps affect labeling; team-stuck differs from individual blocked. | **VERIFIED** (definition) |
| Timeout rate | Fraction of episodes truncated at horizon without terminal success/failure reason. | proportion | `truncated` / `step_count>=max_steps` | episode-level | lower (usually) | Confounded with horizon choice; not equivalent to safe completion. | **VERIFIED** (definition) |
| Team reward | `mean(agent rewards over team) + team_collision penalty` per step (env), episode total/mean commonly aggregated by runners. | scalar reward | `RewardCalculator.team_reward`, `info['team_reward']` (`envs/carla_env.py:874-876,900`) | step and/or episode | higher (contextual) | Shaped reward; depends on term weights/events; can be misleading if episodes differ in length/termination. | **PARTIALLY VERIFIED** |
| Individual reward | Per-agent scalar reward per step from `compute_agent_step_reward` (progress, collision, TTC, comfort, goal, idle). | scalar reward | `envs/reward.py`, `envs/reward_integration.py`, `envs/carla_env.py:860-873` | agent-step/episode | higher (contextual) | Finished agents return 0 after finished; one-time event flags must be step-accurate. | **PARTIALLY VERIFIED** |
| Episode duration | Number of decision steps until episode end (or cap). | steps (and seconds = steps*0.2) | `info['steps']`, runner counters | episode-level | context-dependent | Capped runs vs env terminations must be distinguished; early termination shortens duration. | **PARTIALLY VERIFIED** |
| Arrival steps | Step index when an agent first reaches goal. | steps | `arrival_step` per agent (`envs/carla_env.py:950-951`, info per-agent) | per-agent; aggregate over arrived agents | lower (faster) | Undefined for non-arrivals; averaging only arrivals biases results. | **VERIFIED** |
| Minimum TTC | Minimum (capped) time-to-collision observed over episode (same-lane lead logic). | seconds (cap 10.0) | `envs/metrics.py:13-30`, tracked in env (`min_ttc`) and emitted on episode end | episode-level | higher (safer) | Capped/approximate; depends on detection geometry; may be less informative when no closing interactions. | **PARTIALLY VERIFIED** |
| Jerk / comfort | Longitudinal jerk magnitude derived from acceleration changes, bounded inputs; episode aggregate currently mean of samples in env info. | m/s³ (bounded) | `envs/metrics.py:32-85`, `envs/carla_env.py:795,894` | episode-level (mean samples) | lower jerk better comfort | Longitudinal only; averaging can hide peaks; samples collected for active agents/steps as implemented. | **PARTIALLY VERIFIED** |
| Candidate availability | Eligible pool size / selected count / missing slots per agent-step. | count | `selection` metadata (`pool_size,selected_count,missing_slots`) | per-step/episode means | context-dependent | Depends on range, traffic, V2X delivery, and selector; not identical to geometric neighbors unless ground-truth. | **VERIFIED** (fields available) |
| V2X delivery ratio | Delivered / sent messages (per step or episode), with drops by range/loss. | ratio [0,1] or % | `envs/v2x_channel.py:153-211`, `envs/carla_env.py:935-939` | step/episode | higher | With `loss=0, delay=0`, drops mainly by range; sent counts depend on broadcasters/state. | **PARTIALLY VERIFIED** |
| Missing selection slots | `k - selected_count` padding slots. | count | `grouping/base.py:47-50`, selection info | per-step/episode | lower (more info) | Structural when pool < k; interpretation differs by mode. | **VERIFIED** |
| Action distributions | Frequency of chosen actions per agent/mode/condition. | counts/proportions | Runner logs (env does not store actions) | per-episode/aggregate | descriptive | Must log actual executed actions and masks; mask feasibility constrains distribution. | **PARTIALLY VERIFIED** |

**Early-termination/masking cautions (important):** episode-level rates (success/collision/off-road/stuck/timeout) are mutually exclusive by termination priority but hide per-agent outcomes; means over variable-length episodes (reward, jerk) and averages over only-arrived agents (arrival steps) can be biased. Always report counts, denominators, episode lengths, and per-agent outcomes; stratify by ending reason where useful.

---

## 8. Audit F — Statistical experiment design

### Paired experimental protocol (recommendation)
- **Unit of pairing:** for each `seed s` in `{7,17,27,37,47}` and each `episode_index e` (if >1), run one `nearest_k` and one `random_k` episode under matched scenario settings (`pair_id = s:e`) with identical controller/config/horizon and documented seeds.
- **Primary comparison:** within-pair differences `Δ = metric(nearest_k) - metric(random_k)` for team outcomes and key secondary metrics.
- **Execution order:** randomize or counterbalance order of modes within pairs/runs to mitigate time-drift in CARLA/TM state; restart CARLA or enforce clean resets if drift detected (empirical pre-check).

### Recommendations
| Item | Recommendation | Rationale / status |
|---|---|---|
| Number of seeds | **Pilot/exploratory:** 5 seeds `{7,17,27,37,47}`. **For stronger evidence:** expand (e.g. ≥20 seeds or more episodes/seed) after variance pilot. | **PARTIALLY VERIFIED** need; 5 seeds insufficient for confirmatory inference. |
| Episodes per seed | Start with **1 episode/seed** for strict paired pilot if runtime-limited; prefer **≥5 episodes/seed** (or more) once protocol stable to estimate within-seed stochasticity (esp. random-k/CARLA). | Depends on observed variance/cost. |
| Episode horizon | **300 steps (60 s)** baseline, full env termination; adjust only with pre-specified justification and identical application. | `envs/carla_env.py:58,94` |
| Paired statistical comparisons | Wilcoxon signed-rank or paired permutation test on within-pair deltas (nonparametric, small N); report paired mean/median delta. Avoid unpaired tests. | Robust for small samples; **exploratory** at N=5. |
| Confidence intervals | Bootstrap CIs over pairs (e.g. percentile or BCa) for mean/median deltas; with 5 pairs interpret as **descriptive/unstable**. | Do not overstate precision. |
| Effect sizes | Report absolute/relative deltas and standardized effect (e.g. paired Cohen’s d_z or Cliff’s delta) with caveats at small N. | Communicate magnitude beyond p-values. |
| Missing/failed episodes | Predefine handling: if either mode in a pair fails (spawn/CARLA crash/invalid logs), **exclude the entire pair** and record cause; report failure rates separately. Never impute outcomes. | Preserves pairing integrity. |
| Repeated runs / stochastic random-k | Log `selection_seed` and realized selections; if multiple episodes/seed, model episode as nested within seed/pair; aggregate carefully. Do **not** claim significance from 5 seeds alone. | `grouping/random_sel.py`, env RNG lifecycle risks. |

**Exploratory vs confirmatory:** results from 5 seeds × limited episodes must be labeled **exploratory** (hypothesis-generating). Confirmatory claims require pre-registered protocol, adequate power/sample size, controlled reproducibility, and appropriate uncertainty quantification.

---

## 9. Audit G — Result organization

### Existing result directories (inspected)
- `results/milestone7/` (8 JSON experiment files) — **preserve**.
- `results/milestone8/` (smoke logs/figures/json/csv) — **preserve**.
- `results/milestone8a/smoke/` and `results/milestone8a/validation/` — **preserve**.
- Other legacy artifacts (`qmix_highway/`, txt/md/png/csv logs) — **preserve**.

### Proposed `results/milestone9/` (additive, no overwrites)

```text
results/
└── milestone9/
    ├── raw/                 # unmodified per-run logs, CARLA/stdout captures, failure traces
    ├── episodes/            # one JSON (+ optional per-step CSV/JSONL) per episode
    ├── summaries/           # per-pair and per-mode episode summary tables (CSV/JSON)
    ├── statistics/          # paired deltas, bootstrap outputs, effect sizes, test reports
    ├── figures/             # plots generated only from summaries/statistics/raw refs
    ├── tables/              # publication/report tables
    └── metadata/            # experiment manifest, configs, schema versions, environment snapshot
```

### Experiment identifier and schema (recommendation)
- **Experiment/run ID:** `m9_{timestampUTC}_{gitsha7}_{config_hash8}`
- **Pair ID:** `{run_id}__seed{scenario_seed}__ep{episode_index}`
- **Episode ID:** `{pair_id}__{strategy}` (`nearest_k` or `random_k`)
- **Episode JSON schema (minimum):**
  - `episode_id`, `pair_id`, `strategy`, `scenario_seed`, `selection_seed`, `v2x_seed`, `reset_seed`, `episode_index`
  - `timestamp_utc`, `git_sha`, `execution_status` (`success|failed|aborted`), `failure_reason`
  - `config`: `{town,n_agents,sim_hz,decision_hz,max_steps,spawn_mode,spawn_cluster_max_distance,candidate_source,v2x_config,selection_k,controller:{module,class,params_hash}}`
  - `initial_scenario`: spawn transforms/actor mapping, routes/goals, route lengths, reset info
  - `termination`: `{ended, ending_reason, success, collision, off_road, stuck, timeout, steps}`
  - `agents`: per-agent `{arrival_step, goal_reached, collision, off_road, stuck, stop_reason, reward_total, route_metrics...}`
  - `team_metrics`: `{team_reward_total, team_reward_mean_step, min_ttc, jerk, duration_steps,...}`
  - `communication_metrics`: `{pool_size_stats, selected_count_stats, missing_slots_stats, v2x_sent,delivered,dropped_*,delivery_ratio,...}`
  - `step_log_ref`: path to raw per-step records (actions, masks, selections, V2X received, obs metadata as policy allows)
- **Integrity:** write raw first; generate summaries/statistics only from raw/episode files; include checksums/row counts in manifest.

---

## 10. Identified risks and blockers

| ID | Risk/Blocker | Severity | Classification | Evidence / rationale |
|---|---|---|---|---|
| R1 | Horizon not first-class/configured; loop caps can diverge from env termination | High | **BLOCKER (protocol)** | `envs/carla_env.py:58,94,824`; scripts use own `--max-steps` loops |
| R2 | Selection RNG not reset per episode; random-k reproducibility/paired stream unclear | High | **BLOCKER (scientific)** | `envs/carla_env.py:179`; `grouping/random_sel.py:77-85` |
| R3 | V2X channel reset replays constructor seed (independent of per-episode scenario seed) | Medium/High (if loss>0; low now) | **PARTIALLY VERIFIED** | `envs/carla_env.py:218-224,241`; `envs/v2x_channel.py:100-110` |
| R4 | CARLA/TM/world nondeterminism and sequential-state carryover | High | **NOT VERIFIED** | `envs/carla_env.py` seeding/reset logic; no determinism proof |
| R5 | `ending_reason` string not emitted in env info; metric/log schema inconsistencies across runners | Medium | **PARTIALLY VERIFIED** | `envs/termination.py:154-171`; `envs/carla_env.py:898-899`; runner expectations |
| R6 | Actions/masks/obs details and action disagreement not consistently logged | Medium | **PARTIALLY VERIFIED** | `scripts/m8a_run.py` step records omit actions; env info omits them |
| R7 | Small-N statistics (5 seeds) can mislead; rare-event rates unstable | High | **RISK** | Design limitation |
| R8 | Completed-agent removal/teleport may have subtle side effects | Medium | **PARTIALLY VERIFIED** | `envs/carla_env.py:970-983` |
| R9 | No canonical Milestone 9 runner enforcing pairing, manifests, and failure handling | High | **NOT VERIFIED** | Existing scripts are M8A-oriented |

---

## 11. Recommended implementation changes (minimal, additive; approval required before runtime changes)

| Problem | Minimal change | Why necessary | Proposed tests |
|---|---|---|---|
| Ambiguous horizon control | Add **additive** Milestone 9 runner/config that sets and logs `env.max_steps` (or accepts explicit horizon) and never relies on mismatched loop caps; record `configured_max_steps` and env `ending_reason`. | Ensures timeout/success measured against a documented horizon. | Unit/integration: runner passes horizon through; episode ends at timeout when configured; logs match. |
| Random-k pairing/reproducibility | **Additive** episode-level selector seeding strategy in M9 runner: create fresh env per episode with explicit `selection_seed_pair`, or (if approved) add a minimal API to reseed selection RNG at reset/start-of-episode without changing selector semantics/contracts. | Current init-only RNG (`envs/carla_env.py:179`) prevents controlled per-episode random-k streams. | Determinism test: same `selection_seed_pair` → identical selected IDs given identical candidate pool/order; different pairs differ as expected; nearest-k unaffected. |
| Missing explicit `ending_reason`/schema in outputs | In M9 runner, derive and store canonical `ending_reason` and standardized episode schema from existing info/flags (no env behavior change required). Optionally propose a backward-compatible info-field addition later (approved). | Reliable aggregation/statistics and auditability. | Schema validation tests on saved JSON/CSV. |
| Incomplete paired logging | Add additive per-step/per-episode logging (actions, masks, selections, V2X, obs metadata summaries, disagreement-ready fields) and manifest/checksums under `results/milestone9/`. | Required for action disagreement and defensible analysis. | Logger unit tests; small dry-run schema check (no CARLA). |
| Equivalence validation | Add additive pre-experiment equivalence check script (paired dry-run on few short episodes) comparing spawn/route/reset metadata and logging divergences. | Detect uncontrolled differences before large runs. | Non-CARLA tests for comparator; manual/controlled CARLA pilot. |

**No changes proposed** to protected files, 56-D observation contract, rewards, termination semantics, or MARL algorithms. Any runtime behavior modification requires explicit authorization.

---

## 12. Final readiness verdict

| Area | Verdict | Rationale |
|---|---|---|
| Repository/protection compliance | **VERIFIED** | Audit performed with inspection only; no resets/cleans/overwrites/commits/pushes; protected files untouched; M7/M8A preserved. |
| Controller readiness for paired use | **VERIFIED (functional) / PARTIALLY VERIFIED (scientific)** | Same deterministic controller/params usable for both modes; candidate-sensitive; not a safety/performance guarantee. |
| Termination/metric semantics | **PARTIALLY VERIFIED** | Core logic clear in code; logging/aggregation gaps and horizon control issues remain. |
| Reproducibility & scenario equivalence | **NOT VERIFIED** | Seed lifecycle gaps (esp. random-k), CARLA/TM nondeterminism, and no canonical paired protocol. |
| Logging/analysis readiness | **PARTIALLY VERIFIED** | Key fields exist but M9-grade paired, schema-consistent logging pipeline not in place. |
| Statistical readiness | **NOT VERIFIED** | Protocol/analysis not implemented; N=5 exploratory only. |

**Final verdict: Milestone 9 is NOT READY to run as a scientifically defensible paired comparison.** It may proceed only as a **small, explicitly labeled exploratory pilot** after implementing/approving the minimal additive protocol, seed/logging controls, and pre-registered analysis rules above.

---

## 13. Exact commands to reproduce the audit

Run from repository root `C:\Users\souro\Desktop\V2X-MARL` (inspection only):

```powershell
# Repository inspection (required)
git status
git log --oneline -10
git diff --stat

# Identify artifacts / docs
Get-ChildItem results -Recurse -File | Select-Object FullName, Length
Get-ChildItem docs -Recurse | Select-Object FullName

# Key evidence files reviewed (examples)
Get-Content envs\carla_env.py
Get-Content envs\termination.py
Get-Content envs\reward.py
Get-Content envs\reward_integration.py
Get-Content envs\metrics.py
Get-Content envs\v2x_channel.py
Get-Content envs\obs_builder.py
Get-Content grouping\base.py
Get-Content grouping\radius.py
Get-Content grouping\random_sel.py
Get-Content controllers\observation_heuristic.py
Get-Content scripts\m8a_validate_controller.py
Get-Content scripts\m8a_run.py
Get-Content scripts\check_policies.py
Get-Content configs\env.yaml
Get-Content configs\eval.yaml

# Targeted code evidence (examples)
Select-String -Path envs\carla_env.py -Pattern "max_steps|MAX_STEPS|def reset|_configure_selection|_reset_v2x_state|v2x_seed|selection_seed|set_random_device_seed|spawn|traffic"
Select-String -Path envs\termination.py -Pattern "evaluate_termination|ending_reason|timeout|goal"
Select-String -Path envs\v2x_channel.py -Pattern "default_rng|reset|random"
```

**No runtime behavior-modifying commands, no CARLA experiments, and no Git mutating operations were executed as part of this audit.**
