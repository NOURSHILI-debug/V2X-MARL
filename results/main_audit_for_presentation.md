# Main audit for presentation

**Scope and method.** Read-only audit of `main`/`origin/main` at `79b9e3c` on 2026-11-09. CARLA was not started and packages were not installed. Evidence uses paths and line numbers at this checkout, Git commands, and explicitly quoted check output. Existing untracked files were preserved. For per-file history use `git log --format=%an -- <path>`; authors are abbreviated in the map where clear. “VALID” below means internally supported by repository evidence, not independently reproduced in CARLA.

## 1. Main state

- `git status --short --branch`: `## main...origin/main`; worktree has pre-existing untracked `results/branch_audit_phase1.md`, `check_obs.txt`, `check_policies.txt`, `check_reward.txt`, `diagnose_stop.txt`, `keep_speed.png`, `qmix_run_guide.md`. (None was edited.)
- `git log --oneline --graph -n 25`: tip `79b9e3c Merge pull request #6 from NOURSHILI-debug/feat/phase1-core`; history also contains PR merge commits #1–#6, including #3 QMIX, #4 highway contract, #5 56-D highway adapter.
- `git branch -avv`: local/remote main and origin/main all `79b9e3c`; `origin/feat/phase1-core` is `e044753`; other remote branches: `Qmix`, `feat/env-and-vdn`, `feat/highway-contract`, `feat/highway-adapter-56d`, `test/qmix-on-carla`. Local inspection branch is `main` (requested `main-inspect` is not checked out: UNVERIFIED as a branch name because no such ref appears in command output).
- Merged PRs evidenced by main graph: #1 `feat/env-and-vdn`, #2 `feat/phase1-core`, #3 `Qmix`, #4 `feat/highway-contract`, #5 `feat/highway-adapter-56d`, #6 `feat/phase1-core`. Merged PR titles/commit graph only; hosted PR metadata/state is UNVERIFIED.
- Exact requested `git log origin/main..origin/feat/phase1-core --oneline`: **empty output**. `git diff origin/main...origin/feat/phase1-core --stat`: **empty output**. Thus no `feat/phase1-core` work remains absent from main. Two-dot `git diff origin/main..origin/feat/phase1-core --stat` is also empty.

## 2. Repo map

Purpose is from filenames/content; ownership is the distinct `git log --format=%an -- <file>` author(s). Most code files have authors `sourourjaouadi` and/or `NOURSHILI-debug`; docs include `NOURSHILI-debug`. “Both” means both appear in file history; that does not establish current assignment.

| Folder/file | Purpose | Git-history author(s) |
|---|---|---|
| `configs/` (`train.yaml`, `scenarios.yaml`, `eval.yaml`, `env.yaml`) | Training, scenario, evaluation and environment settings | sourourjaouadi; per-file details not materialized here |
| `docs/env_spec.md`, `env_spec_section12.md` | Frozen contract and extracted section 12 copy | sourourjaouadi / `NOURSHILI-debug` (file histories) |
| `docs/sota_A.md`, `sota_B.md` | Literature review rows for A and selection-focused review for B | both (file histories) |
| `docs/architecture.md` | Intended system architecture | sourourjaouadi |
| `docs/highway_adapter.md` | 25-D/56-D adapter layout and operation | `NOURSHILI-debug` |
| `envs/base.py` | Shared API/constants and candidate-block helper | sourourjaouadi |
| `envs/toy_env.py` | Two-agent 2-D intersection fixture | sourourjaouadi |
| `envs/highway_env.py` | Four-agent Highway-Env adapter | `NOURSHILI-debug` |
| `envs/carla_env.py` | CARLA multi-agent simulator adapter | both |
| `envs/obs_builder.py`, `control.py`, `termination.py`, `metrics.py`, `safety.py` | 56-D observation, action/control helpers, termination, metrics, collision monitor | mainly sourourjaouadi; per-file Git history is authoritative |
| `envs/reward.py`, `reward_integration.py` | Shared reward defaults/terms and CARLA input adaptation | both |
| `envs/v2x_channel.py` | V2X message channel module | UNVERIFIED ownership (see `git log` per file; commit history may identify one author) |
| `envs/scenarios/` | Scenario package placeholder (`__init__.py`) | UNVERIFIED |
| `grouping/` (`radius.py`, `random_sel.py`, `learned_sel.py`) | Radius, random, learned candidate selectors | sourourjaouadi / `NOURSHILI-debug` (module histories); functionality described in §4 |
| `marl/networks.py`, `buffers.py` | Shared agent network and replay buffers | sourourjaouadi |
| `marl/vdn.py`, `qmix.py`, `mappo.py` | VDN, QMIX and MAPPO implementation files | VDN/QMIX histories available; MAPPO is empty/placeholder, author attribution UNVERIFIED |
| `scripts/` | Train/evaluate/demo/diagnose/API and validation entry points | mixed; inspect each with `git log --format=%an -- <path>` |
| `tests/` | Unit, contract, adapter and integration tests | mixed; test counts in §7 |
| `results/` | Existing logs, plots, checkpoints, audit notes; see §8 | mostly tracked artifacts by commits #3/#1; untracked items are pre-existing user data |
| Root `README.md`, `requirements.txt`, `pytest.ini`, `QMIX_CARLA_INTEGRATION.md` | Project overview, dependencies, pytest config, integration guide | README/requirements/config history varies; integration guide introduced in QMIX commit `42889f7` |

## 3. Contract status on main

- Section 12 is present in `docs/env_spec.md` (lines 136 onward); 12.7 is **not found**: document ends after 12.6 at approximately line 194. Exact search `rg -n "^## 12|^### 12|12\.7" docs/env_spec.md docs/env_spec_section12.md` finds no 12.7 heading/reference.
- “Toy” remains in the initial proposal and frozen contract text (`docs/env_spec.md:144,186,188`) and in `envs/base.py:1`, tests and training scripts; the section 12 wording itself permits a toy fixture. Toy is not CARLA evidence.
- `envs/toy_env.py`: `obs_dims=4` (plural), 3 actions; `state()` length 6. It lacks `obs_dim` and `n_actions` attributes (`envs/toy_env.py:13-22`); `step` reports `reached_goal`, positions/speeds, steps instead of required `goal_reached` (`envs/toy_env.py:112-122`). It returns dictionaries for termination/truncation. `results/api_toy.txt` is stale evidence of an earlier incompatible return type (see §8).
- Highway exists at `envs/highway_env.py`; default layout is 25-D, opt-in `contract56` is 56-D, six actions, four CAVs; state length is 20 (four controlled vehicles × five state values), per `docs/highway_adapter.md` and `envs/highway_env.py:21-38,75-104`.
- CARLA exists at `envs/carla_env.py`; 56-D, six actions, configurable agent count (default four), `state()` size depends on agent count. `results/api_carla.txt` documents a two-agent run with 56-D observations, six-action masks and state length 10; it is a historical artifact, not a current CARLA run.
- Contract API: abstract `MultiAgentEnv` specifies dict observations/rewards/terminated/truncated and `state/action_masks` (`envs/base.py:46-69`). `tests/test_contract.py:58-68,138-190` defines toy/highway/CARLA factories and strict checks. Current toy deviates as noted. Highway/CARLA runtime API validity beyond tests is UNVERIFIED without CARLA.
- Info names in spec are `collision`, `goal_reached`, `steps`; terminal info adds `success`, `min_ttc`, `jerk`, scenario/town/seed (`docs/env_spec.md:162-168`). Highway and CARLA implementations include team reward and per-agent detail; toy differs (`envs/toy_env.py:112-122`).

## 4. Algorithms

- **QMIX:** substantive learner and replay buffer in `marl/qmix.py`; constructor takes `n_agents`, `obs_dim`, `state_dim`, requires six actions; shared per-agent network and state-conditioned mixer are dimension-parameterized (`marl/qmix.py`, especially `SharedAgentQNetwork`, `QMixer`, `QMIX.__init__`). Tests: `tests/test_qmix.py` (9 test functions). Training entry point `scripts/train_qmix_highway.py` derives dimensions from the selected environment (lines 135-177); its documented use is Highway, not verified CARLA training.
- **VDN:** learner in `marl/vdn.py`, shared network from `marl/networks.py`; tests include toy API and general training paths. Trainer `scripts/train_toy.py` is also optionally configured for highway. No evidence of VDN trained on CARLA.
- **MAPPO:** `marl/mappo.py` is empty (0 bytes as visible in file listing/content); no implementation interface or dedicated MAPPO test exists. `docs/sota_A.md` lists MAPPO as literature only.
- **Hard-coded dimensions:** CARLA observation layout constants are 8+6+(6×6)+6=56 in `envs/obs_builder.py:10-16`; CARLA exposes that dimension in `envs/carla_env.py:48-86`. QMIX itself is dimension-generic. Highway training derives observation/state/action dimensions from env (`scripts/train_qmix_highway.py:135-177`). CARLA retraining still needs a CARLA-capable training route and valid simulator/reward pipeline; no successful run is evidenced. A network trained with 25-D weights cannot accept 56-D input without a new model (`docs/env_spec.md:182-188`; QMIX first layer input in `marl/qmix.py`).
- `results/qmix_highway/*` has learner checkpoints and training data, but its CSV explicitly marks success, TTC, jerk and V2X/selection metrics unavailable; earliest shown episodes collide (`collision_rate=1`). It cannot validate teammate-selection value. No tracked results demonstrate a trained CARLA policy.

## 5. Reward

- `RewardCalculator.DEFAULT_WEIGHTS` matches frozen contract §12.4: progress `+1`, collision `-10`, TTC `-0.5` below threshold `3s`, comfort `-0.05`, goal `+10`, idle `-0.1`, team collision `-10` (`envs/reward.py:22-31`; contract `docs/env_spec.md:172-178`).
- `compute_terms` returns named `progress`, `collision`, `safety`, `comfort`, `goal`, `idle` contributions (`envs/reward.py:76-102`); team reward is mean plus team collision penalty (`:138-153`). Tests: `tests/test_reward.py` has 20 test functions, including term/default and team-reward assertions (lines around 147-152 and 209).
- Comfort currently sums absolute acceleration and jerk under a single weight (`envs/reward.py:60-74`); spec prose says jerk and lateral acceleration (`docs/env_spec.md` §6). This is a code/spec detail mismatch; CARLA helper supplies longitudinal acceleration/jerk (`envs/reward_integration.py:10-57`). Highway reward inputs are adapter-derived. Toy passes only progress/collision/goal (`envs/toy_env.py:99-108`), so its reward does not exercise TTC/comfort/idle.

## 6. Highway adapter: 56-D layout and differences

`envs/obs_builder.py:10-16` defines 8 ego + 6 route + 36 candidate + 6 mask. `docs/highway_adapter.md` table specifies exact offsets:

| Block | Indices | Width | Contents |
|---|---:|---:|---|
| Ego | 0–7 | 8 | speed, longitudinal acceleration, heading error, lateral offset, lane index/count, distance to goal, progress fraction |
| Route | 8–13 | 6 | next waypoint ego-frame x/y, speed limit, intersection flag, light state, next-waypoint distance |
| Candidates | 14–49 | 36 | six rows `[rel_x, rel_y, rel_vx, rel_vy, rel_heading, valid]` |
| Action mask | 50–55 | 6 | keep, accelerate, decelerate, emergency brake, lane left, lane right |

This is the same block structure the CARLA builder consumes; highway `contract56` calls that shared builder (`envs/highway_env.py:12-18, 90-100`; `docs/highway_adapter.md`). Differences are simulator semantics: highway stand-ins use a lane point 10m ahead, no intersection/light, and optional longitudinal goal; no goal means distance defaults to `D_MAX` and progress fraction zero (`docs/highway_adapter.md`, “Highway stand-ins”). CARLA uses actual route/road/light context. Highway contract mode requests at least 12 vehicles at density 3.0 (`envs/highway_env.py:28-30, 42-65`).

## 7. Tests and static check

Top-level test function counts (from `Select-String` over each test file): `test_action_dispatch.py` 4; `test_carla_reward_state.py` 1; `test_check_obs_script.py` 1; `test_collision_monitor.py` 5; `test_contract.py` 9; `test_control.py` 4; `test_env_info.py` 4; `test_grouping.py` 0; `test_highway_adapter.py` 13; `test_metrics.py` 3; `test_obs_builder.py` 7; `test_qmix.py` 9; `test_reward.py` 20; `test_route_light_reset.py` 3; `test_script_termination_reasons.py` 2; `test_termination_logic.py` 11; `test_toy_env.py` 7. Total 106 function tests, excluding parameterized expansions.

Allowed command, using requested venv: `C:\Users\souro\carla-env\Scripts\python.exe -m pytest tests -q -k "not carla and not highway" --ignore=tests/test_highway_adapter.py` -> `92 passed, 4 skipped, 26 deselected, 2 errors` (24.80s). Two QMIX checkpoint tests errored before test execution because pytest could not scan `C:\Users\souro\AppData\Local\Temp\pytest-of-souro` (`PermissionError: [WinError 5] Access is denied`); cache warnings also cite denied access. This is an execution environment failure, not evidence of assertion failure. CARLA not started.

Allowed `python -m pyflakes envs scripts tests marl` with system Python -> `No module named pyflakes`. Retried using the provided venv (no install): one finding, `envs\highway_env.py:12:1: 'envs.obs_builder.V_MAX' imported but unused`.

## 8. Results folder

Tracked files below are assigned the commit shown by `git log -1 --format="%h %an %s" -- <path>`. “Validity” concerns claims as current experiment evidence.

| File | What it shows | Provenance / assessment |
|---|---|---|
| `api_carla.txt` | Historical two-agent API smoke output; says 56-D, state 10 | `cea5f8a`; STALE for current runtime; no run metadata/server state |
| `api_toy.txt` | Historical toy API output with scalar termination/truncation | `cea5f8a`; INVALID for current contract (contradicts current dict API) |
| `checkpoint_A.md` | Phase notes/checkpoint | `5c4a9f5`; historical context only, STALE |
| `repo_audit.md` | Earlier audit | `42889f7`/repo history; STALE relative to main tip |
| `test_env.txt`, `test_termination.txt` | Captured manual outputs | commits in `git log results`; STALE, not rerun in this audit |
| `train_log.txt`, `train_log2.txt`, `train_log3.txt` | Toy training attempts; logs show errors/interruption | `42889f7`/training history; INVALID as converged evidence |
| `vdn_toy_log.csv`, `vdn_toy_curve.png` | Toy VDN returns/curve | `0640753` lineage; VALID only as toy training artifact, not V2X/CARLA or selection evidence |
| `qmix_highway/highway_training_metrics.csv` | Highway QMIX metrics; includes collision rate and unavailable key metrics | `42889f7`; VALID as recorded highway run, INVALID for selection question or CARLA performance |
| `qmix_highway/highway_learning_curves.png` | Highway learning curves | `42889f7`; VALID only as a plot of that run, not selection evidence |
| `qmix_highway/training_stability.png` | Highway training stability plot | `42889f7`; same limitation |
| `qmix_highway/qmix_parameter_evolution.csv/.png` | QMIX parameter norms over episodes | `42889f7`; VALID diagnostic artifact, not task performance/selection evidence |
| `qmix_highway/qmix_highway.pt`, `vdn_highway.pt` | Highway checkpoints | file history/commit provenance UNVERIFIED (ignored `*.pt`); dimensions/run metadata not audited here; INVALID as proof of CARLA learning |
| Existing untracked `branch_audit_phase1.md`, `check_obs.txt`, `check_policies.txt`, `check_reward.txt`, `diagnose_stop.txt`, `keep_speed.png`, `qmix_run_guide.md` | User-provided artifacts, mixed diagnostics/docs/image | No Git commit; provenance UNVERIFIED. Text files include server connection refusal in `check_policies.txt`; not current reproducible results. |

All generated-looking results lack complete run revision/seed/config metadata in their files, so code revision linkage beyond tracked file commit history is UNVERIFIED. Current tracked image list: `results/vdn_toy_curve.png`, and under `results/qmix_highway/`: `highway_learning_curves.png`, `qmix_parameter_evolution.png`, `training_stability.png`. Untracked `results/keep_speed.png` also exists.

## 9. Docs and code inconsistencies

- `docs/env_spec.md` contains original proposal §§1–11 plus frozen §12. It says 25-D highway accepted and highway 56-D was Phase 1 work (`:182-188`), but main now contains the `contract56` adapter and docs; frozen wording is not updated. There is no 12.7.
- `docs/env_spec_section12.md` duplicates section 12; treat it as a copy, not evidence of an additional clause.
- `docs/sota_A.md` has VDN/QMIX/MAPPO rows and says CARLA results to be filled later. `docs/sota_B.md` explicitly frames selection claims as hypotheses/plans, not results. This matches absence of matched selection experiments.
- `docs/highway_adapter.md` documents 25-D default and shared 56-D option, boundaries, dimensions and stand-ins. `QMIX_CARLA_INTEGRATION.md` is an integration guide, not proof of integration/training success.
- Code gaps against contract: toy lacks `obs_dim`/`n_actions` and misnames `goal_reached`; section-12 frozen contract wording says all envs expose these (`docs/env_spec.md:140-168`, `envs/toy_env.py:13-22,112-122`). Section 12 claims last-step info includes scenario/town/seed; tests’ `REQUIRED_EPISODE_INFO` checks only success/min_ttc/jerk (`envs/base.py:41-43`).
- README and docs describe architecture/plans; they do not demonstrate end-to-end CARLA training. `envs/v2x_channel.py`/grouping implementations are not shown integrated into highway/CARLA policy learning or experiment runner: integration status UNVERIFIED from interfaces and `rg` call-sites; no selection metrics in highway CSV.

## 10. Phase conformance table (phases inferred from repository plan artifacts)

No single authoritative Phase 0–5 plan document was found in tracked `docs/`; labels below follow the user's A/B scope and `results/branch_audit_phase1.md`, which is untracked and cannot establish frozen project scope. Phase-by-phase mapping is therefore partly UNVERIFIED.

| Phase / owner | Status | Evidence |
|---|---|---|
| Phase 0 shared contract | Partial | `docs/env_spec.md` §12 and `envs/base.py`; toy deviates in attrs/info; contract tests exist |
| Phase 1 A: CARLA adapter, observation/control/termination/metrics/reward validation | Partial | CARLA implementation and unit tests exist; no verified live server or CARLA learning; logs show connection failure in user artifact |
| Phase 1 B: highway fixture + reward + 56-D adapter | Done for adapter implementation; Partial for validation | `envs/highway_env.py`, `docs/highway_adapter.md`, `tests/test_highway_adapter.py`; main pytest command excluded highway tests by user instruction |
| Phase 2 algorithms (VDN/QMIX; MAPPO) | Partial | VDN/QMIX implemented; MAPPO empty; highway checkpoints/logs only |
| Phase 3 V2X/grouping integration | Not started / integration UNVERIFIED | channel and selectors exist; no evidence of integration into policy/training comparison, no logged selections |
| Phase 4 controlled selection experiments | Not started | no radius/random/learned matched result table or selection metrics in CSV |
| Phase 5 CARLA retraining, evaluation and presentation analysis | Not started | no CARLA checkpoint/training log or valid CARLA evaluation in tracked results |

## 11. Presentation pack

### (a) Ten valid facts

1. Frozen CARLA observation width is 56 = 8 ego + 6 route + 36 candidate + 6 mask (`envs/obs_builder.py:10-16`).
2. Candidate input holds six rows of six values each (`envs/obs_builder.py:12-16`).
3. CARLA default is four agents and six actions; timeout is 300 decisions (`envs/carla_env.py:39-45`).
4. Decision rate is 5 Hz from `DECISION_DT_SECONDS` and CARLA timing constants (`envs/carla_env.py:39-45`; `envs/metrics.py` constants).
5. Highway adapter has four controlled CAVs and six actions (`envs/highway_env.py:21-22,98-104`).
6. Highway has a default 25-D layout and optional contract 56-D layout (`envs/highway_env.py:34-38,91-98`).
7. Four-agent highway global state has 20 values (`docs/highway_adapter.md`, “Training and dimensions”).
8. Reward defaults include progress +1, collision -10, TTC -0.5, comfort -0.05, goal +10, idle -0.1 (`envs/reward.py:22-31`).
9. Main graph contains six merged PR commits (#1–#6) (`git log --oneline --graph -n 25`).
10. The permitted non-CARLA pytest run completed 92 passes, 4 skips, 26 deselections, 2 environment errors; exact command/result in §7.

These facts describe repository implementation or checks, not a demonstrated teammate-selection benefit.

### (b) Plain-language description

Main contains a CARLA adapter with a shared 56-value observation builder, control/termination/metrics/reward helpers, a Highway-Env fixture with both native 25-value and contract-shaped 56-value observations, toy VDN and highway QMIX/VDN learning code, and radius/random/learned grouping modules. The current results demonstrate toy/highway development runs, while MAPPO is unimplemented and the repository does not show a completed CARLA training run or controlled comparison of teammate-selection strategies.

### (c) Architecture diagram description

Draw three environment boxes—Toy fixture, Highway-Env adapter (25-D or shared 56-D), and CARLA adapter (56-D)—feeding agent observations/masks to shared policies (VDN/QMIX; MAPPO placeholder). Draw `obs_builder`, `reward`, `metrics`, and `termination` as shared helpers called by adapters. Place V2X channel and radius/random/learned selectors beside the highway/CARLA observation path, with a dashed arrow labeled “implementation exists; experiment integration unverified” toward the policy. Show QMIX global state flowing only to its mixer during training. Add results/metrics output after the environment/policy loop. Cite code: `envs/base.py`, `envs/highway_env.py`, `envs/carla_env.py`, `marl/qmix.py`, `grouping/`, `envs/v2x_channel.py`.

### (d) Figures available

- `results/vdn_toy_curve.png`: toy VDN episode returns (valid only as toy artifact).
- `results/qmix_highway/highway_learning_curves.png`: highway learner curves.
- `results/qmix_highway/training_stability.png`: highway stability diagnostic.
- `results/qmix_highway/qmix_parameter_evolution.png`: parameter norms.
- `results/keep_speed.png` (untracked): image artifact, what it depicts is UNVERIFIED from filename alone.

### (e) Honest gaps and risks for supervisors

No evidence currently answers whether teammate selection adds value beyond group size. The necessary matched experiment, fixed-k selection logging, V2X impairment sweep, and CARLA training/evaluation are not evidenced. Highway logs mark success/TTC/comfort/V2X/selected teammate fields unavailable and show collision-heavy early episodes. The toy contract deviations and incomplete MAPPO reduce claim scope. Current pytest run has two temp-permission errors, while pyflakes reports one unused import. Avoid presenting highway or toy learning curves as CARLA safety evidence or as a teammate-selection result.

## 12. Main versus `feat/phase1-core`

There are no code differences to explain for the named feature branch: both tips share ancestry and `git log origin/main..origin/feat/phase1-core --oneline` is empty; `git diff origin/main...origin/feat/phase1-core --stat` is empty. The feature work visible in history—CARLA observation/control/termination, metrics/reward integration—was merged through PR #2 and PR #6. Branch-specific, unmerged changes: **none**.

## Questions for the owner

1. Can you provide the authoritative phase plan (Phase 0–5) and define completion criteria for each owner?
2. Should toy be brought into full section-12 conformance (`obs_dim`, `n_actions`, `goal_reached`, dict info), or formally excluded from the shared contract?
3. Which successful CARLA server run and checkpoint should be treated as the baseline; where are its seed/config/revision records?
4. Is MAPPO expected in this project, and what interface/test defines “done”?
5. Where is V2X/grouping selection integrated into action selection, and how will fixed group size and selected teammate IDs be verified in the planned comparison?
