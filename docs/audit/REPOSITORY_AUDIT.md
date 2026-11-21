# Repository Audit - V2X-MARL

- Date: 2026-11-16
- Branch / commit: `main` / `d4b5674` (`docs: V2X channel`)
- Scope: read-only audit of the current checkout. No source, test, config, or result file was modified. The only new files are under `docs/audit/`.
- Purpose: establish what exists, what is verified, what is missing, and what can honestly be claimed about the research question before the supervisor meeting.

## 1. Executive summary

The repository contains a substantial amount of engineering work, but it does **not** yet contain the experiment needed to answer the project's research question.

**What is real and verified in the current checkout**

1. A shared environment contract and a 56-D CARLA observation builder exist. The CARLA adapter implements reset, route planning, six meta-actions, collision monitoring, termination, per-agent reward, team reward, action masks, and global state (`envs/carla_env.py`).
2. VDN, QMIX, and MAPPO learners exist with masked action selection, replay/rollout buffers, checkpointing, normalization, and unit tests (`marl/`).
3. A pure-NumPy V2X channel exists with range, Bernoulli loss, fixed delay, counters, and conversion to the six-feature candidate layout (`envs/v2x_channel.py`, `docs/v2x_channel.md`). It is **not wired into any environment**.
4. Highway-Env trainers and plots exist, and a 25-D Highway QMIX checkpoint plus a small metrics CSV are present (`results/qmix_highway/`).
5. The non-CARLA, non-Highway test subset is green in the current Python environment: **164 passed, 14 skipped, 16 deselected** with the command in Section 8.

**What is missing or unverified**

1. **The research hook is empty.** `grouping/learned_sel.py`, `grouping/radius.py`, `grouping/random_sel.py`, `grouping/__init__.py`, and `tests/test_grouping.py` are all 0 bytes. There is no radius baseline, random baseline, learned selector, fixed-k selection, padding, or selected-teammate logging.
2. **V2X does not affect observations.** `CarlaMultiAgentEnv.get_candidates()` enumerates raw CARLA actors within 50 m; no channel, delay, loss, age, or intent is used. The V2X module is correct in isolation but is not part of the policy's information path.
3. **There is no CARLA 4-agent evidence at the current commit.** Historical live-server files exist (`results/api_carla.txt`, `results/test_env.txt`, `results/check_reward.txt`) but they use 2 agents, older commits, and ad-hoc scripts. No CARLA pytest run or 4-agent training run is present or reproducible in this environment.
4. **The Highway evidence cannot answer the selection question.** The checkpoint is 25-D (`obs_dim=25`, `state_dim=20`), while CARLA uses 56-D. The project's own `QMIX_CARLA_INTEGRATION.md` says the 25-D checkpoint is not directly loadable into a 56-D CARLA policy. The Highway CSV marks success, TTC, jerk, V2X, and selected-teammate metrics as `NA`, and collision rates are ~0.93-0.97.
5. **The claimed 15,000-step QMIX/VDN/MAPPO comparison is not in the checkout.** `docs/PROGRESS.md` describes `results/raw/comparison/` and `results/raw/plots/`, but `results/raw/` is absent. Those claims are historical and cannot be independently verified from this tree.
6. **Core experiment infrastructure is placeholder.** `configs/*.yaml`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/make_report.py`, `README.md`, and `docs/architecture.md` are empty.
7. **Scenario coverage is missing.** The CARLA reset spawns four CAVs on random Town03 routes but no scripted background vehicles and no intersection/merge/lane-change scenario. `scenario_id` is hard-coded to `"free_drive"`.
8. **The environment cannot currently run the advertised stack end-to-end.** `carla`, `highway_env`, and `gymnasium` are not installed. Python is 3.13.3 and Torch is CPU-only (`2.6.0+cpu`). CARLA 0.9.16 compatibility with Python 3.13 is not established.

**Bottom line:** this is a promising partial implementation with good unit-level engineering, but the repository cannot currently support a claim that learned teammate selection improves cooperative driving. The next milestone is not more algorithm work; it is a frozen selection/V2X interface, implemented grouping variants, a controlled CARLA experiment, and honest evidence logging.

## 2. Repository snapshot

| Item | Value |
|---|---|
| Working directory | `C:\Users\souro\Desktop\V2X-MARL` |
| Branch | `main` |
| HEAD | `d4b5674` - `docs: V2X channel` |
| Recent commits | `7272ff3` V2X implementation; `90ebded` V2X tests; `d4b5674` V2X docs; `f0c14d7` MAPPO results; `24ae44f` MAPPO trainer; `80bf389` MAPPO |
| Other branches | `Qmix` `42889f7`; `feat/env-and-vdn` `8132f06`; `feat/phase1-core` `e044753`; `test/qmix-on-carla` `d7d14b6` |
| Uncommitted state | No tracked modifications. Eight untracked files under `results/` (audits, diagnostics, one image). |
| Accidental root file | `t tests -q -k not carla and not highway` (876 bytes) |
| Test configuration | `pytest.ini` only; no lint/typecheck configuration found |
| Runtime | Python 3.13.3, NumPy 2.2.3, Torch 2.6.0+cpu, CUDA unavailable |

### Repository and documentation status

| Path | Size / status | Audit note |
|---|---|---|
| `envs/` | Implemented | Contract env, CARLA, Highway, Toy, obs, control, reward, termination, metrics, safety, V2X channel |
| `marl/` | Implemented | VDN, QMIX, MAPPO, buffers, networks, normalization |
| `grouping/` | **0 bytes** | All four files empty; research hook missing |
| `tests/` | Implemented except grouping | `tests/test_grouping.py` is 0 bytes |
| `scripts/` | Mostly implemented | `train.py`, `evaluate.py`, `make_report.py` are 0 bytes; Highway trainers and diagnostics exist |
| `configs/` | **0 bytes** | All four YAML files empty; no config-driven runs |
| `results/` | Mixed | Highway artifacts and historical logs; `results/raw/` absent |
| `docs/` | Partly implemented | Spec, V2X, Highway adapter, SOTA, progress; `architecture.md` empty |
| `README.md` | **0 bytes** | No setup or reproduction instructions |

## 3. Architecture as implemented

```text
                    +-----------------------------+
                    |        scripts/             |
                    |  train_qmix_highway.py      |
                    |  train_mappo_highway.py     |
                    |  train_common.py            |
                    |  check_obs.py               |
                    |  check_policies.py          |
                    +--------------+--------------+
                                   |
              +--------------------+--------------------+
              |                                         |
    +---------v----------+                   +----------v----------+
    | envs/highway_env   |                   | envs/carla_env      |
    | 25-D or 56-D       |                   | 56-D, raw candidates|
    +---------+----------+                   +----------+----------+
              |                                         |
              +------------------+----------------------+
                                 |
                    +------------v-------------+
                    | shared env helpers        |
                    | obs_builder, control,     |
                    | reward, termination,      |
                    | metrics, safety           |
                    +------------+--------------+
                                 |
                    +------------v-------------+
                    | marl/                     |
                    | VDN, QMIX, MAPPO,        |
                    | buffers, networks        |
                    +--------------------------+

    Intended but NOT connected:
      carla world/CAV state -> V2XChannel -> received messages
      -> candidate rows -> grouping (radius/random/learned, fixed k)
      -> policy observation
```

The design intent in `docs/env_spec.md` is correct: keep the environment generic and put the research in grouping, with a fixed group size `k=2` across radius, random, and learned selection. The implemented boundary currently stops before grouping and before the V2X channel is used.

## 4. Implementation matrix

Legend: **VERIFIED** = code exists and is exercised by tests or direct inspection; **PARTIAL** = some pieces exist but a required condition is missing or unverified; **NOT IMPLEMENTED** = placeholder or absent; **NOT VERIFIED** = cannot be checked in this environment.

| Component | Expected | Actual in current checkout | Status | Evidence |
|---|---|---|---|---|
| Frozen API | Dict API, attributes, `state()`, `action_masks()` | CARLA and Highway expose dict API and attributes; Toy exposes 3 actions/4-D obs as allowed by Contract 12 | VERIFIED | `envs/base.py`, `envs/carla_env.py`, `envs/highway_env.py`, `envs/toy_env.py` |
| CARLA reset | Seeded spawn, routes, TM seed, cleanup | Implemented; random 150 m route, TM path, 5 settling ticks, collision sensors after settling | PARTIAL | `envs/carla_env.py:283-431` |
| CARLA step | 4 ticks at 20 Hz -> 5 Hz decision | Implemented | VERIFIED by inspection; live 4-agent run NOT VERIFIED | `envs/carla_env.py:433-595` |
| 56-D observation | ego 8 + route 6 + candidates 36 + mask 6 | Implemented exactly | VERIFIED | `envs/obs_builder.py:10-16`, `84-175`; `tests/test_obs_builder.py` |
| Candidate source | Received V2X messages (spec 4) | Raw world actors within 50 m, sorted by distance/ID | **PARTIAL / CONTRACT RISK** | `envs/carla_env.py:702-741`; spec `docs/env_spec.md:38,55` |
| Action mask | Legal lane changes only | Implemented; invalid lane-change actions fall back to KEEP | VERIFIED | `envs/carla_env.py:443-453`, `963-979`; `envs/obs_builder.py:61-70` |
| Actions | 6 meta-actions | Implemented via TM target speed and force lane change | PARTIAL | `envs/carla_env.py:640-668`; TM behaviour not equivalent to a PID/local planner |
| Reward | Contract 12.4 weights and team reward | Implemented; weights match, team reward is mean + collision penalty | PARTIAL | `envs/reward.py:23-32,138-151`; `envs/reward_integration.py`; `envs/carla_env.py:523-565` |
| Reward comfort formula | `-|jerk|`, `-|lateral accel|` in spec 6 | Scalar `-0.05 * (|a_long| + |jerk|)`; no lateral acceleration | **PARTIAL / MISMATCH** | `envs/reward.py:69-79`; `docs/env_spec.md:75` |
| Gap safety penalty | Small penalty for gap < 5 m | Not implemented | NOT IMPLEMENTED | `docs/env_spec.md:74`; no corresponding code |
| Termination | Collision, goal, timeout, off-road, stuck | Implemented with priority collision > goal > off-road > stuck > timeout | PARTIAL | `envs/termination.py:80-101`; `envs/carla_env.py:489-520` |
| Per-agent termination | Team objective; all agents share episode end | `terminated[a] = result.terminated or self.reached[a]` can end a trainer episode when only the first agent reaches its goal | **BUG / CONTRACT RISK** | `envs/carla_env.py:566-569`; trainers break on `any_flag(terminated)` (`scripts/train_qmix_highway.py:358,420`) |
| Info keys | Required per-step and final keys | Required keys present; final adds success/min_ttc/jerk/town/seed; `scenario_id` hard-coded | PARTIAL | `envs/termination.py:104-180`; `envs/carla_env.py:570-594` |
| Metrics | Success, collision, travel time, speed, TTC, jerk, V2X, selected IDs | min TTC and jerk tracked; travel time and V2X/selection metrics absent from training CSV | **PARTIAL** | `envs/metrics.py`; `scripts/train_qmix_highway.py:36-49`; Highway CSV shows `NA` |
| V2X channel | Range, delay, loss, determinism, conversion | Implemented and tested in isolation | VERIFIED in isolation | `envs/v2x_channel.py`; `tests/test_v2x_channel.py`; `docs/v2x_channel.md` |
| V2X integration | Candidates come from received messages | None | **NOT IMPLEMENTED** | No import of `envs.v2x_channel` in `envs/carla_env.py` |
| Grouping radius | Deterministic radius baseline, fixed k | File is 0 bytes | **NOT IMPLEMENTED** | `grouping/radius.py` |
| Grouping random | Seeded random baseline, fixed k | File is 0 bytes | **NOT IMPLEMENTED** | `grouping/random_sel.py` |
| Grouping learned | Attention top-k selector | File is 0 bytes | **NOT IMPLEMENTED** | `grouping/learned_sel.py` |
| Fixed-k padding | Always k selected, pad/mask | Not implemented | **NOT IMPLEMENTED** | No grouping code |
| Selected-ID logging | Required for selection analysis | Not implemented | **NOT IMPLEMENTED** | CSV columns lack selected IDs |
| VDN | CTDE value decomposition | Implemented and unit-tested | VERIFIED | `marl/vdn.py`; `tests/test_vdn.py` |
| QMIX | Monotonic mixer, masked target max | Implemented and unit-tested | VERIFIED | `marl/qmix.py`; `tests/test_qmix.py`, `tests/test_qmix_correctness.py` |
| MAPPO | Masked actor, centralized critic, GAE/PPO | Implemented and unit-tested | VERIFIED | `marl/mappo.py`; `tests/test_mappo.py` |
| Highway adapter | 25-D or shared 56-D, 4 CAVs | Implemented; contract56 mode pads candidates and supports goal mode | PARTIAL | `envs/highway_env.py`; `docs/highway_adapter.md` |
| CARLA scenario setup | Intersection, merge, lane change, background traffic | Random routes only; no scripted background traffic; `scenario_id="free_drive"` | **NOT IMPLEMENTED** | `envs/carla_env.py:309-375`; `envs/termination.py:175` |
| Config-driven training | YAML configs, unified train/eval/report | Config files and `scripts/train.py`, `evaluate.py`, `make_report.py` empty | **NOT IMPLEMENTED** | `configs/`; `scripts/` |
| Reproducibility | Same seed -> same starts/routes/goals | Seeded spawn, route RNG, and TM seed exist; full CARLA determinism unverified | PARTIAL | `envs/carla_env.py:284-417` |
| Documentation | Setup, architecture, reproduction | README and architecture empty; spec and V2X docs good | PARTIAL | `README.md`; `docs/architecture.md` |

## 5. Phase completion matrix

| Phase | Intended outcome | Current status | What exists | Missing exit evidence |
|---|---|---|---|---|
| Phase 0 - Spec and contract | Frozen API, observation, reward, termination, reproducibility | **COMPLETE with caveats** | `docs/env_spec.md`, `docs/env_spec_section12.md`, `envs/base.py` | Remaining ambiguities: V2X vs candidates, delay units, loss timing, range normalization, k semantics |
| Phase 1 - Environment | CARLA 4-agent env, 56-D obs, reward, termination, scenarios, V2X | **PARTIAL** | CARLA env, obs builder, reward, termination, V2X module | No background traffic/scenarios; V2X not wired; no 4-agent live run at HEAD; determinism unverified |
| Phase 2 - MARL | VDN, QMIX, MAPPO, buffers, normalization, checkpoints | **MOSTLY COMPLETE** | Learners and unit tests | No CARLA training; Highway evidence only; no 56-D CARLA checkpoint |
| Phase 3 - V2X and grouping | Channel integrated, three selectors, fixed k, selected IDs | **PARTIAL / BLOCKED** | Channel implemented in isolation | Grouping empty; no integration; no fixed-k or selection logs |
| Phase 4 - Training and evaluation | Matched QMIX/VDN/MAPPO runs, held-out test, uncertainty | **PARTIAL / HISTORICAL** | Highway trainers and plots; PROGRESS claims 15k-step runs | `results/raw/` absent; no CARLA runs; no held-out test artifacts; no uncertainty report |
| Phase 5 - Research analysis | Answer selection-vs-group-size question | **NOT STARTED** | SOTA and planned design in `docs/sota_B.md` | No treatment variable, no controlled comparison, no evidence |

## 6. Experimental evidence audit

### 6.1 What the unit tests prove

The current non-CARLA/non-Highway subset proves deterministic, contract-level behavior of helpers and learners:

- observation layout and normalization;
- action masks and action dispatch;
- reward defaults and per-term arithmetic;
- termination priority and edge cases;
- V2X range/loss/delay/conversion semantics;
- VDN/QMIX/MAPPO masked action selection, target masking, truncation bootstrap, checkpoint round trips, and simple matrix-game convergence.

It does **not** prove live CARLA behavior, 4-agent coordination, scenario difficulty, V2X robustness, or selection quality.

### 6.2 CARLA evidence present in the tree

| Artifact | Content | What it proves | Limitation |
|---|---|---|---|
| `results/api_carla.txt` | Live CARLA reset/step API: agents `['cav_0','cav_1']`, obs `(56,)`, state `(10,)` | The adapter once ran against a live server with 2 agents | Not 4 agents; not current HEAD; no pytest; no selection/V2X |
| `results/test_env.txt` | 100 ticks OK; 2 vehicles after 10 resets; 5 episodes x 50 steps | Basic reset/cleanup behavior once worked | 2 agents; ad-hoc script; no contract assertions |
| `results/check_reward.txt` | Live server, commit `e044753`, seeds 0..9; keep return 3.02, crash -17.50; comfort share 57.3% | Reward integration produced plausible keep-vs-crash signals at an older commit | Not current HEAD; no 4-agent contract; comfort dominates 57.3% of reward magnitude; no statistical test |
| `results/check_obs.txt`, `check_policies.txt`, `diagnose_stop.txt` | Diagnostic output | Historical debugging context | Untracked; no controlled experiment |

### 6.3 Highway evidence present in the tree

`results/qmix_highway/qmix_highway.pt` was inspected directly:

- `algorithm=qmix`
- `n_agents=4`
- `obs_dim=25`
- `state_dim=20`
- `n_actions=6`
- `hidden_dim=128`, `mixing_dim=32`, `hypernet_dim=64`
- `gamma=0.99`, `tau=0.01`, `reward_scale=0.1`
- Optimizer state exists with step 465.

This is a valid Highway checkpoint, but it is **not CARLA-compatible**. `QMIX_CARLA_INTEGRATION.md` states that the 25-D Highway checkpoint cannot be loaded into a 56-D CARLA policy without a shared feature encoder/schema or retraining.

`results/qmix_highway/highway_training_metrics.csv` was inspected:

| Property | QMIX | VDN |
|---|---:|---:|
| Rows | 30 | 30 |
| Unique `seed` values | 0..29 | 0..29 |
| Mean episode return | 37.33 | 48.75 |
| Last-5 mean return | 17.02 | 110.95 |
| Mean collision rate | 0.967 | 0.933 |
| Success rate | `NA` | `NA` |
| Min TTC | `NA` | `NA` |
| Mean jerk | `NA` | `NA` |
| V2X received/dropped | `NA` | `NA` |
| Selected teammate IDs | `NA` | `NA` |

The CSV is not a replicated learning curve. It records 30 rows per algorithm with `seed` 0..29 and no run-level grouping; its schema also differs from the current trainer. It is historical, weakly specified evidence. It cannot support a selection claim and it cannot support a CARLA safety claim.

`results/qmix_highway/qmix_parameter_evolution.csv` shows identical parameter norms for the first six rows, consistent with updates starting only after warmup. That is not a defect, but it reinforces that this is a short/smoke-scale artifact, not a mature training run.

### 6.4 What `docs/PROGRESS.md` claims but the checkout cannot verify

`docs/PROGRESS.md` is detailed and internally consistent, but several claims depend on artifacts that are absent:

- Stage B claims six 15,000-step QMIX/VDN runs and `results/raw/comparison/` with 15 eval rows per run.
- Stage C claims 4,107 rows and four plots under `results/raw/plots/`.
- MAPPO Stage 2 claims five 15,000-step runs under `results/raw/comparison/`.
- Final QMIX/VDN eval claims QMIX return -90.94 with collision 1.00; VDN -113.58 with collision 0.67; random -24.68 with collision 1.00.

`results/raw/` is absent from the checkout. These are therefore **historical claims**, not reproducible current evidence. They also describe a crash-dominated Highway scenario, so even if restored they would not demonstrate competent driving or selection value.

### 6.5 Historical audits are stale

- `results/repo_audit.md` says `envs/v2x_channel.py` is an empty placeholder and rewards are zero. That is no longer true at HEAD: the V2X channel is implemented and CARLA reward integration exists.
- `results/main_audit_for_presentation.md` (older commit, 2026-11-09) says V2X/grouping integration has not started. The V2X module now exists, but grouping is still absent and V2X is still not integrated.
- `results/branch_audit_phase1.md` correctly identified the key boundary: candidates are raw simulator state, not V2X-delivered messages. That finding remains valid.

Use current code and this audit for decisions; use older audits only as history.

## 7. Environment and dependency audit

### 7.1 Runtime

| Item | Value |
|---|---|
| OS | Windows |
| Python | 3.13.3 |
| NumPy | 2.2.3 |
| Torch | 2.6.0+cpu |
| CUDA available | No |
| `torch.version.cuda` | None |
| pytest | 8.3.4 |

### 7.2 Package availability

| Package | Status | Consequence |
|---|---|---|
| `numpy` | Installed | Core env and learners run |
| `torch` | Installed, CPU-only | Learners run; training is slow; no GPU acceleration |
| `pytest` | Installed | Non-CARLA tests run |
| `matplotlib` | Installed | Plotting possible |
| `pandas` | Installed | Analysis possible, but not declared in `requirements.txt` |
| `yaml` | Installed | Config parsing possible, but configs are empty and PyYAML is not declared |
| `carla` | **Missing** | CARLA tests and live runs cannot start |
| `highway_env` | **Missing** | Highway adapter and training cannot run |
| `gymnasium` | **Missing** | `tests/test_highway_adapter.py` fails at collection time |
| `pytest-timeout` | Missing | Timeout flag unavailable; not required |

`requirements.txt` lists only `numpy`, `torch`, `gymnasium`, `highway-env`, `matplotlib`, and `pytest`. It omits CARLA (not pip-installable in the usual way), `pandas`, and PyYAML. It also does not record the CARLA version or Python compatibility.

### 7.3 CARLA compatibility risk

The project targets CARLA 0.9.16. The current Python is 3.13.3. CARLA's Python API is typically tied to older Python versions, and the `carla` module is absent. Therefore:

- the CARLA code path is **unverified in this environment**;
- the historical live-server logs cannot be reproduced without setting up a compatible CARLA environment;
- a dedicated environment (for example Python 3.10/3.11 with the matching CARLA wheel and server) should be documented before any CARLA experiment.

## 8. Test and verification audit

### 8.1 Commands executed

```powershell
python --version
python -c "import numpy, torch; ..."
python -c "import importlib.util as u; ..."
python -m pytest --version
python -m pytest tests -q -k "not carla"
python -m pytest tests -q -k "not carla and not highway"
python -m pytest tests -q --ignore=tests/test_highway_adapter.py -k "not carla"
python -m pytest tests -q -rs --ignore=tests/test_highway_adapter.py -k "not carla"
```

### 8.2 Results

| Command | Result |
|---|---|
| `pytest tests -q -k "not carla"` | Collection error: `tests/test_highway_adapter.py` imports `gymnasium` at module import; `-k` cannot deselect collection |
| `pytest tests -q -k "not carla and not highway"` | Same collection error |
| `pytest tests -q --ignore=tests/test_highway_adapter.py -k "not carla"` | **164 passed, 14 skipped, 16 deselected in 29.13 s** |
| Same with `-rs` | **164 passed, 14 skipped, 16 deselected in 24.68 s** |

Skipped breakdown:

- 5 Highway contract tests: `highway env not available (is highway-env installed?): No module named 'gymnasium'`
- 8 strict contract tests: `set CONTRACT_STRICT=1 to run`
- 1 QMIX Highway interaction test: `could not import 'gymnasium'`

The 16 deselected tests are the CARLA-named tests. No CARLA test ran because the `carla` module is missing and no server was used.

### 8.3 Test gaps

| Gap | Consequence |
|---|---|
| No `tests/test_grouping.py` content | No selection correctness, fixed-k, padding, determinism, or tie-breaking tests |
| No integrated V2X-environment test | Channel tests pass, but no test proves the env consumes received messages |
| No CARLA pytest at HEAD | 56-D CARLA contract is only tested through pure helpers |
| `tests/test_highway_adapter.py` fails collection without gymnasium | A standard `pytest tests` is not portable; dependency or skip handling needs fixing |
| No lint/typecheck command | Code style and type errors are not gated |
| No CI | Test results depend on the local machine |
| No end-to-end 4-agent CARLA smoke test | The main integration path is unverified |

### 8.4 Prior test claims versus current run

`docs/PROGRESS.md` claims 114 passed/17 skipped, then 136 passed/4 skipped, and V2X docs claim 173 passed/4 skipped/33 deselected. These are from different branch states and dependency sets. The current checkout, with CARLA/Highway missing, produces **164 passed, 14 skipped, 16 deselected** when the Highway adapter module is ignored. Both sets can be legitimate; they are not directly comparable.

## 9. Contract compliance matrix

Contract source: `docs/env_spec.md` section 12, which explicitly overrides sections 6 and 9 where they differ.

| Contract item | Requirement | Current status | Evidence / caveat |
|---|---|---|---|
| 12.1 API | `reset(seed)` -> `(obs, info)`; `step` -> 5-tuple; dicts keyed by agent; attributes and `state()`/`action_masks()` | PASS for CARLA/Highway by inspection; Toy uses 3 actions/4-D as allowed | `envs/carla_env.py`, `envs/highway_env.py`, `envs/toy_env.py` |
| 12.2 Collision termination | Any CAV collision terminates all agents | PASS | `envs/termination.py:80-81`; `envs/carla_env.py:566-569` |
| 12.2 Goal termination | All agents at goal, no collision -> all terminate, success | PARTIAL | Global goal termination works; per-agent `reached` also sets that agent's `terminated=True`, which can prematurely stop trainers that break on `any_flag` |
| 12.2 Timeout | 300 decision steps -> all truncated | PASS | `MAX_STEPS=300`; `termination.py:90-91` |
| 12.2 Stuck | > 10 s -> episode ends, `info["stuck"]` | PASS with threshold nuance | `STUCK_STEPS=50`, strict `>50`, grace 25; `envs/carla_env.py:63-65,597-623` |
| 12.2 Off-road | Off-road or stuck > 10 s -> agent flagged, episode ends | PARTIAL | Off-road limit is 51 decision steps (10.2 s), not exactly 10 s; priority is after goal |
| 12.3 Per-step info | `collision`, `goal_reached`, `steps` | PASS | `envs/termination.py:141-152` |
| 12.3 Final info | `success`, `min_ttc`, `jerk`, `scenario_id`, `town`, `seed` | PARTIAL | Keys exist; `scenario_id` is always `"free_drive"` |
| 12.4 Reward weights | Progress 1.0, collision -10, TTC -0.5, comfort -0.05, goal +10, idle -0.1 | PASS for weights | `envs/reward.py:23-32` |
| 12.4 Team reward | Mean per-agent reward + -10 team collision | PASS | `envs/reward.py:138-151`; `envs/carla_env.py:563-565` |
| 12.4 Finished agent | Frozen, no further progress reward | PASS | `envs/carla_env.py:525-531`; `reward_integration.py:27-30` |
| 12.5 CARLA obs | 56-D | PASS | `envs/obs_builder.py:10-16`; `tests/test_obs_builder.py` |
| 12.5 Highway obs | 25-D accepted; 56-D adapter is Phase 1 | PASS | `docs/highway_adapter.md`; `envs/highway_env.py` |
| 12.6 Reproducibility | Same seed -> same starts/routes/goals | PARTIAL | Seeding exists; CARLA physics/TM determinism not verified |
| Spec 4 V2X | Candidate set comes from received messages; range/delay/loss affect grouping | **FAIL** | Raw actor candidates; channel unused |
| Spec 5 Grouping | Radius/random/learned, fixed k=2 | **FAIL** | All grouping files empty |
| Spec 8 Scenarios | Intersection, merge, lane change; train/test towns | **FAIL** | Random Town03 routes only; `scenario_id` hard-coded |
| Spec 10 Metrics | Success, collision, travel time, speed, TTC, jerk, V2X, selected IDs | **FAIL / PARTIAL** | Some metrics exist; V2X, selected IDs, and travel time are not in the training CSV |

### 9.1 Specific defects and risks found by inspection

1. **Early team termination in trainers (high impact).** In `envs/carla_env.py:566-569`, a single agent reaching its goal produces `terminated[a]=True`. Both Highway trainers break on `any_flag(terminated)` (`scripts/train_qmix_highway.py:144,420`; `scripts/train_mappo_highway.py:133,319`). A CARLA training/eval episode can therefore stop when the first CAV reaches its goal, not when all CAVs do. This biases success and return metrics.
2. **Stale CARLA TODO.** `envs/carla_env.py:9-10` says reward integration is TODO, but the code imports and uses `RewardCalculator` and `compute_agent_step_reward` (`35-36`, `523-565`). The comment is misleading; update it.
3. **Reward comfort mismatch.** The spec describes `-|jerk|` and `-|lateral acceleration|`; the implementation uses `-0.05 * (|a_long| + |jerk|)` and has no lateral acceleration. The historical `check_reward.txt` reports comfort at 57.3% of total reward magnitude, so this term dominates behavior.
4. **Missing gap penalty.** The spec mentions a small penalty for gap < 5 m; it is not implemented. Safety reward is TTC-only.
5. **Candidate/V2X boundary is the central research risk.** Raw world-state candidates mean range, delay, and loss cannot affect the policy. Any "robustness" claim under the current adapter would be false.
6. **No background traffic in CARLA.** Reset spawns only the CAVs. The candidate set may be empty or contain only other CAVs, so the six-candidate observation is not representative of the specified scenario.
7. **No scenario diversity.** All routes are random 150 m paths; there is no intersection, merge, or dense lane-change scenario, and `scenario_id="free_drive"`.
8. **Model/environment mismatch.** The only checkpoint is 25-D Highway; CARLA is 56-D. The integration guide explicitly warns against silently padding/truncating.
9. **Test portability.** `tests/test_highway_adapter.py` imports `gymnasium` at module level, so `pytest tests` fails collection instead of skipping cleanly.
10. **Empty entry points.** `scripts/train.py`, `scripts/evaluate.py`, `scripts/make_report.py`, and all configs are empty, so there is no single reproducible command for the full experiment.

## 10. Research design audit

### 10.1 Research question

> Does learned teammate selection bring measurable benefits beyond group size in cooperative driving?

This requires a controlled comparison in which **only the selection strategy changes**, while group size, candidate set, policy, reward, traffic, seeds, scenario, and communication budget are held constant.

### 10.2 Required design

| Element | Requirement |
|---|---|
| Treatment | Learned attention top-k vs radius vs random selection |
| Control | Fixed `k=2` for all strategies; same `M=6` candidate slots |
| Information | Candidates derived from received V2X messages with range/delay/loss conditions |
| Policy | Same algorithm and architecture across strategies; shared training budget |
| Scenarios | Intersection, merge, lane change; train/test town split |
| Seeds | Paired seeds across strategies; enough seeds for uncertainty estimates |
| Evaluation | Held-out test seeds/towns, no tuning on test |
| Outcomes | Collision rate, success rate, travel time, speed, min TTC, jerk, V2X received/dropped, selected IDs |
| Logging | Selected teammate IDs, group composition, communication counters, per-term rewards |
| Analysis | Mean, variance/CI, paired comparisons, ablation of k and communication conditions |

### 10.3 Current gaps that prevent a valid answer

| Gap | Why it invalidates the comparison | Severity |
|---|---|---|
| No grouping implementation | There is no treatment variable | Critical |
| No fixed-k/padding | Group size and selection quality are confounded | Critical |
| V2X not integrated | Range/delay/loss cannot affect observations; robustness cannot be measured | Critical |
| Raw ground-truth candidates | Leaks perfect state; no communication model | Critical |
| No selected-ID logging | Cannot audit what the selector chose | High |
| No CARLA 4-agent run at HEAD | No environment-level evidence | Critical |
| Highway 25-D checkpoint | Not transferable to CARLA 56-D | High |
| Crash-dominated Highway scenario | Returns mostly reflect collision/survival, not selection | High |
| Early termination on first goal | Biases episode completion metrics | High |
| No scenario diversity | Results may not generalize across the specified scenarios | High |
| No held-out test artifacts | No generalization evidence | Critical |
| No uncertainty analysis | Cannot distinguish signal from seed noise | High |
| No communication overhead metric | Cannot test the safety/overhead trade-off in H3 | Medium |
| Reward comfort/gap mismatch | Safety/comfort conclusions may not match the frozen spec | Medium |

### 10.4 What can be claimed now

Safe claims:

- "We implemented a 56-D CARLA observation/action/reward/termination stack and a pure-NumPy V2X channel."
- "We implemented and unit-tested VDN, QMIX, and MAPPO learners with masked actions and CTDE."
- "The non-CARLA unit-test subset passes in the current environment."
- "A 25-D Highway QMIX checkpoint exists, but it is not CARLA-compatible."
- "We have not yet measured whether teammate selection improves cooperative driving."

Unsafe claims:

- "QMIX learns safe driving in CARLA."
- "Learned selection outperforms radius or random selection."
- "The V2X channel provides robustness to delay and packet loss."
- "The Highway results generalize to CARLA."
- "The full test suite passes" without stating that CARLA/Highway tests were skipped or deselected.
- "The 15,000-step comparison was completed" without restoring `results/raw/` and its metadata.

### 10.5 Minimum viable experiment

A defensible minimum experiment is:

1. Freeze the candidate/V2X/selection interface (Section 11 and `V2X_INTEGRATION_DECISIONS.md`).
2. Implement `radius`, `random`, and `learned` selectors with `k=2`, deterministic tie-breaking, and zero padding.
3. Feed each selector's selected candidates into the same 56-D observation path.
4. Integrate `V2XChannel` into CARLA candidates and log received/dropped/age.
5. Run paired seeds across strategies with identical reward, traffic, scenarios, and step budget.
6. Evaluate on held-out seeds/towns with collision, success, travel time, TTC, jerk, and communication overhead.
7. Report mean and 95% CI, and explicitly test whether differences exceed seed noise.

## 11. V2X readiness

| Question | Current answer |
|---|---|
| Does a V2X model exist? | Yes, `envs/v2x_channel.py`, pure NumPy, tested in isolation. |
| Does it support range? | Yes, `range_m=50` default; 30/100 are the intended robustness values. |
| Does it support delay? | Yes, `delay_steps` in decision steps; 0 default, 1-3 intended. |
| Does it support loss? | Yes, independent Bernoulli per directed transmission; 0 default, 0.1-0.3 intended. |
| Does it carry intent? | Yes, message includes `intent` (action id). |
| Is intent used by observations? | No. `obs_builder` has six candidate features and no intent column. |
| Are candidates produced from received messages? | No. CARLA uses raw world actors. |
| Are V2X metrics logged? | No. The channel has counters, but the env/trainer does not expose them. |
| Is the channel stepped exactly once per decision? | Not integrated; the documented contract requires once per decision step. |
| Are background vehicles broadcasters? | Undecided; the channel supports either, and the caller must decide. |

The detailed ten-decision matrix and recommended interface are in `docs/audit/V2X_INTEGRATION_DECISIONS.md`.

## 12. Technical debt and risks

| ID | Risk | Severity | Likelihood | Impact | Mitigation |
|---|---|---|---|---|---|
| R1 | Grouping package empty | Critical | Certain | No research treatment | Implement selectors + tests before any experiment |
| R2 | V2X not integrated | Critical | Certain | No communication robustness claim | Wire channel into candidate generation; log stats |
| R3 | No 4-agent CARLA evidence at HEAD | Critical | Certain | No environment-level result | Set up CARLA env; run 4-agent smoke + contract tests |
| R4 | 25-D checkpoint vs 56-D CARLA | High | Certain | Cannot deploy Highway policy in CARLA | Retrain on 56-D or define a shared encoder; do not pad silently |
| R5 | Early termination on first goal | High | Certain | Biased success/return | Fix per-agent/team termination semantics; add a test |
| R6 | Crash-dominated Highway scenario | High | Certain | Learning signal dominated by collisions | Improve scenario/traffic balance; report collision rate |
| R7 | No background traffic/scenarios | High | Certain | Candidate set and generalization unrepresentative | Add scripted traffic and scenario definitions |
| R8 | Missing dependencies / Python mismatch | High | Certain | Cannot reproduce CARLA/Highway runs | Document and pin a compatible environment |
| R9 | `results/raw/` absent | High | Certain | Historical claims unverifiable | Restore artifacts or rerun with fixed seeds |
| R10 | Comfort reward dominates | Medium | Likely | Policy may optimize comfort over progress/safety | Rebalance/freeze weights; log per-term shares |
| R11 | No selected-ID/V2X metrics | High | Certain | Cannot answer selection or overhead questions | Add info fields and CSV columns |
| R12 | No uncertainty analysis | High | Likely | Cannot distinguish effect from noise | Multiple seeds, paired tests, 95% CI |
| R13 | Empty config/entry-point files | Medium | Certain | No reproducible experiment command | Implement configs and unified train/eval/report |
| R14 | Test collection portability | Medium | Certain | CI/`pytest tests` fails without gymnasium | Use `pytest.importorskip` or module-level guard |
| R15 | Stale docs/audits | Medium | Certain | Confusion about current state | Date-stamp docs; keep this audit as current baseline |
| R16 | Accidental root file | Low | Certain | Repository hygiene | Remove after confirming it is not needed |

## 13. Prioritized recommendations

### P0 - required before any research claim

1. **Freeze the V2X/selection interface.** Resolve the ten decisions in `V2X_INTEGRATION_DECISIONS.md`; record the approved version in the spec before coding.
2. **Implement grouping.** Radius, random, learned top-k; fixed `k=2`; deterministic tie-breaking; zero padding and mask; unit tests for all three.
3. **Integrate V2X into CARLA candidates.** Step the channel once per decision; convert received messages to candidate rows; expose received/dropped/age; keep raw candidates only as an explicit debug mode.
4. **Implement scenarios and background traffic.** At minimum: unsignalized intersection, merge, dense lane change; set a real `scenario_id`; seed spawn/routes/goals.
5. **Fix contract defects.** Correct per-agent/team termination behavior; align comfort/gap reward with the frozen spec; document any intentional deviation.
6. **Restore a runnable environment.** Pin Python/CARLA, install gymnasium/highway-env, run 4-agent CARLA smoke, and run the full test suite without collection errors.
7. **Build the matched experiment.** Same policy, reward, seeds, traffic, budget; only selector changes; log selected IDs and communication counters.
8. **Evaluate on held-out seeds/towns.** Report mean and 95% CI; compare against random and keep-speed baselines; do not tune on test.

### P1 - required for a credible submission

9. Implement `configs/env.yaml`, `train.yaml`, `eval.yaml`, `scenarios.yaml` and unified `scripts/train.py`, `evaluate.py`, `make_report.py`.
10. Add a README with setup, exact commands, dependency versions, and artifact locations.
11. Add tests for the integrated channel-selector-observation path, fixed-k, and selected-ID logging.
12. Restore or rerun the `results/raw/` comparison and keep metadata with the artifacts.
13. Add communication-overhead and per-term reward reporting.

### P2 - polish

14. Clean stale TODOs, old audit references, and the accidental root file.
15. Add CI or a local `make check`/PowerShell script for tests and lint.
16. Improve plots and the final report once the experiment is valid.

## Appendix A - Evidence commands and observations

```powershell
# Repository state
git log --oneline -12
git status --short
git diff --stat

# Runtime and packages
python --version
python -c "import numpy, torch; print(numpy.__version__, torch.__version__, torch.cuda.is_available())"
python -c "import importlib.util as u; [print(m, bool(u.find_spec(m))) for m in ['carla','highway_env','gymnasium','pytest','matplotlib','pandas','yaml']]"

# Test suite (current environment, no CARLA/Highway)
python -m pytest tests -q --ignore=tests/test_highway_adapter.py -k "not carla"
# 164 passed, 14 skipped, 16 deselected

# Checkpoint dimensions
python -c "import torch; c=torch.load(r'results/qmix_highway/qmix_highway.pt', map_location='cpu', weights_only=False); print({k:c[k] for k in ['n_agents','obs_dim','state_dim','n_actions','hidden_dim','mixing_dim','hypernet_dim','gamma','tau','reward_scale']})"
```

## Appendix B - Key file map

| File | Why it matters |
|---|---|
| `docs/env_spec.md` | Full proposal and frozen Contract 12 |
| `docs/env_spec_section12.md` | Short frozen contract |
| `envs/base.py` | Agent ids, feature counts, action enum, required info keys |
| `envs/carla_env.py` | CARLA adapter; reward integration, routes, termination, candidates |
| `envs/obs_builder.py` | 56-D observation layout and normalization |
| `envs/reward.py` | Reward weights and team reward |
| `envs/reward_integration.py` | CARLA reward adapter and clamps |
| `envs/termination.py` | Termination priority and info schema |
| `envs/metrics.py` | TTC, jerk, longitudinal dynamics |
| `envs/safety.py` | Collision monitor |
| `envs/v2x_channel.py` | V2X range/loss/delay model |
| `envs/highway_env.py` | Highway development fixture |
| `grouping/*.py` | Empty research hook |
| `marl/vdn.py`, `marl/qmix.py`, `marl/mappo.py` | Learners |
| `scripts/train_qmix_highway.py`, `scripts/train_mappo_highway.py` | Existing trainers |
| `docs/PROGRESS.md` | Historical stage claims |
| `docs/v2x_channel.md` | V2X design and open questions |
| `QMIX_CARLA_INTEGRATION.md` | 25-D vs 56-D transfer warning |
| `results/qmix_highway/` | Only present learner checkpoint and small metrics CSV |
| `results/main_audit_for_presentation.md` | Older audit; partially stale |
