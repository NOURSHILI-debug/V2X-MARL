# V2X-MARL Repository Audit

**Audit date:** 2026-11-03  
**Scope:** Current `feat/env-and-vdn` worktree, visible local/remote refs, and checked-in history. Source claims cite line-numbered files; Git-state claims cite the command and captured output described below. No CARLA server was started, no packages were installed, and no CARLA tests were run. The only repository file created for this audit is this report.

## 1. Repo map and ownership

The inventory is `git ls-files` at the audit HEAD, plus the separately listed untracked files in §2. Directories/files with zero bytes are explicitly marked. Ownership is inferred from the last substantive commit author and subject, not from an unverified project-role assumption. Primary attribution command: `git log --format="%h %an <%ae> %s" --all -- <path>`.

| Path | Purpose / observed state | Last substantive author evidence |
|---|---|---|
| `configs/` | Four empty YAML placeholders: `env.yaml`, `eval.yaml`, `scenarios.yaml`, `train.yaml`; no configured experiment. | Git tree reports all four blobs as 0 bytes. |
| `docs/architecture.md` | Empty architecture-doc placeholder. | 0-byte blob. |
| [`docs/env_spec.md`](../docs/env_spec.md#L1) | Spec v1, sections 1–11; ends by saying the CARLA skeleton can be written. No section 12. | A-owned proposal; file history is in Git. |
| [`docs/sota_A.md`](../docs/sota_A.md#L1) | A’s VDN/QMIX/MAPPO comparison table; CARLA results remain “Fill later.” | `cea5f8a` / A. |
| [`docs/sota_B.md`](../docs/sota_B.md#L1) | B’s literature review and proposed hypotheses; explicitly says there is no experimental system/results yet. | B’s `NOURSHILI-debug` commits. |
| [`envs/base.py`](../envs/base.py#L15) | Generic API, actions, candidate-block helper, required info constants. | `624077c` / A. |
| [`envs/carla_env.py`](../envs/carla_env.py#L43) | CARLA adapter: spawn, TM autopilot, routes, collision/termination; live worktree is modified and uncommitted. | `624077c` / A; current edits are not committed. |
| [`envs/highway_env.py`](../envs/highway_env.py#L1) | Highway-Env adapter for four vehicles; 6-action translation, native Kinematics observations. | `1c8dc44` / B. |
| [`envs/toy_env.py`](../envs/toy_env.py#L7) | Two-vehicle toy intersection. | `1c8dc44` / B; the VDN CSV was produced by an older version (see §6–7). |
| [`envs/reward.py`](../envs/reward.py#L28) | Configurable per-agent progress/collision/TTC/comfort/goal calculator; no idle or team term. | `296b80c` / B. |
| [`envs/safety.py`](../envs/safety.py#L12) | CARLA collision-sensor monitor; callback tests use fakes. | `296b80c` / B. |
| `envs/v2x_channel.py` | Empty V2X placeholder. | Ownership of empty placeholder is UNVERIFIED. |
| `envs/scenarios/__init__.py` | Empty package marker; no scenarios implemented. | 0-byte blob. |
| `grouping/` | `learned_sel.py`, `radius.py`, `random_sel.py`, `__init__.py` are all empty; no selector implementation. | Ownership of empty placeholders is UNVERIFIED. |
| [`marl/vdn.py`](../marl/vdn.py#L10) | VDN learner. | `cea5f8a` / A. |
| [`marl/networks.py`](../marl/networks.py#L6) | Shared per-agent Q network. | A’s VDN implementation history. |
| [`marl/buffers.py`](../marl/buffers.py#L6) | VDN replay buffer. | A’s VDN implementation history. |
| `marl/qmix.py`, `marl/mappo.py` | Both empty on current branch. A non-empty QMIX implementation exists only on `origin/Qmix`; MAPPO is empty there too. | QMIX implementation: `42889f7` / B on `origin/Qmix`; MAPPO authorship UNVERIFIED. |
| [`scripts/train_toy.py`](../scripts/train_toy.py#L1) | Toy VDN training, random baseline, fixed-seed evaluation, CSV/plot/checkpoint output. | `cea5f8a` / A. |
| [`scripts/demo_carla.py`](../scripts/demo_carla.py#L1) | Scripted CARLA visual demonstration, not an evaluation baseline. | A’s CARLA/demo history. |
| [`scripts/check_api.py`](../scripts/check_api.py#L1) | Prints env reset/step/state/mask shapes. | A’s API-check history. |
| [`scripts/test_env.py`](../scripts/test_env.py#L1) | Manual CARLA tick/reset/random-action smoke script; requires an already-running server. | A’s CARLA test history. |
| [`scripts/test_highway.py`](../scripts/test_highway.py#L1) | Manual Highway-Env visual/headless rollout. | `1c8dc44` / B. |
| `scripts/train.py`, `scripts/evaluate.py`, `scripts/make_report.py` | Empty runner/evaluation/report placeholders. | Git tree reports 0-byte blobs. |
| [`tests/test_contract.py`](../tests/test_contract.py#L1) | Parametric toy/highway/CARLA API tests, with opt-in strict attributes/seeding/info checks; live worktree is modified. | `624077c` / A; current edits are uncommitted. |
| [`tests/test_toy_env.py`](../tests/test_toy_env.py#L1) | Toy reset, movement, action, state, and collision checks. | B’s toy-env history. |
| [`tests/test_reward.py`](../tests/test_reward.py#L1) | Reward calculator terms/input validation. | B’s reward history. |
| [`tests/test_collision_monitor.py`](../tests/test_collision_monitor.py#L1) | Sensor monitor tests using fake world/sensor objects; not CARLA integration. | B’s collision-monitor history. |
| `tests/test_grouping.py` | Empty; no selector tests. | 0-byte blob. |
| `results/` | Historical API/smoke/training logs, VDN CSV/PNG/checkpoint. See §9. | `vdn_toy_log.csv` and training artifacts introduced in `cea5f8a` / A. |
| `README.md`, `requirements.txt` | Both empty; no usage/dependency guide. | Git tree reports 0 bytes. |

There is no `core/` directory and no MDrive implementation in the tracked tree. Candidate formatting exists in [`envs/base.py`](../envs/base.py#L15), but the grouping and V2X modules are empty.

## 2. Git state

**Commands checked:** `git branch -a`, `git branch -vv`, `git log --oneline --graph --all -n 40`, `git status --short --branch`, `git diff origin/main --stat`, `git rev-list --left-right --count origin/feat/env-and-vdn...HEAD`, `git log origin/feat/env-and-vdn..HEAD`, `git merge-base HEAD origin/main`, and `git branch -r --contains <commit>`.

| Ref | Tip / status |
|---|---|
| `feat/env-and-vdn` (checked out) | `7d9cc01`, tracks `origin/feat/env-and-vdn`, reported ahead 4. |
| `main` | `b417608`, reported behind `origin/main` by 8. |
| `origin/feat/env-and-vdn` | `624077c`. |
| `origin/main` | `1c8dc44`. |
| `origin/Qmix` | `42889f7`. |

The local merge of `origin/main` is confirmed: `HEAD` is merge commit `7d9cc01`, whose parents are `624077c` and `1c8dc44`; `git merge-base HEAD origin/main` returns `1c8dc44`. The merge commit itself is not contained by any currently visible remote-tracking branch, so it is **not pushed according to the local refs**. The four commits in `origin/feat/env-and-vdn..HEAD` are `7d9cc01`, `1c8dc44`, `d0c99b7`, `ccaa230`; the latter three are reachable through other visible remote refs (`origin/main`/`origin/Qmix`). No fetch was performed, so whether the server’s refs changed since the last fetch is UNVERIFIED.

Worktree status at audit start and after the test runs:

```text
## feat/env-and-vdn...origin/feat/env-and-vdn [ahead 4]
 M envs/carla_env.py
 M tests/test_contract.py
?? assert
?? dict[str
?? dir
?? findstr
?? git
?? pytest
?? results/train_log2.txt
?? set
?? tatus
```

I did not alter these pre-existing changes/files. `git diff origin/main --stat` reports `envs/base.py` +72, `envs/carla_env.py` +126/-20, `tests/test_contract.py` +175 in the branch comparison; this is a branch diff, not a clean-tree diff. Current local modifications are in the two paths shown above.

`.gitignore` covers `.venv/`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.coverage`, and only `results/raw/*` (with `.gitkeep` exception). It does **not** cover `*.pt` or all of `results/`; `results/vdn_toy.pt` is tracked, and `results/train_log2.txt` is untracked. `__pycache__` is ignored. There is no `pytest.ini`; `docs/env_spec_section12.md` is absent, and section 12 is not in `docs/env_spec.md`.

Largest blobs found by `git rev-list --objects --all` plus `git cat-file -s`:

| Bytes | Path in history | Note |
|---:|---|---|
| 603,367 | `results/qmix_highway/qmix_highway.pt` | In remote/history, not current worktree. |
| 227,059 | `results/qmix_highway/highway_learning_curves.png` | In remote/history. |
| 134,850 | `results/vdn_toy_curve.png` | Current tracked result. |
| 87,685 | `results/qmix_highway/vdn_highway.pt` | In remote/history. |
| 73,480 | `results/vdn_toy.pt` | Current tracked checkpoint. |
| 69,412 | `results/vdn_toy_log.csv` | Current tracked log. |
| 48,239 | `results/qmix_highway/training_stability.png` | In remote/history. |
| 39,362 | `results/qmix_highway/qmix_parameter_evolution.png` | In remote/history. |
| 16,907 | `marl/qmix.py` | QMIX source blob on `origin/Qmix`; current branch file is empty. |
| 15,793 | `docs/sota_B.md` | Literature review. |

## 3. Contract compliance table

Contract reference: [spec §9](../docs/env_spec.md#L102), [base API and required info](../envs/base.py#L42). “Pass” means source/runtime contract shape, not semantic completion.

| Env | reset / step / termination | dimensions, agents, actions | state / masks | info and result |
|---|---|---|---|---|
| `envs/base.py` | Abstract declarations only: `reset(seed)->(obs,info)`, `step()->5-tuple`, dict-style completion. | `agents`, `obs_dim`, `n_actions` are annotations, not initialized values; six canonical actions. | Abstract `state()` and `action_masks()`. | Declared required step keys are `collision`, `goal_reached`, `steps`; episode keys `success`, `min_ttc`, `jerk`. This differs from spec §9’s plural `collisions` and its `ttc_min`, `scenario_id`, `town`, `seed`. Not a runnable env. [base.py](../envs/base.py#L30) |
| `ToyEnv` | Reset returns `(obs,{})`; step returns five items; `terminated` and `truncated` are dicts of bools. | **Fail strict:** 2 agents, 4-D observation, 3 actions; `obs_dims` typo, no `obs_dim`/`n_actions`. [toy_env.py](../envs/toy_env.py#L15) | `state()` and 3-wide bool masks exist. | **Fail strict:** `collision` and `steps` exist, but key is `reached_goal` (array), not required scalar `goal_reached`; episode `success`, `min_ttc`, `jerk` absent. Strict test reproduced both the missing `obs_dim` and info-key failures. [toy_env.py](../envs/toy_env.py#L146) |
| `CarlaMultiAgentEnv` (live worktree) | Reset returns `(obs,info)`; step returns five items; term/trunc dicts of bools. | **Declared pass:** default 4 agents, `obs_dim=56`, `n_actions=6`. | `state()` exists; masks have six bools but are all `True`, so lane-change validity is not enforced. | Step `collision`, `goal_reached`, `steps` exist; end info adds `success`, but omits `min_ttc` and `jerk`; no `scenario_id`. Shape passes, observation content does not. [carla_env.py](../envs/carla_env.py#L24) |
| `HighwayV2XEnv` | Reset/step return dictionary-style observations/rewards/termination/truncation; dict termination values are bools. | 4 agent IDs and six actions. No `obs_dim` or `n_actions` instance attributes. Exact native Kinematics vector width is **UNVERIFIED** (not declared; native feature/default settings are not overridden). | `state()` and lane-adjacency masks exist. | Info is nested by agent and uses `crashed`, not required `collision`; no `goal_reached`, `success`, TTC, jerk, town/scenario keys. [highway_env.py](../envs/highway_env.py#L75) |

The checked-in contract test’s strict mode is disabled by default; its strict checks cover attributes, seed behavior, and info keys. Results and exact failures are in §8.

## 4. CARLA environment deep read

- **World/sync:** Connects with a 30-second client timeout, loads the requested town, enables synchronous world mode with `fixed_delta_seconds=0.05`, and enables TM sync. It seeds TM and Python route/spawn selection. [carla_env.py](../envs/carla_env.py#L54)
- **Spawning:** Shuffles map spawn points with the environment seed; tries a Tesla Model 3 at each until `n_agents` spawn; enables TM autopilot, disables automatic lane changes, and initializes a 30 km/h target. No background traffic is spawned. If too few points succeed, an assertion fires after partial spawning and does not itself clean up those actors. [carla_env.py](../envs/carla_env.py#L135)
- **Routes/goals:** Plans a lane-following chain of 5 m waypoints to roughly 150 m, choosing randomly (seeded) among each waypoint’s `next()` results; the last waypoint location becomes the goal. The route logic does not enforce scenario-specific paths or a train/test town split. [carla_env.py](../envs/carla_env.py#L103)
- **`tm.set_path` risk:** Sends every fourth route point plus the final point to `tm.set_path`; catches `AttributeError`/`RuntimeError` and silently continues, leaving TM autopilot free to choose its own path. There is no route-success flag or observed-path assertion. CARLA 0.9.16 behavior was not verified because no server test was run. [carla_env.py](../envs/carla_env.py#L114)
- **Decision rate/actions:** Each `step` performs four world ticks (20 Hz × 4 => 5 Hz decisions). Actions alter TM target speed by 5 km/h up/down, set it to zero for brake, or call `force_lane_change`; KEEP preserves target. This is TM autopilot control, not the spec’s GlobalRoutePlanner/LocalPlanner/PID stack. Input keys/ranges/types are not validated in this method. [carla_env.py](../envs/carla_env.py#L172)
- **Collision:** Creates one collision sensor per controlled vehicle after five settling ticks via `CollisionMonitor`; `step` polls per-agent flags and ends on any collision. The monitor’s tests use fake sensors/worlds, not CARLA. [carla_env.py](../envs/carla_env.py#L166) [safety.py](../envs/safety.py#L12)
- **Goal/termination:** Goal is distance `<6 m` from the last route waypoint; reached agents ignore future actions and TM speed difference is set to 100%. Ends on collision, all goals, or stuck; truncates at 300 decisions. Stuck means speed `<0.5 m/s` and not at a traffic light for `>50` steps (51 decisions, 10.2 s); there is no off-road condition. [carla_env.py](../envs/carla_env.py#L207)
- **Info/reward:** Rewards are currently all zero. End info has `success`, but `min_ttc`/`jerk` are TODO; no TTC, jerk, scenario ID, travel-time, or episode metrics are computed. [carla_env.py](../envs/carla_env.py#L190)
- **Teardown:** Destroys monitor sensors, batches vehicle destruction, clears actor/state maps, ticks the world, then disables world and TM sync. It does not inspect batch destroy responses or protect teardown with `finally`; CARLA teardown was not run. [carla_env.py](../envs/carla_env.py#L85)
- **Stubs/TODOs:** `_get_obs()` is 56 zeros per agent; `action_masks()` is all-true; reward and TTC/jerk are placeholders. [carla_env.py](../envs/carla_env.py#L241)
- **Not implemented vs spec:** actual 56-D feature encoding, candidate selection inputs, action-mask encoding/validity, V2X channel, grouping, reward integration/team reward, off-road termination, scenario setup, metrics/CSV, train runner, paired evaluation, and CARLA validation. CARLA state has five values per agent (x/y/vx/vy/yaw) and omits the spec’s goal distance. [carla_env.py](../envs/carla_env.py#L245) [spec](../docs/env_spec.md#L31)

## 5. Observation and masks

| Environment | Current vector | Spec 56-D comparison |
|---|---:|---|
| CARLA | Declared 56, currently all zeros. | Dimension matches nominal total only; all 8 ego values, 6 route values, 36 candidate features, and 6 mask values are missing as data. `candidate_block()` exists but is not called by CARLA. |
| Toy | 4 values: x, y, speed, distance-to-intersection. | Missing ego/route/CAV context, candidates, V2X, and six action-mask inputs; only 3 actions. |
| Highway | Native Kinematics vector; exact width **UNVERIFIED** without running its installed dependency/config. | Not declared or padded to 56; no candidate/V2X feature block. |

Evidence: [CARLA observation/mask stubs](../envs/carla_env.py#L241), [toy observations](../envs/toy_env.py#L50), [Highway native observation config](../envs/highway_env.py#L40), [candidate helper](../envs/base.py#L19).

## 6. Reward audit

| Term | Spec §6 | `RewardCalculator` default / current wiring |
|---|---|---|
| Progress | +1.0 × normalized route progress | +1.0 × supplied distance; ToyEnv supplies Euclidean displacement. |
| Collision | −10; plus −10 team collision penalty | −100 per agent; no separate team penalty. |
| TTC safety | −0.5 formula below 3 s, plus small gap penalty | Weight 15, threshold 2.5 s, normalized linear penalty; no gap term. |
| Comfort | −|jerk| and −|lateral acceleration|, weight 0.05 | Acceleration and jerk excess over 2.0, weight 1.0 each; no lateral acceleration input. |
| Goal | +10 once | +50 when caller passes `goal_reached=True`. |
| Idle | −0.1 in specified conditions | Not implemented. |
| Team | `mean(r_i)` plus collision penalty | Calculator has no team term. `train_toy.py` averages individual rewards for the VDN return; CARLA does not call the calculator. |

Sources: [spec reward](../docs/env_spec.md#L67), [weights](../envs/reward.py#L28), [reward computation](../envs/reward.py#L120), [ToyEnv wiring](../envs/toy_env.py#L133), [trainer averaging](../scripts/train_toy.py#L38). Thus the claim that the calculator is unused is false for current ToyEnv, but true for CARLA/Highway; the spec weights and terms remain materially different.

**Goal-farming hypothesis:** Confirmed for the historical code that produced the checked-in VDN log; refuted for current ToyEnv. `results/vdn_toy_log.csv` was introduced in commit `cea5f8a`, and at that revision `envs/toy_env.py` advanced positions without a goal clamp, did not stop reached agents, rewarded progress every step, and ended only when both goals were reached or timeout (`git show cea5f8a:envs/toy_env.py`, lines 93–125). An early finisher could therefore continue accumulating progress while the other agent lagged or until timeout. Current ToyEnv explicitly freezes reached agents, caps their positions at 10, and rewards only newly reached goals, so this exploit is absent now. [current toy freeze/cap/reward](../envs/toy_env.py#L89)

## 7. VDN / MARL audit

- **VDN learner:** Shared per-agent MLP with one-hot agent ID; `Q_tot` is the sum of chosen local Qs. Uses Double-Q action selection/evaluation, reward scaling default 0.1, Smooth-L1 loss, gradient clipping at 10, and Polyak target update `tau=0.01`. Replay stores one scalar mean team reward, terminal flag, and next masks. [VDN](../marl/vdn.py#L10) [network](../marl/networks.py#L6) [buffer](../marl/buffers.py#L6)
- **Seeds/evaluation:** Training episode seeds vary as `args.seed + ep`. Random baseline uses seeds `10000+i`; greedy evaluation uses the same fixed set `20000+i` at each checkpoint (20 episodes, every 50). Thus training starts vary, but repeated greedy evaluations are not independent across checkpoints. [train_toy.py](../scripts/train_toy.py#L94)
- **Recorded result:** CSV has 2,000 episodes and 40 greedy-eval points; training return ranges −91.40 to 133.57, greedy eval −91.39 to 132.70, with the episode-1950 eval at 101.25. The curve is highly variable. These high values are from the older exploit-prone ToyEnv at commit `cea5f8a`, not evidence that current ToyEnv learns those returns. [CSV](./vdn_toy_log.csv#L1952) [curve](./vdn_toy_curve.png)
- **Other stability caveats:** One training seed/run, fixed evaluation seed set, exploration and eval share global NumPy randomness, and no confidence intervals or multiple independent runs. VDN itself has no recurrent state or prioritized replay. The observed historical swings cannot isolate algorithm instability from the old reward exploit.
- **QMIX:** Current `marl/qmix.py` is zero bytes. On `origin/Qmix` (`42889f7`), `origin/Qmix:marl/qmix.py` (lines 210–457; inspect with `git show origin/Qmix:marl/qmix.py`) contains `QMIX`, `QMixer`, `SharedAgentQNetwork`, and `QMIXReplayBuffer`; constructor requires `n_agents`, `obs_dim`, `state_dim`, defaults to six actions; `act(observations, action_masks, epsilon)` uses local observations; `update(batch)` consumes centralized states; `save/load/from_checkpoint` exist. `scripts/train_qmix_highway.py` is present on that branch. This code targets Highway-Env and does not establish CARLA integration.
- **MAPPO:** `marl/mappo.py` is empty on current branch and zero bytes on `origin/Qmix`; no MAPPO learner/interface is implemented. `docs/sota_A.md` is a comparison table only and says to fill results later. [SOTA A](../docs/sota_A.md#L1)

## 8. Tests

| Test/script | Coverage / dependencies |
|---|---|
| `tests/test_toy_env.py` | Toy reset/reproducibility, movement, invalid actions, state dimensions, collision geometry; pure Python + NumPy/pytest. |
| `tests/test_reward.py` | Reward terms and invalid numeric/agent inputs; pure Python + pytest. |
| `tests/test_collision_monitor.py` | Collision callback/reset/error/cleanup using fakes; no CARLA server. |
| `tests/test_contract.py` | Parametric toy/highway/CARLA API shapes, episode end, plus opt-in strict attribute/seed/info checks. Highway case requires `highway-env`; CARLA case imports `carla` and constructs a live `CarlaMultiAgentEnv`, so needs a running server. |
| `tests/test_grouping.py` | Empty; no tests. |
| `scripts/test_env.py` | Manual CARLA sync tick, actor cleanup across resets, random-action loop; requires running CARLA. **Not run.** |
| `scripts/test_highway.py` | Manual visual/headless Highway-Env rollout; requires `highway-env`, not CARLA. |

No CARLA test was run. The requested safe subset ran on the available Python 3.13 interpreter (with bytecode/cache writes disabled):

```text
python -m pytest tests -q -k "not carla and not highway" -p no:cacheprovider
27 passed, 4 skipped, 18 deselected in 1.92s
```

The Python 3.12 CARLA venv exists, but `python -m pytest --version` there reports `No module named pytest`; I did not install it. Python 3.13 has pytest 8.3.4.

Strict toy contract run (`CONTRACT_STRICT=1`, `tests/test_contract.py -k toy`):

```text
2 failed, 7 passed, 18 deselected
FAIL test_attributes[toy]: env must expose .obs_dim
FAIL test_info_keys[toy]: info is missing 'goal_reached'
```

Suggested commands (run from repo root; do not start a server as part of this audit):

```powershell
# Toy/core subset (available interpreter used in this audit)
$env:PYTHONDONTWRITEBYTECODE='1'
C:\Users\souro\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests -q -k "not carla and not highway" -p no:cacheprovider

# Strict toy contract
$env:CONTRACT_STRICT='1'
C:\Users\souro\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests\test_contract.py -q -k toy -p no:cacheprovider

# Highway contract/manual script, only in an environment with pytest and highway-env
python -m pytest tests\test_contract.py -v -k highway
python scripts\test_highway.py --headless

# CARLA contract/manual smoke test, only when CARLA is already running
C:\Users\souro\carla-env\Scripts\python.exe -m pytest tests\test_contract.py -v -k carla
C:\Users\souro\carla-env\Scripts\python.exe scripts\test_env.py
```

`requirements.txt` is empty, and pytest is absent from the provided Python 3.12 CARLA venv. The CARLA commands above are therefore not currently runnable in that venv without environment setup. The manual smoke script also reads `info['step']`, while the current CARLA env emits `info['steps']`; it needs alignment before its reported `ALL OK` can be treated as current evidence. [script line](../scripts/test_env.py#L35) [CARLA info](../envs/carla_env.py#L194)

## 9. Scripts and results

| File | Purpose / state |
|---|---|
| `scripts/train_toy.py` | VDN toy training/evaluation and CSV/PNG/checkpoint output. |
| `scripts/train.py` | Empty intended general runner. |
| `scripts/evaluate.py` | Empty intended evaluator. |
| `scripts/make_report.py` | Empty report generator. |
| `scripts/test_env.py` | CARLA manual smoke test. |
| `scripts/test_highway.py` | Highway-Env manual rollout. |
| `scripts/demo_carla.py` | Scripted CARLA video/demo, not a baseline evaluation. |
| `scripts/check_api.py` | Reset/step/state/mask shape printer. |

Current `results/` files: `api_carla.txt`, `api_toy.txt`, `test_env.txt`, `train_log.txt`, tracked `vdn_toy_log.csv`, `vdn_toy_curve.png`, `vdn_toy.pt`; `train_log2.txt` is untracked. The `api_toy.txt` output says termination/truncation are scalars, but current ToyEnv returns dicts. `test_env.txt` says `ALL OK`, but the current script’s `info['step']` access conflicts with current `info['steps']`; these logs are historical, not validation of the current worktree. `train_log.txt` and untracked `train_log2.txt` show interrupted training traces/KeyboardInterrupts, not successful complete reruns. The PNG shows a volatile training curve and eval points spanning poor (about −91) to >100 returns. VDN data details and historical exploit are in §6–7.

The current branch does not contain `results/qmix_highway/`; those results/checkpoints appear in history on `origin/Qmix`.

## 10. Docs and code inconsistencies

- `docs/env_spec.md` contains sections 1–11 only and ends with “I can write the `carla_env.py` skeleton following this spec next”; no section 12 is appended. The header in `envs/base.py` nevertheless refers to “section 12.” [spec ending](../docs/env_spec.md#L116) [base header](../envs/base.py#L1)
- Spec says 56-D normalized observation with ego/route/candidates/mask; CARLA returns zeros. Spec requires lane-change masks; CARLA masks all actions valid. [spec observation](../docs/env_spec.md#L31) [CARLA stubs](../envs/carla_env.py#L241)
- Spec rewards and team term differ from `RewardCalculator` defaults and current environment wiring (§6).
- Spec termination includes off-road; CARLA only checks collision/all-goal/timeout/stuck. Spec API asks for scenario/town/seed and TTC/jerk metrics; these are incomplete in step info.
- Spec control describes GlobalRoutePlanner/LocalPlanner/PID; implementation delegates to TM autopilot.
- `docs/sota_A.md` is only a baseline comparison and explicitly leaves project results for later. `docs/sota_B.md` describes selection methods as a planned controlled study and says the repo is not yet an experimental system. Neither constitutes project results.
- `README.md`, `requirements.txt`, `docs/architecture.md`, all four YAML configs, and grouping/V2X/scenario modules are empty. No `pytest.ini`, `docs/env_spec_section12.md`, or `sota_rows_*.md` exists.

## 11. Gap analysis vs plan

| Work item | Status | Evidence |
|---|---|---|
| Phase 0: `base.py` | **Partial** | Abstract API/candidate/action helpers exist, but info names diverge from spec and toy strict checks fail. [base](../envs/base.py#L42) |
| Phase 0: `test_contract.py` | **Partial** | Core tests exist; strict toy run fails 2 checks; CARLA integration not run. [test](../tests/test_contract.py#L138) |
| Phase 0: env_spec update / section 12 | **Not started** | Spec ends at section 11; no separate section-12 file. [spec](../docs/env_spec.md#L116) |
| Phase 0: contract passes on toy and CARLA | **Blocked / failing** | Toy strict failures verified; CARLA was not run and requires a server. |
| T1.1 collision sensor + termination | **Partial** | Current code includes collision sensor, collision/all-goal/timeout/stuck; off-road absent and route/TM behavior unverified against real CARLA. [CARLA termination](../envs/carla_env.py#L181) |
| T1.2 56-D observation + masks | **Not started** | Shape constant exists but obs is all zeros and masks all true. [stubs](../envs/carla_env.py#L241) |
| T1.3 reward integration + team reward | **Not started in CARLA** | CARLA returns zero rewards; ToyEnv has a different individual calculator; no shared spec team reward. [CARLA reward](../envs/carla_env.py#L190) |
| T1.4 per-episode CSV + random/scripted baselines | **Not started for CARLA** | Only toy VDN outputs exist; CARLA demo is not an evaluation harness; no episode metric logger. |
| T1.5 SOTA rows VDN/QMIX/MAPPO | **Partial** | A comparison table exists; QMIX is remote-branch/Highway code; MAPPO and measured CARLA results absent. [SOTA A](../docs/sota_A.md#L1) |
| Phase 2 scenarios | **Not started** | Scenario YAML and package marker are empty. |
| Phase 2 train/unseen-town split | **Partial** | Spec lists Town split; CARLA constructor accepts a town but no split/evaluation harness exists. [spec](../docs/env_spec.md#L92) |
| Phase 2 `train.py` runner | **Not started** | `scripts/train.py` is zero bytes. |
| Phase 2 `validate_carla.py` | **Not started** | No such tracked file. |
| Rule: only adapters import `carla`/`highway_env`; all else in `core/` | **Partial / not satisfied literally** | CARLA imports are in `envs/carla_env.py`, `envs/safety.py`, and `scripts/demo_carla.py`; Highway import is in its adapter. Generic code remains under `envs/`; no `core/` directory. [safety import](../envs/safety.py#L62) |

## 12. Ranked risk list

1. **Contract/API drift blocks reuse:** toy strict contract fails `.obs_dim` and `goal_reached`; highway omits dimension/action attributes and uses nested/non-contract info; base required-key names differ from spec.
2. **CARLA policy sees no state:** declared 56-D vectors are zeros and masks allow invalid lane changes; any learner currently receives no driving/candidate information.
3. **CARLA reward is zero:** no learning signal, TTC, jerk, or team reward; calculator weights also conflict with the frozen spec.
4. **Route following may silently fail:** `tm.set_path` errors are swallowed and no route adherence is measured; real CARLA/TM behavior is untested.
5. **Safety termination is incomplete/unverified:** no off-road check; stuck boundary triggers after 51 decisions; sensor callback behavior and cleanup have only fake-world tests.
6. **Core research hook is absent:** V2X channel and all selection modules are empty, so learned-vs-radius-vs-random comparison cannot run.
7. **Historical VDN score is misleading for current code:** the CSV’s source revision has a confirmed progress-farming path; it is not evidence for the current frozen-at-goal toy implementation.
8. **Branch/artifact ambiguity:** the only local merge commit is not on visible remote refs; QMIX/checkpoints exist on `origin/Qmix` but current branch has empty QMIX/MAPPO files; `.pt` files are not ignored.
9. **Validation evidence is stale/incomplete:** `test_env.txt` and `api_toy.txt` disagree with current interfaces; no CARLA test was run; Python 3.12 CARLA venv lacks pytest.
10. **Runner/scenario/evaluation infrastructure is empty:** configs and `train.py`/`evaluate.py` are placeholders; no paired seed/town split, episode CSV, or reproducible CARLA baseline.

## 13. Recommended next 10 actions for A

| # | Action (file or exact command) | Done when |
|---:|---|---|
| 1 | Agree with B on spec §9/§12 field names and strict expectations; update `docs/env_spec.md`, `envs/base.py`, and `tests/test_contract.py`. | Spec, required info constants, and tests encode one API consistently. |
| 2 | Coordinate B’s toy API fixes in `envs/toy_env.py`; run `$env:CONTRACT_STRICT='1'; C:\Users\souro\AppData\Local\Programs\Python\Python313\python.exe -m pytest tests\test_contract.py -q -k toy`. | Strict toy contract completes with zero failures. |
| 3 | In `envs/carla_env.py`, replace the zero observation with the spec’s ego/route/candidate/mask blocks and normalize/validate each block. | Real reset returns finite, nonzero features with exact length 56 for each agent. |
| 4 | In `envs/carla_env.py`, implement lane-aware `action_masks()` and validate exact action keys/types/ranges before stepping. | Invalid actions are rejected; masks disable unavailable lanes and always leave a valid action. |
| 5 | Verify spawn/route/goal/TM behavior with an owner-started CARLA 0.9.16 server; first reconcile `scripts/test_env.py` from `info['step']` to the canonical key and add route-adherence assertions. | Repeated resets leave only controlled actors, seeded routes repeat, and vehicles reach intended goals without silent path fallback. |
| 6 | Integrate the frozen reward specification in `envs/carla_env.py`, coordinating exact per-term weights with B’s `envs/reward.py`. | CARLA step rewards and terminal info include spec terms, TTC/jerk, and team reward with tests for each term. |
| 7 | Implement episode metric collection/CSV and actual random + scripted CARLA baselines (not just the demo) in a non-empty evaluation script. | One seeded episode produces the specified metrics and baseline rows in a documented schema. |
| 8 | Fill `configs/scenarios.yaml` and `envs/scenarios/`; implement scenario selection and the train/unseen-town split. | Each required scenario is seed-reproducible and split is enforced by runner tests. |
| 9 | Implement `scripts/train.py`, `scripts/evaluate.py`, and `validate_carla.py`; define/execute a CARLA validation command only when the server is already running. | Runner can train/evaluate VDN using the same API and emits reproducible artifacts; validation reports failures explicitly. |
| 10 | Once B’s V2X/grouping modules are integrated, run paired fixed-seed comparisons at identical `k=2`: learned attention, radius, random. | Results contain paired per-seed metrics, selector IDs, communication counts, and uncertainty estimates; no selection claim relies on toy VDN scores. |

## Questions I need the owner to answer

- Is the visible local-only merge commit `7d9cc01` intended to be pushed to `feat/env-and-vdn`, or should it remain local?
- Which info-key spelling is canonical: spec §9 (`collisions`, `ttc_min`) or `envs/base.py` (`collision`, `min_ttc`)?
- Are the default `RewardCalculator` weights allowed to change to spec §6 before the main experiments, and who owns the final freeze?
- Should `origin/Qmix` be merged into this branch, or remain a separate Highway-Env experiment?
- Is Highway-Env intended to remain in scope as an adapter/test fixture despite the CARLA-only research goal?
- Who owns resolving the untracked root files and `results/train_log2.txt`? They were left untouched.
