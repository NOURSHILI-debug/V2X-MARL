# Checkpoint A → B

**Audit date:** 2026-11-05  
**Branch checked:** `feat/env-and-vdn` at `93bf9f0` (`origin/feat/env-and-vdn`); remote refs fetched before inspection.  
**Scope:** Read-only inspection plus the two permitted pytest runs. No CARLA server started and no packages installed.

## 1. Status summary

- **Current phase:** Phase 0 contract work is present; T1.1 code is present but not validated against a live CARLA server.
- **Current task:** Finish CARLA T1.1 validation, then implement T1.2–T1.5 in order.
- **Cross-team blocker:** Toy strict contract fails; reward defaults and Highway info/attributes also diverge from Contract v1.
- **Research status:** No completed VDN run or CARLA baseline/metric evidence found.

## 2. Done and verified

| Item | Evidence |
|---|---|
| Phase 0 files exist and are pushed | `envs/base.py`, `tests/test_contract.py`, and `pytest.ini` exist locally and in `origin/feat/env-and-vdn` (`git ls-tree`). |
| Contract v1 appended and pushed | `docs/env_spec.md:128+` contains Section 12; `git show origin/feat/env-and-vdn:docs/env_spec.md` also contains “Contract v1”. Branch tip is `93bf9f0`. |
| T1.1 implementation is present (static inspection only) | [`envs/carla_env.py`](../envs/carla_env.py#L109): seeded lane-following routes and goals; `set_path` errors (`AttributeError`/`RuntimeError`) are silently swallowed (`:114-120`). Collision monitor is created after settling (`:167`). Any collision, all goals reached, or a stuck agent ends all agents; 300-step timeout truncates all (`:181-205`). Stuck condition is `> 50` slow decisions (`:186`), not an off-road check. No `route_applied` info flag exists. |
| `.pt` ignore and tracking status | `.gitignore:8` is exactly `*.pt`; `results/vdn_toy.pt` is present locally but absent from `git ls-files` and from tracked trees on `origin/feat/env-and-vdn`, `origin/main`, and `origin/Qmix`. |
| Main merge is on A’s pushed branch | Merge commit `7d9cc0169336841e93066b239bc103af0f4ac5c8` (“Merge remote-tracking branch 'origin/main' into feat/env-and-vdn”, 2026-11-01) is contained in `origin/feat/env-and-vdn`. |
| Permitted pure-Python test command passes | `pytest tests -q -k "not carla and not highway"` → **27 passed, 4 skipped, 18 deselected** (exit 0). Last terminal lines: `..........ssss................. [100%]`; `27 passed, 4 skipped, 18 deselected in 0.41s`; `Exit code: 0`. |
| Smoke script now uses plural info key | [`scripts/test_env.py`](../scripts/test_env.py#L35) reads `info['steps']`. The worktree diff changes the previous `info['step']` to `info['steps']`. |

## 3. Written but not validated

| Item | What is missing |
|---|---|
| T1.1 | The code exists, but it has not been validated against a live CARLA server in this checkpoint. `results/test_env.txt` records an older `ALL OK` run dated 2026-10-28, before the 2026-11-01 merge. `tests/test_termination.py` is absent; no recent output proves `test_env.py`, termination tests, or `pytest -k carla` ran. |
| Contract v1 checks | Strict toy contract run fails (details in §5); CARLA and Highway were not exercised in the permitted test run. |
| SOTA comparison | [`docs/sota_A.md`](../docs/sota_A.md#L1) contains VDN/QMIX/MAPPO comparison rows, but is not the requested `docs/sota_rows_1.md`; its “Code available” cells are labels, not code links. |
| VDN toy attempts | `results/train_log.txt`, `train_log2.txt`, and `train_log3.txt` end in `KeyboardInterrupt`/`^C`. The latest reaches episode 100 only. CSV/curve/checkpoint artifacts exist, but no log proves a completed run or convergence. |

## 4. Not started / incomplete

- **T1.2:** CARLA `_get_obs()` still returns 56 zeros per agent and `action_masks()` returns all `True` ([`envs/carla_env.py`](../envs/carla_env.py#L241), [`:254-256`](../envs/carla_env.py#L254)).
- **T1.3:** CARLA reward values remain all zero; `min_ttc` and `jerk` are TODOs ([`envs/carla_env.py`](../envs/carla_env.py#L191), [`:203-204`](../envs/carla_env.py#L203)).
- **T1.4:** No CARLA per-episode metrics CSV or random/scripted baseline results found. `scripts/evaluate.py` is empty; `docs/env_spec.md:103-114` describes the required episode metrics.
- **T1.5:** The required `docs/sota_rows_1.md` file does not exist (`Test-Path` false); `sota_A.md` is only a partial-format draft.

## 5. Contract compliance snapshot

Contract reference: [`docs/env_spec.md`](../docs/env_spec.md#L128) (Section 12). “Missing” refers to the current checked-out implementation; Highway observations are native Highway-Env vectors.

| Env | Reset / termination shape | `obs_dim` / `n_actions` | Info keys | Shared collision termination | Reward weights |
|---|---|---|---|---|---|
| Toy | Reset tuple; `terminated`/`truncated` are agent dictionaries; collision/all-goal termination is shared ([`toy_env.py:35-45,126-131`](../envs/toy_env.py#L35)). | **Fail:** `obs_dims=4`, no `obs_dim` or instance `n_actions` (`:15-18`). | Has `collision`, `steps`, and array `reached_goal`; lacks scalar `goal_reached` and episode `success`/`min_ttc`/`jerk` (`:144-150`). | Yes: collision sets every agent’s `terminated` true. | **Fail:** uses `RewardCalculator` defaults; those remain old weights (below). |
| CARLA | Reset tuple; agent dictionaries for `terminated` and `truncated` ([`carla_env.py:169-170,191-193`](../envs/carla_env.py#L169)). | `56` / `6` declared (`:24,50-51`). | Has `steps`, `collision`, `goal_reached`; `success` on episode end; missing `min_ttc`, `jerk`, `scenario_id` and end-of-episode `town`/`seed` (`:194-204`; reset returns town/seed at `:170`). | Yes: a collision ends all agents. | **Fail:** all CARLA rewards are `0.0` (`:191`). |
| Highway | Reset tuple; termination/truncation dictionaries broadcast shared flags ([`highway_env.py:74-96,148-167`](../envs/highway_env.py#L74)). | **Fail:** no instance `obs_dim` or `n_actions`; module-level `N_ACTIONS=6` only. | Per-agent `crashed` and singular `step`; no `collision`, `goal_reached`, `steps`, or episode metrics (`:151-160`). | Yes: any crash sets the common terminated flag (`:148-163`). | **Not wired:** native Highway reward is returned (`:160`), not Contract v1 weights. |

**Toy strict test:** `$env:CONTRACT_STRICT='1'; pytest tests/test_contract.py -q -k toy` → **2 failed, 7 passed, 18 deselected**. Failures: missing `obs_dim` and missing info key `goal_reached` (`tests/test_contract.py:138-175`). Inspection of `origin/main` (`1c8dc44`) and `origin/Qmix` (`42889f7`) shows the same `obs_dims`/`reached_goal` gaps. Toy goal freezing and one-time goal bonus are present (`toy_env.py:89-91,120-142`), but episode info keys are still missing.

**Reward defaults on the checked-out branch and both remote refs:** [`envs/reward.py`](../envs/reward.py#L28) uses collision `100`, safety/TTC `15`, comfort accel/jerk `1`, and goal `50`; no idle weight. This does not match Contract v1 / §6 (`docs/env_spec.md:73-77,174`), which specifies collision `10`, TTC `0.5`, comfort `0.05`, goal `10`, and idle `0.1`.

## 6. Problems found and handling

- The worktree was already dirty at the start. I left all pre-existing changes untouched: modified `results/vdn_toy_curve.png`, `results/vdn_toy_log.csv`, `scripts/test_env.py`; untracked `results/repo_audit.md`, `results/train_log2.txt`, and `results/train_log3.txt`. The same status remained after the permitted tests.
- `results/vdn_toy.pt` is an ignored local artifact, not a tracked result; it was not removed.
- Captured VDN logs are interrupted, not completion evidence (see §3).
- The only `test_env.txt` output is dated 2026-10-28 and cannot validate the current code. No `test_termination.py` exists.
- The existing `results/repo_audit.md:207` documents an empty `requirements.txt` and missing pytest in a Python 3.12 CARLA venv; this was not independently retested. No API-key mismatch or CARLA shader-crash evidence was found in the inspected `docs/` or `results/` text.
- No files were staged, committed, installed, or cleaned up; no CARLA server or CARLA test was run.

## 7. What I need from B

1. **Bring ToyEnv into Contract v1:** expose `obs_dim=4` and `n_actions=3`; report scalar `goal_reached` (true only when all agents have reached their goals); add required end-of-episode `success`, `min_ttc`, and `jerk`. Keep the existing goal freeze and one-time bonus. Verify with `$env:CONTRACT_STRICT='1'; pytest tests/test_contract.py -q -k toy`.
2. **Align RewardCalculator defaults to the frozen spec:** collision 10, TTC 0.5, comfort 0.05, goal 10, idle 0.1 (and the corresponding reward behavior). Verify/update with `pytest tests/test_reward.py -q`.
3. **Bring HighwayV2XEnv to the adapter contract:** expose instance `obs_dim`/`n_actions`; use required common info names (`collision`, `goal_reached`, `steps`) and required episode metrics; retain team-wide collision termination. Verify with `pytest tests/test_contract.py -q -k highway`.
4. **Clarify Qmix integration:** no PR was found for head `Qmix` → base `main` (GitHub search returned 0), and `origin/Qmix` is not an ancestor of `origin/main` (`git merge-base --is-ancestor origin/Qmix origin/main` exits 1). If integration is intended, open a PR and run the same ancestry check after merge. No `*.pt` is tracked on either ref.

## 8. What I will do next

1. **T1.1 CARLA validation:** run `scripts/test_env.py`, add/run `tests/test_termination.py`, and run `pytest tests/test_contract.py -q -k carla` only with CARLA already available. **Done when:** real-server checks cover collision/shared termination, all-goal, 300-step timeout, stuck and off-road behavior, and the contract test passes. Current implementation has no off-road check.
2. **T1.2 observations/masks:** implement the specified 56-D layout (ego 8 + route 6 + 6 candidates × 6 + mask 6) and lane-aware action masks. **Done when:** emitted vectors have the specified populated layout and masks reject unavailable lane changes, with tests.
3. **T1.3 rewards:** wire CARLA to the frozen per-agent reward terms and populate `min_ttc`/`jerk` plus the specified team reward. **Done when:** no zero-reward placeholder remains and formula/info tests match Contract v1.
4. **T1.4 metrics/baselines:** add per-episode CSV output and random/scripted baselines. **Done when:** repeatable runs record the episode metrics in `docs/env_spec.md` §10 and baseline results.
5. **T1.5 SOTA rows:** complete `docs/sota_rows_1.md` with paradigm, credit assignment, grouping, scalability, and code link. **Done when:** all required fields are populated and project-result claims are backed by reproduced experiments.

## 9. Things not to claim

- “CARLA env is validated against a real server.”
- “CARLA observation/action masks are implemented” or “CARLA rewards are wired.”
- “Toy strict Contract v1 passes.”
- “RewardCalculator defaults match the frozen spec.”
- “Highway exposes the full Contract v1 attributes/info.”
- “A VDN run completed” or “VDN converged.”
- “CARLA metrics/baselines are recorded” or “the required SOTA rows file is complete.”
- “Qmix has been merged into main” or “a Qmix-to-main PR exists.”

## 10. Ready-to-send message to B

Hi B — I verified Phase 0 and Contract v1 are on `feat/env-and-vdn`; the pure-Python suite passes (27 passed, 4 skipped), but strict Toy contract still fails on `obs_dim` and `goal_reached`. Toy already freezes at the goal and gives its bonus once. Please add the missing Toy attributes/info, align RewardCalculator’s defaults with §6, and bring Highway’s attributes/info keys into contract (while keeping shared collision termination). The strict Toy command is `$env:CONTRACT_STRICT='1'; pytest tests/test_contract.py -q -k toy`; use `pytest tests/test_reward.py -q` and `pytest tests/test_contract.py -q -k highway` for the other checks. I found no Qmix→main PR or merge; please open a PR if integration is intended. I’ll validate CARLA T1.1, then implement T1.2 observations/masks, T1.3 rewards, T1.4 metrics/baselines, and finish T1.5 rows. No CARLA validation or completed VDN run is evidenced yet.
