# PROGRESS

Branch: `feat/marl-audit-mappo`

| Stage | Commit | Test status | Next |
|-------|--------|-------------|------|
| 0 inspect | (none) | `pytest tests/test_qmix.py -q` -> 7 failed, 2 passed (pre-fix) | 1 audit |
| 1 audit doc | `6a2dc26` docs: QMIX/VDN audit | N/A (doc only) | 2 fix and green |
| 2 fix + green | `d179014` fix(qmix,vdn): audit fixes, mse loss, double_q, masks | `test_qmix.py`+`test_vdn.py` 20 passed; `tests -k "not carla and not toy"` 84 passed, 4 skipped; contract highway 5 passed/4 skipped; CONTRACT_STRICT=1 9 passed | 3 correctness |
| 3 correctness | `47996e3` test: QMIX/VDN correctness | `test_qmix_correctness.py` 11 passed; `tests -k "not carla and not toy"` 95 passed, 4 skipped | 4 trainer |
| 4 trainer | `8ad131d` feat: qmix/vdn comparison trainer | tests 95 passed, 4 skipped; contract56 smoke 2 seeds x 5 eps for QMIX+VDN populated CSVs (see `results/raw/smoke_stage4/`) | 5 measure + budget |
| 5 measure | `64ce2c0` docs: progress stage 5 timing | timing below; budget proposed | A survivability |
| A env fix | `a55d771` fix(env): contract56 car-following safety speed | `tests/test_contract.py`+`test_highway_adapter.py` 26 passed/17 skipped; strict highway 9 passed; `tests` 114 passed/17 skipped | B step budget + launch |
| B step budget + launch | `0e94ae6` feat(trainer): step-based budget and eval cadence | `tests` 114 passed/17 skipped; smoke+resume verified; 6/6 runs hit 15000 steps with eval at 3000/6000/9000/12000/15000 | C plots |
| C plots | `8b1e817` feat(plots): partial-data QMIX vs VDN curves | `tests` 114 passed/17 skipped; partial smoke + full 4 PNGs | D held-out test |
| 1 MAPPO | (this commit) feat: MAPPO | `tests/test_mappo.py` 11 passed; `tests -k "not carla and not toy"` 136 passed, 4 skipped | 2 MAPPO trainer |
| 2 MAPPO trainer | (this commit) feat: MAPPO trainer | trainer smoke + resume verified; timing 0.0714 s/step | 3 launch background |
| 2b MAPPO run | (this commit) docs: MAPPO run results | 5/5 seeds hit 15000 steps, 15 updates, 15 eval rows; stderr log empty | 3 comparison (new session) |
| 3 comparison | (this commit) feat: comparison plots; docs: three-way results | 420 held-out episodes; all required plots; `tests -k "not carla and not toy"` 136 passed, 4 skipped | final report |

## Stage 1 summary

- Wrote `docs/qmix_audit.md` with per-item verdicts and evidence.
- Bugs found: StateNormalizer always built + assumes groups of 5; test_qmix.py:173
  old info shape; VDN lacking checkpoint tests; trainer/reward/termination not
  implemented yet.
- Measured contract56 raw state ranges over 20 episodes (see audit doc).
- Uncommitted pre-existing stage-2 work remains in `marl/buffers.py`,
  `marl/qmix.py`, `marl/vdn.py`, `marl/normalization.py`,
  `scripts/train_common.py`.

## Stage 2 summary

- `StateNormalizer` is no longer built when `state_norm=False`; when enabled it
  requires explicit `state_mean` / `state_std` of shape `(state_dim,)`.
  Checkpoints store/restore the scales.
- QMIX and VDN both support `double_q`, MSE loss, masked target max, rounded
  epsilon-greedy, and checkpoints. VDN default `reward_scale` aligned to 1.0.
- Buffer stores current + next masks (numpy copies).
- Fixed `tests/test_qmix.py` info handling to `info["per_agent"]` and `env.obs_dim`.
- Added `tests/test_vdn.py` (acts, update, checkpoint round trip, mismatch) and
  QMIX state-normalization tests.

## Stage 3 summary

- Added `tests/test_qmix_correctness.py`: mixer monotonicity, VDN sum, masked
  epsilon-greedy, masked target max (both `double_q` branches), truncation
  bootstrap vs termination, checkpoint Q-value round trip, 2-agent matrix game
  convergence for QMIX and VDN, and one update at `obs_dim=56`.

## Stage 4 summary

- Rewrote `scripts/train_qmix_highway.py` for contract56: prints/asserts config,
  disjoint train/eval/test seeds, greedy eval on fixed seeds, random baseline,
  per-episode CSV rows (flushed), per-episode resumable learner+buffer+state in
  `results/raw/`, and sequential algorithms/seeds in one process.
- CSV columns: algorithm, seed, episode, team_return, collision, mean_speed,
  steps, epsilon, loss, eval, env_seed.
- Smoke: 2 seeds x 5 episodes, contract56, QMIX and VDN, losses and eval rows
  populated (`results/raw/smoke_stage4/`). Resume was also exercised.

## Stage 5 measurements (2 episodes per algorithm, contract56, CPU, 2 threads)

| Algorithm | eps | steps/ep | env s | act s | update s | sec/step | steps/s |
|-----------|-----|----------|-------|-------|----------|----------|---------|
| QMIX | 1.00 | 6.0 | 0.636 | 0.007 | 0.126 | 0.0642 | 15.6 |
| QMIX | 0.05 | 12.0 | 1.271 | 0.012 | 0.234 | 0.0632 | 15.8 |
| VDN | 1.00 | 6.0 | 0.657 | 0.007 | 0.060 | 0.0604 | 16.6 |
| VDN | 0.05 | 3.5 | 0.403 | 0.004 | 0.047 | 0.0649 | 15.4 |

- Update-only: QMIX 10.2 ms/update, VDN 8.5 ms/update (batch 128).
- Env stepping dominates (~84% of wall time); replay sample/add is negligible.
- Episode length is short early due to collisions (3-12 steps); max is duration*5.
- Buffer save: capacity 5000 -> 0.02 s / ~11 MB; capacity 20000 -> 0.08 s / ~44 MB.

## Stage A summary (contract56 survivability)

- Baseline before fix (20 seeds, 300 max steps): keep-speed median 6.5 steps,
  100% collisions (21 CAV-background events); random median 5.0, 100%.
  CAVs are pure speed controllers at 25 m/s while IDM background traffic spawns
  at 21-24 m/s; median same-lane bumper gap at reset is ~2.5 m.
- Spacing/density alone did not fix it (keep-speed still 100% collisions) and
  lowering density pushed candidates below 6, so spawn settings were kept.
- Fix in `envs/highway_env.py` only: contract56 CAVs use a car-following safe
  target speed from the nearest lead in the current/target lane
  (`CAR_FOLLOW_MIN_GAP_M=15`, `CAR_FOLLOW_HEADWAY_S=1`). Spawn, density,
  observations, actions, and non-contract layouts are unchanged.
- After fix (20 seeds): keep-speed median 300/300 steps, 0% collisions; random
  median 9.5 steps, 100% collisions (19 CAV-background, 1 CAV-CAV); candidate
  minimum at reset remains 6; no episode seed diverges.
- `pytest tests/test_contract.py tests/test_highway_adapter.py -q`: 26 passed,
  17 skipped. `CONTRACT_STRICT=1 pytest tests/test_contract.py -k highway`:
  9 passed. `pytest tests -q`: 114 passed, 17 skipped. (`pytest -q` at repo root
  still collects `scripts/test_env.py` and `scripts/test_termination.py`, which
  import CARLA and fail without the CARLA package; unchanged.)

## Stage B summary (step-based training, completed)

Command (launched in background with `Start-Process`, PID 11344):
`python scripts/train_qmix_highway.py --algorithms qmix vdn --seeds 0 1 2
--total-steps 15000 --eval-interval 3000 --eval-seeds 7001 7002 7003
--out results/raw/comparison --torch-threads 2`

- Outputs: `results/raw/comparison/` (run_metadata.json, runs/*/metrics.csv,
  resume.pt, buffer.npz, state.json), log `results/raw/comparison_train.log`,
  stderr log empty, launch record `results/raw/comparison_launch.txt`.
- 6/6 runs reached exactly 15000 steps: qmix_seed0/1/2, vdn_seed0/1/2.
  Each has 15 eval rows = 5 eval points (3000, 6000, 9000, 12000, 15000) x 3
  eval seeds (7001-7003); random baseline has 3 eval rows.
- Wall time from the log: 8147.6 s total (~2.26 h); per run: QMIX
  1218.7/1863.6/1656.8 s, VDN 1167.1/1100.4/1141.0 s.
- Pre-launch gates: `pytest tests -q` -> 114 passed, 17 skipped; step-based
  smoke and resume exercised under `results/raw/smoke_stageB/`.
- Raw results only; no plots yet.

## Stage C summary (plots, completed)

- `scripts/plot_comparison.py` reads `runs/*/metrics.csv`, skips missing runs,
  eval points, or seeds, and prefixes every figure with `PRELIMINARY - ` unless
  `run_metadata.json` shows all expected runs reached the total step budget and
  the final eval.
- Command: `python scripts/plot_comparison.py --input results/raw/comparison
  --output results/raw/plots`.
- Partial smoke: first 3000 steps of `qmix_seed0` and `vdn_seed0` under
  `results/raw/plot_smoke_input/` produced all four PNGs under
  `results/raw/plot_smoke/` with a PRELIMINARY title.
- Full run: 4107 rows, 6 run dirs, complete -> `results/raw/plots/`:
  `eval_return_curve.png`, `eval_collision_curve.png`,
  `training_return_curve.png`, `final_eval_bars.png`.
- Final eval at 15000 steps (3 seeds): QMIX mean return -90.94, collision 1.00;
  VDN mean return -113.58, collision 0.67. Random baseline (3 eval seeds at
  step 0): mean return -24.68, collision 1.00. Variance is high; these are raw
  observations, not evidence of success.
- `pytest tests -q`: 114 passed, 17 skipped.

## Stage MAPPO-1 summary (MAPPO learner)

- Added `marl/mappo.py`: shared masked actor (local obs + agent one-hot),
  centralized team-value critic, `RolloutBuffer` (obs, state, actions,
  log-probs, values, team reward, masks, terminated, truncated, next state),
  pure `compute_gae`, PPO update (advantage normalization, clipped surrogate,
  MSE value loss with optional clipping, entropy bonus, grad clip, epochs and
  minibatches), and `act`/`save`/`load`/`from_checkpoint`.
- GAE: no bootstrap after true termination; bootstrap from `V(next_state)` on
  truncation; episode boundaries reset the recursion.
- State normalization stays opt-in with explicit scales, matching QMIX.
- Added `tests/test_mappo.py` (11 tests): hand-computed GAE terminal and
  truncation cases, ratio 1 / clip fraction 0 before the first update, zero
  probability for masked actions, no invalid actions from `act`, checkpoint
  round trip, seeded determinism, 2-agent matrix game solved, one update at
  `obs_dim=56`, and rollout buffer round trip.
- Gate: `pytest tests -k "not carla and not toy" -q` -> 136 passed, 4 skipped.
- No env files or partner files changed; `marl/` imports neither `highway_env`
  nor `carla`.

## Stage MAPPO-2 preflight (read-only checks, no runs changed)

- Consistency: QMIX and VDN share `results/raw/comparison/run_metadata.json`;
  env config and budget are identical (contract56, 12 background vehicles,
  duration 60, 15,000 steps, eval every 3,000 on 7001-7003, test 9001-9030).
- `reward_scale=0.1` is applied only inside the learner updates (QMIX
  `td_target`, VDN `td_target`); the trainer logs `info["team_reward"]`
  directly, so the logged `team_return` is unscaled and comparable across
  QMIX/VDN. No between-algorithm config confound found. MAPPO reuses
  `reward_scale=0.1` so its logged returns stay on the same scale.
- Read-only 20-seed diagnostic (contract56, duration 60, 12 vehicles):
  keep-speed median 300/300 steps and 0% collision; random median 8 steps
  (mean 10.9), 100% collision, nearest-vehicle cause background 21 / CAV 0.
  Limitation: the scenario is crash-dominated for a random policy, and
  keep-speed survives only because it is degenerate/slow. This is a scenario
  limitation, not a fix, and Stage 4 must state it.

## Stage MAPPO-2 summary (trainer)

- Added `scripts/train_mappo_highway.py`, mirroring
  `scripts/train_qmix_highway.py`: same env (`contract56`, duration 60,
  effective 12 background vehicles), same eval seeds 7001-7003, same test
  seeds 9001-9030, same 15,000-step budget and 3,000-step eval cadence, and the
  same CSV columns (algorithm, seed, episode, total_steps, team_return,
  collision, mean_speed, steps, epsilon, loss, eval, env_seed).
- Per-episode flushed CSV rows, resumable learner + rollout + state
  (`resume.pt`, `rollout.npz`, `state.json`), start-of-run config print,
  `timing.json` per run, and metadata merge that appends `mappo` without
  overwriting the shared QMIX/VDN `train_seeds`.
- Smoke: 256-step and 2048-step runs populated CSVs and updates; resume from
  step 256 to 384 restored the learner and continued (`mappo_train_seeds`).
- Timing smoke (2,048 steps, rollout 1,024, 4 epochs, minibatch 256,
  2 threads): 146.3 s, 0.0714 s/step, 14.0 steps/s, 2 updates -> about
  18 min per 15,000-step seed.
- No partner or env files changed; `marl/` still imports neither `highway_env`
  nor `carla`.

## Stage MAPPO-2 launch results (completed)

- Command (background `Start-Process`, PID 11368; log
  `results/raw/mappo_train.log`, stderr `results/raw/mappo_train.err.log`
  empty, record `results/raw/mappo_launch.txt`):
  `python scripts/train_mappo_highway.py --seeds 0 1 2 3 4 --total-steps 15000
  --eval-interval 3000 --eval-seeds 7001 7002 7003
  --out results/raw/comparison --torch-threads 2`
- 5/5 runs reached exactly 15000 steps with 15 PPO updates and 15 eval rows
  each; per-seed wall time 1218-1434 s (~0.081-0.096 s/step, 10.5-12.3
  steps/s); total ~1.83 h. CSV columns match QMIX/VDN exactly.
- Per-seed training mean steps 7.8/14.6/8.5/9.9/12.3, training collision
  0.999 for all seeds. Final eval (15000 steps, 3 eval seeds): seeds 0/2/3
  return -10.2/-8.5/-9.3, collision 1.00; seeds 1/4 return +206.864,
  collision 0.00.
- Honest reading: seeds 1 and 4 collapsed to the degenerate keep-speed
  survival policy (identical final eval return), which is the same
  crash-dominated scenario limitation noted above. The high variance and the
  collapse are raw observations, not evidence of competence; Stage 3 must
  evaluate final checkpoints on held-out test seeds 9001-9030 and compare
  every learner against the random baseline.
- `run_metadata.json` now lists algorithms `qmix`, `vdn`, `mappo`, keeps the
  shared `train_seeds` `[0, 1, 2]`, and records `mappo_train_seeds`
  `[0, 1, 2, 3, 4]` plus `mappo_config`. No existing run was modified.

## Stage 3 comparison summary (completed)

- Held-out evaluation: 420 episodes = 14 groups x 30 test seeds
  (QMIX/VDN 0-2, MAPPO 0-4, random, keep_speed, accelerate). Source tables:
  `results/raw/eval/heldout_episodes.csv`, `summary.csv`,
  `paired_differences.csv`, `reference_comparison.csv`, `collapse_check.csv`.
- `scripts/plot_comparison.py` now covers QMIX/VDN/MAPPO plus gray references
  and writes training/eval curves, held-out per-seed panels, a held-out
  summary, action histograms and a wall-clock/throughput panel under
  `results/raw/plots/`. Every title is `PRELIMINARY` and shows the layout,
  seeds, 15,000-step budget and date.
- Results: QMIX return -63.32, VDN -118.95, MAPPO 0-2 +59.20, MAPPO 0-4
  +73.21; random -21.40; keep_speed +199.74; accelerate -12.17. Keep-speed is
  the bar to beat and no learner beats it. MAPPO is bimodal: seeds 1 and 4
  are exact keep-speed clones (`a0=1.000`, return 199.7367, collision 0.0333),
  seeds 0/2/3 crash with collision 1.000. QMIX/VDN do not collapse.
- Paired return differences over seeds 0-2 exclude zero (QMIX-VDN +55.63
  [14.19, 119.54]; MAPPO-VDN +178.15 [65.21, 328.26]; MAPPO-QMIX +122.52
  [21.45, 295.09]), but the MAPPO advantage is the seed-1 collapse and the
  MAPPO seed-level return CI includes zero. No learned winner is declared.
- `docs/three_way_results.md` records setup, metrics, paired differences,
  collapse flags, figures, honest reading and limitations. Final summary
  figures are copied to `docs/figures/`.
- Gate: `pytest tests -k "not carla and not toy" -q` -> 136 passed, 4 skipped,
  33 deselected. No partner or environment files changed; no training or new
  evaluation episodes were run in this stage.

## Stage B budget (mandated, step-based; target ~2 h)

- algorithms: QMIX and VDN; seeds: 0, 1, 2; ~15,000 env steps/seed/algo.
- eval every 3,000 steps on fixed greedy eval seeds 7001-7003.
- warmup and epsilon schedule expressed in env steps (not episodes).
- reserved held-out test seeds: 9001-9030 (never used for training/eval).
- background vehicles: 12 (contract56 minimum; lower values are ignored).
- replay 5000; batch 128; hidden 128, mixing 32, hypernet 64;
  lr 3e-4, gamma 0.99, tau 0.01, reward scale 0.1.
- epsilon end 0.05; double_q False; state_norm False.
- The current trainer budgets in episodes; Stage B must convert it to the
  step-based budget above and launch in the background without waiting.

## Resume instructions

A fresh session should read this file plus `docs/qmix_audit.md` and
`docs/three_way_results.md`. Stages 0-C, MAPPO and the three-way comparison are
complete. Raw runs are under `results/raw/comparison/`, held-out tables under
`results/raw/eval/`, and plots under `results/raw/plots/` (ignored by git).
Do not retrain or rerun evaluation episodes. To regenerate the figures:

```text
python scripts/plot_comparison.py --input results/raw/comparison \
  --eval results/raw/eval/heldout_episodes.csv --output results/raw/plots
```

Gate: `pytest tests -k "not carla and not toy" -q` -> 136 passed, 4 skipped.
The next step is the final report, not another training or evaluation run.

