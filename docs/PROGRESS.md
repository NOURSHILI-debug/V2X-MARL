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

A fresh session should read this file plus `docs/qmix_audit.md`. Stages A and B
are complete (`a55d771`, `0e94ae6`); raw runs are under
`results/raw/comparison/`. Stage C is next: write `scripts/plot_comparison.py`
so it tolerates partial CSVs, write PNGs under `results/raw/plots/`, test it on
the first CSV rows, report the paths, and stop. Do not redo stages 0-B.
