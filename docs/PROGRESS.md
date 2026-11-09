# PROGRESS

Branch: `feat/marl-audit-mappo`

| Stage | Commit | Test status | Next |
|-------|--------|-------------|------|
| 0 inspect | (none) | `pytest tests/test_qmix.py -q` -> 7 failed, 2 passed (pre-fix) | 1 audit |
| 1 audit doc | `6a2dc26` docs: QMIX/VDN audit | N/A (doc only) | 2 fix and green |
| 2 fix + green | `d179014` fix(qmix,vdn): audit fixes, mse loss, double_q, masks | `test_qmix.py`+`test_vdn.py` 20 passed; `tests -k "not carla and not toy"` 84 passed, 4 skipped; contract highway 5 passed/4 skipped; CONTRACT_STRICT=1 9 passed | 3 correctness |
| 3 correctness | `47996e3` test: QMIX/VDN correctness | `test_qmix_correctness.py` 11 passed; `tests -k "not carla and not toy"` 95 passed, 4 skipped | 4 trainer |
| 4 trainer | `8ad131d` feat: qmix/vdn comparison trainer | tests 95 passed, 4 skipped; contract56 smoke 2 seeds x 5 eps for QMIX+VDN populated CSVs (see `results/raw/smoke_stage4/`) | 5 measure + budget |
| 5 measure | pending | timing below; budget proposed, waiting for user OK | 5 launch after OK |

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

## Proposed budget (target ~2 h total; user target hours were not filled in)

- algorithms: QMIX and VDN; seeds: 3 (0,1,2); episodes/seed: 200.
- eval interval: 20 (10 eval points); eval seeds: 5 (7001-7005).
- background vehicles: 12 (contract56 minimum; lower values are ignored).
- duration: 20 s (max 100 steps); replay 5000; warmup 200; batch 128.
- hidden 128, mixing 32, hypernet 64; lr 3e-4, gamma 0.99, tau 0.01, reward scale 0.1.
- epsilon fraction 0.7, end 0.05; double_q False; state_norm False.
- Reserved test seeds: 9001-9030 (disjoint from train episode seeds and eval seeds).
- Estimate: ~0.5-3 h depending on how long episodes survive (collisions shorten them).

## Resume instructions

A fresh session should read this file plus `docs/qmix_audit.md`. Stage 5 is waiting
for user approval of the budget; after approval, commit and launch the background
training command recorded here. Do not redo stages 0-4.
