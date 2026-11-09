# PROGRESS

Branch: `feat/marl-audit-mappo`

| Stage | Commit | Test status | Next |
|-------|--------|-------------|------|
| 0 inspect | (none) | `pytest tests/test_qmix.py -q` -> 7 failed, 2 passed (pre-fix) | 1 audit |
| 1 audit doc | `6a2dc26` docs: QMIX/VDN audit | N/A (doc only) | 2 fix and green |
| 2 fix + green | `d179014` fix(qmix,vdn): audit fixes, mse loss, double_q, masks | `test_qmix.py`+`test_vdn.py` 20 passed; `tests -k "not carla and not toy"` 84 passed, 4 skipped; contract highway 5 passed/4 skipped; CONTRACT_STRICT=1 9 passed | 3 correctness |
| 3 correctness | `47996e3` test: QMIX/VDN correctness | `test_qmix_correctness.py` 11 passed; `tests -k "not carla and not toy"` 95 passed, 4 skipped | 4 trainer |

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

## Resume instructions

A fresh session should read this file plus `docs/qmix_audit.md`, then continue at
stage 4 (comparison trainer). Do not redo stages 0-3.
