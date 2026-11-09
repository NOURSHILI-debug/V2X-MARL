# PROGRESS

Branch: `feat/marl-audit-mappo`

| Stage | Commit | Test status | Next |
|-------|--------|-------------|------|
| 0 inspect | (none) | `pytest tests/test_qmix.py -q` -> 7 failed, 2 passed (pre-fix) | 1 audit |
| 1 audit doc | `6a2dc26` docs: QMIX/VDN audit | N/A (doc only) | 2 fix and green |

## Stage 1 summary

- Wrote `docs/qmix_audit.md` with per-item verdicts and evidence.
- Bugs found: StateNormalizer always built + assumes groups of 5; test_qmix.py:173
  old info shape; VDN lacking checkpoint tests; trainer/reward/termination not
  implemented yet.
- Measured contract56 raw state ranges over 20 episodes (see audit doc).
- Uncommitted pre-existing stage-2 work remains in `marl/buffers.py`,
  `marl/qmix.py`, `marl/vdn.py`, `marl/normalization.py`,
  `scripts/train_common.py`.

## Resume instructions

A fresh session should read this file plus `docs/qmix_audit.md`, then continue at
stage 2 (fix bugs, get all gates green). Do not redo the audit doc.
