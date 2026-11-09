# QMIX / VDN audit (contract56, highway-env only)

Scope: `marl/qmix.py`, `marl/vdn.py`, `marl/buffers.py`, `marl/networks.py`,
`marl/normalization.py`, `scripts/train_common.py`, and their use of
`envs/highway_env.py`. No CARLA, no MAPPO implementation. Verdicts describe the
code as inspected at the start of the audit; "Change needed" says whether the
stage-2 fix alters behaviour.

| # | Item | Verdict | File:line evidence | Change needed |
|---|------|---------|--------------------|---------------|
| 1 | Mixer monotonicity | CORRECT | `QMixer` uses `F.softplus` for `w1`/`w2` so weights are non-negative (`marl/qmix.py:85`, `marl/qmix.py:94`); test asserts non-negative gradients (`tests/test_qmix.py:31`) | No |
| 2 | Q_tot uses the taken actions | CORRECT | `update` gathers online Q at `actions` before mixing (`marl/qmix.py:405-409`) | No |
| 3 | Target uses masked max with next masks | CORRECT | `target_actions` fills invalid entries with `-inf` then `argmax` (`marl/qmix.py:355-359`); VDN equivalent (`marl/vdn.py:92-96`) | No |
| 4 | Terminated vs truncated | PARTIAL | Env exposes both: `terminated_flag` from true end/collision/goal and `truncated_flag` for time limit (`envs/highway_env.py:432-433`, `454-455`); learner boots on `1-terminated` (`marl/qmix.py:388`, `marl/vdn.py:109`). No trainer exists yet to prove only `terminated` is stored | Yes: trainer must store `terminated`, never `truncated` |
| 5 | Shared agent network, agent one-hot, no global-state leakage | CORRECT | `SharedAgentQNetwork` concatenates a one-hot id (`marl/qmix.py:36-43`); `act` uses local obs only (`marl/qmix.py:323-328`); VDN `AgentQNet` identical (`marl/networks.py:19-22`) | No |
| 6 | Epsilon-greedy restricted to valid actions | CORRECT | Random branch samples `np.flatnonzero(mask)` and greedy masks with `-inf` (`marl/qmix.py:330-340`, `marl/vdn.py:68-74`) | No |
| 7 | Replay buffer contents and copies | QUESTIONABLE | QMIX buffer allocates current + next masks (`marl/qmix.py:132-137`) and assigns with numpy copy semantics (`marl/qmix.py:181-189`), but `sample` returns only `next_action_masks` (`marl/qmix.py:207-216`); VDN `ReplayBuffer` stores `mask` (`marl/buffers.py:14`) but never samples it (`marl/buffers.py:29-30`) | Yes: make current masks sampled/used where relevant, keep explicit copies |
| 8 | Reward key used | CORRECT (helper only) | `episode_step_metrics` returns `infos["team_reward"]` (`scripts/train_common.py:74-80`); no trainer yet | Trainer must call this, not `mean(rewards)` |
| 9 | Target update, warmup, grad clip, loss | QUESTIONABLE | Soft target `lerp_` each update (`marl/qmix.py:424-431`); grad clip 10 (`marl/qmix.py:415-418`); MSE loss (`marl/qmix.py:412`, `marl/vdn.py:115`). No warmup/batch-size/target-interval parameters | Yes: add warmup + update cadence in trainer |
| 10 | Global state scaling | BUG | `StateNormalizer` is always constructed even when `state_norm=False` (`marl/qmix.py:272-277`); default constants assume groups of 5 (`marl/normalization.py:20-27`), so `state_dim=8` learners raise. Measured raw ranges below | Yes: build normalizer only when enabled; require explicit scales |
| 11 | No hard-coded dimensions | CORRECT | Dims are constructor args in QMIX (`marl/qmix.py:229-248`) and VDN (`marl/vdn.py:17-30`); agent one-hot derived from `n_agents` | No |
| 12 | Info handling | BUG (test) | Env returns top-level dict info with `per_agent`, `team_reward`, `collision`, `speed_mps` (`envs/highway_env.py:441-453`); `tests/test_qmix.py:173` asserts the old per-agent key shape | Yes: fix test to `set(info["per_agent"]) == set(agents)` and derive dims from `env.obs_dim` |
| 13 | Evaluation protocol | CORRECT (helper only) | `evaluate_policy` runs greedy on fixed seeds and aggregates return/speed/collision (`scripts/train_common.py:95-140`) | Wire into trainer |
| 14 | Multi-seed support | PARTIAL | `write_run_metadata` accepts train/eval seed lists (`scripts/train_common.py:150-196`); no trainer/looper yet | Yes: trainer loops algorithms/seeds in one process |
| 15 | Seeding (python, numpy, torch, per-learner RNG) | QUESTIONABLE | `seed_everything` sets python/numpy/torch + deterministic algorithms (`scripts/train_common.py:17-21`); per-learner RNG (`marl/qmix.py:271`, `marl/vdn.py:40`). Deterministic mode can be slow; no per-environment seeding wrapper beyond `reset(seed)` | Keep; verify smoke run speed |
| 16 | Checkpoint round trip (QMIX and VDN) | CORRECT | QMIX save/load/from_checkpoint stores dims, `double_q`, `state_norm`, normalizer (`marl/qmix.py:447-525`); VDN equivalent stores `double_q` (`marl/vdn.py:125-185`). VDN has no tests yet | Add VDN checkpoint tests |

## Measured global-state ranges (contract56, 20 random-policy episodes)

`env.state()` is 4 controlled vehicles x `[x, y, vx, vy, heading]` = 20-D.
20 episodes, `duration=10`, 12 background vehicles, 147 state samples.

| Feature | min | max | mean | std |
|---------|-----|-----|------|-----|
| x (m) | 48.6363 | 198.7491 | 115.0486 | 38.3609 |
| y (m) | 0.0000 | 12.0000 | 6.3123 | 4.0605 |
| vx (m/s) | 8.2760 | 34.6099 | 20.5184 | 4.5683 |
| vy (m/s) | -9.7484 | 9.9242 | 0.0374 | 3.1521 |
| heading (rad) | -0.5332 | 0.5620 | 0.0006 | 0.1605 |

Overall raw range: `[-9.7484, 198.7491]`. The mixer already applies internal
`LayerNorm(state_dim)` (`marl/qmix.py:60`), so optional pre-mixer normalization
stays default OFF and, when enabled, must receive explicit `state_mean` /
`state_std` rather than hard-coded highway groups.

## Bugs to fix in stage 2

1. `StateNormalizer` must not be constructed when `state_norm=False`; when
   enabled it must accept explicit scales and not assume a multiple of 5.
2. `tests/test_qmix.py:173` must read `info["per_agent"]` and derive dims from
   `env.obs_dim`.
3. VDN needs its own checkpoint tests and the audit fixes applied identically.
4. Trainer must use `info["team_reward"]`, store only true `terminated`, and
   include warmup/cadence.
5. Add a clear-error test for enabling state normalization with a mismatched
   dimension.

## Open QUESTIONABLE items

- QMIX and VDN expose different default `reward_scale` (1.0 vs 0.1); the trainer
  will pass one identical value, but defaults differ.
- Current action masks are stored but not sampled by either buffer; harmless for
  the TD target but should be made explicit.
- `torch.use_deterministic_algorithms(True)` may slow the CPU smoke run; revisit
  if timing is unacceptable.
