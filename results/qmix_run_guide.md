# QMIX branch audit and run guide

Audited `Qmix` at `42889f7` (`origin/Qmix` resolves to the same commit). The
working tree already had unrelated untracked result files; this report is the
only file created for this audit. No CARLA process was started.

## 1. Repository files

### `marl/`

| File | Purpose |
|---|---|
| `marl/__init__.py` | Empty package marker. |
| `marl/qmix.py` | QMIX shared per-agent Q network, state-conditioned mixer, state/mask replay buffer, learner, action selection, update and checkpoint methods. [`qmix.py:14-42,44-99,101-207,210-454`] |
| `marl/vdn.py` | VDN learner: shared agent Q network, sum mixer, Double-Q update, soft target update, save. [`vdn.py:1-50`] |
| `marl/networks.py` | Shared Q network used by VDN; appends agent identity one-hot. [`networks.py:6-22`] |
| `marl/buffers.py` | Circular VDN replay buffer with observations/actions/team reward/true-terminal/next mask. [`buffers.py:6-30`] |
| `marl/mappo.py` | Empty placeholder; no MAPPO implementation. **UNVERIFIED** why it is retained. |

### `scripts/`

| File | Purpose / output |
|---|---|
| `scripts/train_toy.py` | VDN-only toy trainer. Writes `vdn_toy_log.csv`, `vdn_toy.pt`, `vdn_toy_curve.png` under `--out`; periodic greedy evaluation is included. [`train_toy.py:1-3,70-128`] |
| `scripts/train_qmix_highway.py` | Trains QMIX and/or VDN on Highway-Env, not ToyEnv. Writes Highway checkpoints, CSVs and plots described in §2. [`train_qmix_highway.py:121-244,247-323,327-370`] |
| `scripts/train.py` | Empty; no generic trainer. |
| `scripts/evaluate.py` | Empty; no standalone checkpoint evaluator. |
| `scripts/test_env.py` | CARLA server smoke script (100 ticks, resets, random actions); it starts CARLA when run. Do not use for the no-CARLA test command. [`test_env.py:1-37`] |
| `scripts/test_highway.py` | Visual Highway-Env scripted simulation; no files written. [`test_highway.py:15-85`] |
| `scripts/make_report.py` | Empty; no report generator. |
| `scripts/demo_qmix_highway.py` | Loads QMIX checkpoint and runs Highway-Env greedy demonstration; no files written. [`demo_qmix_highway.py:16-90`] |
| `scripts/demo_carla.py` | Interactive CARLA demo; not needed for this guide and starts CARLA on execution. [`demo_carla.py:11-23,40-64`] |
| `scripts/check_api.py` | Manually checks ToyEnv by default; supplying `carla` imports/starts CARLA. Prints API shapes; no files written. [`check_api.py:6-25`] |

### `tests/`

| File | Purpose |
|---|---|
| `tests/test_qmix.py` | QMIX network shape, mixer monotonicity, action masks, replay sampling, update, checkpoint round-trip/dimension rejection; includes optional Highway interaction test that skips when dependencies are missing. [`test_qmix.py:8-140,143-176`] |
| `tests/test_toy_env.py` | Toy reset/reproducibility, movement, invalid action, state dimensions, collision detection. [`test_toy_env.py:8-104`] |
| `tests/test_reward.py` | `RewardCalculator` behavior and invalid inputs. [`test_reward.py:8-93`] |
| `tests/test_collision_monitor.py` | Fake-world collision sensor record/reset/error/cleanup behavior. [`test_collision_monitor.py:8-124`] |
| `tests/test_grouping.py` | **UNVERIFIED** exact coverage; file is empty on this branch. |

## 2. Commands, arguments, and output files

Run from repository root in PowerShell. The venv Python is:

```powershell
$py = 'C:\Users\souro\carla-env\Scripts\python.exe'
```

### Tests

```powershell
& $py -m pytest tests -q -k "not carla and not highway"
```

The filter skips tests whose names contain `carla` or `highway`; it does not
guarantee safety if another selected test starts a simulator. The visible CARLA
script is `scripts/test_env.py`, not under `tests/`. [`tests/test_qmix.py:143-176`]

### Toy VDN training (the only toy trainer present)

```powershell
& $py scripts\train_toy.py --episodes 2000 --seed 0 --batch 128 --warmup 2000 --eval_every 50 --reward_scale 0.1 --out results\vdn_toy
```

Arguments and defaults: `--episodes 2000`, `--seed 0`, `--batch 128`,
`--warmup 2000`, `--eval_every 50`, `--reward_scale 0.1`, `--out results`.
The command overrides `--out` only so its generated files do not overwrite
the existing `results/vdn_toy_*` evidence. Outputs are
`results/vdn_toy/vdn_toy_log.csv`, `results/vdn_toy/vdn_toy.pt`,
`results/vdn_toy/vdn_toy_curve.png`. [`train_toy.py:70-78,115-128`]

The script seeds NumPy and Torch from `--seed`; episode seeds are `seed + ep`,
and greedy evaluation seeds are fixed at 20000–20019. [`train_toy.py:81,100-110`]

### QMIX on the same toy env / paired toy VDN

**No runnable command exists in this checkout.** There is no QMIX ToyEnv trainer.
`train_qmix_highway.py` only constructs `HighwayV2XEnv` [`train_qmix_highway.py:16,127-139`],
and the toy environment is two agents with four observation values and three
actions [`toy_env.py:7-18`], while `QMIX` rejects any `n_actions != 6`
[`qmix.py:217-243`]. Therefore it is not valid to claim same-seed toy QMIX/VDN
curves from the current scripts. To compare on toy, the owner must add a toy
QMIX trainer and decide whether to extend QMIX to the toy's three actions or
adapt ToyEnv to six. The toy VDN script above is the available baseline only.

For reference, the existing Highway trainer’s complete CLI (not a toy command)
is: `--algorithms qmix vdn` (choices qmix/vdn, default both), `--episodes 300`,
`--seed 0`, `--duration 60.0`, `--background-vehicles 0`,
`--replay-capacity 50000`, `--warmup 1000`, `--batch-size 128`,
`--hidden-dim 128`, `--learning-rate 0.0003`, `--gamma 0.99`, `--tau 0.01`,
`--reward-scale 0.1`, `--epsilon-end 0.05`, `--epsilon-fraction 0.7`,
`--log-every 1`, `--device cpu`, `--torch-threads 1`,
`--out results/qmix_highway`. [`train_qmix_highway.py:327-347`]

```powershell
& $py scripts\train_qmix_highway.py --algorithms qmix vdn --episodes 300 --seed 0 --duration 60 --background-vehicles 0 --replay-capacity 50000 --warmup 1000 --batch-size 128 --hidden-dim 128 --learning-rate 0.0003 --gamma 0.99 --tau 0.01 --reward-scale 0.1 --epsilon-end 0.05 --epsilon-fraction 0.7 --log-every 1 --device cpu --torch-threads 1 --out results\qmix_highway
```

It writes `qmix_highway.pt`, `vdn_highway.pt`,
`highway_training_metrics.csv`, `qmix_parameter_evolution.csv`,
`highway_learning_curves.png`, `training_stability.png`,
`qmix_parameter_evolution.png` (the last CSV/PNG is QMIX-only and may be empty
or omitted if `qmix` is not selected). [`train_qmix_highway.py:228-240,247-253,289,307,323,364-368`]

### Evaluation

There is no `scripts/evaluate.py` implementation (empty file). For a visual
Highway checkpoint demo, the available command is:

```powershell
& $py scripts\demo_qmix_highway.py --checkpoint results\qmix_highway\qmix_highway.pt --seed 42 --steps 300 --background-vehicles 0 --headless --device cpu
```

Defaults: `--checkpoint results/qmix_highway/qmix_highway.pt`, `--seed 42`,
`--steps 300`, `--background-vehicles 0`, `--headless` false (flag),
`--device cpu`. It writes no output files. [`demo_qmix_highway.py:16-24,28-90`]

## 3. Contract and dimensions

- `docs/env_spec.md` in this ref ends at §11 (lines 116–126); Contract §12 is
  **absent on the audited branch**, so §12 requirements cannot be verified from
  this ref. The requested 56-D/24-state target is present in the integration
  guide as a proposal, not a frozen §12 contract. [`docs/env_spec.md:116-126`,
  `QMIX_CARLA_INTEGRATION.md:5-20`]
- QMIX’s implementation accepts dimensions as constructor parameters; it does
  not hard-code obs 25, state 20, or 4 agents. It does require exactly six
  actions. The Highway script derives dimensions from env observations/state/
  masks and checks four agents/six actions. [`qmix.py:217-247`,
  `train_qmix_highway.py:133-160`]
- The current Highway adapter is 4 agents, 25 local values (5×5 flattened),
  20 state values, 6 actions. CARLA candidate config is 4, 56, 24, 6. The
  network can be constructed with `QMIX(n_agents=4, obs_dim=56, state_dim=24,
  n_actions=6)`; a checkpoint trained at 25/20 will fail the dimension check.
  Retrain with CARLA features; agree/freeze exact 56-slot order, scaling, and
  mask representation first. [`QMIX_CARLA_INTEGRATION.md:5-16,37-57,75-94`,
  `qmix.py:414-423`]
- `train_qmix_highway.py` calls `env.state()` for stored states and
  `env.action_masks()` for current and next masks. Thus it uses centralized
  state during training, as a CTDE mixer requires; `act()` receives only local
  observations and masks. The Highway demo checks state size but does not use
  state for action choice. [`train_qmix_highway.py:58-76,133-137`,
  `qmix.py:210-215,278-315`, `demo_qmix_highway.py:39-64`]
- Masks are not part of the Q network input. `agent_q` emits unmasked action
  values; `act()` masks only greedy selection and samples exploration from valid
  actions. The update masks online next-action selection. [`qmix.py:14-42,301-315,337-347`]

## 4. QMIX correctness review

- **Mixer:** state-normalized hypernetworks generate both weight matrices;
  `softplus` makes each nonnegative. ELU between layers and the state-value
  baseline produce a monotonic `Q_tot`. This is the expected monotonicity
  construction. [`qmix.py:58-99`]
- **Double-Q:** online utilities select masked next actions; target utilities
  evaluate those actions and target mixer evaluates the joint value.
  [`qmix.py:337-348`]
- **Target update:** target networks start as online copies and receive a soft
  parameter update after each optimizer step, `lerp(..., tau)`.
  [`qmix.py:257-275,361-373`]
- **Loss:** Smooth L1 / Huber TD loss; gradients are clipped to norm 10.
  [`qmix.py:349-364`]
- **Reward scaling:** QMIX defaults `reward_scale=1.0`; Highway trainer defaults
  to 0.1. Target is `reward * scale + gamma * (1-terminated) * target_Q`.
  No reward normalization beyond multiplication. [`qmix.py:227-253,349-352`,
  `train_qmix_highway.py:341`]
- **Termination vs truncation:** replay stores only `terminated`, no truncated
  field. The Highway trainer records `terminal = any(terminated)` and ends on
  either termination or truncation, so truncations continue bootstrapping.
  This is appropriate when truncation is only a time limit and the next state is
  valid; verify against §12 (not present here). [`qmix.py:129-143,169-177,349-352`,
  `train_qmix_highway.py:77-103`]
- **Episode-end bootstrap:** true terminal episodes suppress bootstrap. Time
  limit truncations do not suppress it; the trainer stores the final next state
  and next masks before ending. Toy VDN likewise distinguishes `terminal` from
  `ended` in the loop, but only VDN exists there; this is a deliberate time-limit
  bootstrap pattern, not an episode-end zero bootstrap. [`train_qmix_highway.py:70-110`,
  `train_toy.py:58-67`, `vdn.py:31-39`]
- **Reproducibility caveat:** QMIX’s replay sampling uses global `np.random`,
  not its learner-local RNG; trainers reset global NumPy/Torch per algorithm,
  but comparison fairness depends on whether both runs use independent identical
  seeding and on environment version/determinism. [`qmix.py:190-193,255`,
  `train_qmix_highway.py:125-126`]

## 5. Phase 1 evidence and gap

“Done when” should mean a fixed toy benchmark with identical episode/evaluation
seeds, training budget, reward scale, and comparable exploration/evaluation;
then report held-out greedy return curves (mean and variability over multiple
training seeds) showing QMIX at least matches VDN. This operationalizes the
requested criterion; **UNVERIFIED** whether the owner intended a numerical
threshold or statistical test.

Evidence present: `results/vdn_toy_log.csv`, `vdn_toy_curve.png`, and
`vdn_toy.pt` are present but untracked; the toy log contains VDN return,
epsilon, and periodic greedy evaluation. `results/qmix_highway/` contains
QMIX/VDN Highway metrics, weights, and plots; this is a different environment,
not evidence QMIX matches VDN on toy. [`train_toy.py:108-127`,
`train_qmix_highway.py:197-215,364-368`]

Evidence missing: a toy QMIX trainer, paired same-seed toy QMIX/VDN results,
multiple training-seed variation, clear held-out comparison summary, and
evaluation script. The old `results/train_log.txt` records a VDN training
interruption at `KeyboardInterrupt`, so do not treat that log as a successful
complete run. Current repository evidence cannot establish the criterion.

## 6. Risks and proposed CARLA interface test

Risks: no frozen §12 on this ref; 56-D feature order/normalization is not
specified; Highway checkpoint weights cannot load into 56/24 dimensions; state
is training-only but must match the mixer’s 24-D expectation if training in
CARLA; package versions and CARLA client/server compatibility are **UNVERIFIED**;
no learned-policy CARLA performance evidence is present. [`QMIX_CARLA_INTEGRATION.md:5-20,75-101`,
`qmix.py:414-423`]

Proposed script (report only; do not run during this audit): instantiate the
model, verify masked forward/action selection, then run 100 legal random actions
through the CARLA env and verify observations/masks/state shapes and termination
API. This is an interface smoke test, not a quality evaluation.

```python
import numpy as np
from marl.qmix import QMIX
from envs.carla_env import CarlaMultiAgentEnv

env = CarlaMultiAgentEnv(town="Town03", n_agents=4)
try:
    model = QMIX(n_agents=4, obs_dim=56, state_dim=24, n_actions=6, seed=0)
    observations, info = env.reset(seed=0)
    agents = list(env.agents)
    assert len(agents) == 4
    assert np.asarray(env.state()).shape == (24,)
    for _ in range(100):
        obs = np.stack([np.asarray(observations[a], np.float32) for a in agents])
        masks_by_agent = env.action_masks()
        masks = np.stack([np.asarray(masks_by_agent[a], bool) for a in agents])
        assert obs.shape == (4, 56) and masks.shape == (4, 6)
        actions = model.act(obs, masks, epsilon=1.0)
        observations, rewards, terminated, truncated, info = env.step(
            {a: int(actions[i]) for i, a in enumerate(agents)}
        )
        if any(terminated.values()) or any(truncated.values()):
            break
finally:
    env.close()
```

## 7. Questions for the owner

1. Is `docs/env_spec.md` §12 intentionally absent from `origin/Qmix`, and which
   exact document/ref is authoritative for the contract?
2. Should the toy benchmark be made 6-action/four-agent to match CARLA, or should
   QMIX support ToyEnv’s current 3-action/two-agent fixture?
3. Should a new paired toy runner train both algorithms from identical seeds
   and evaluation seeds, and what statistical/quantitative meaning does “at
   least matches VDN” require?
4. Should timeout truncation bootstrap from the final state for the intended
   benchmark, or should both learners suppress bootstrap at truncation?
5. What is the frozen exact order, frame, units, and normalization for all 56
   local observation values and 24 global-state values?
6. Which checkpoint format should be used for VDN deployment? The toy VDN save
   writes only the Q state dict, while Highway VDN writes a dimensioned wrapper;
   no generic evaluator/loader is present. [`vdn.py:49-50`,
   `train_qmix_highway.py:233-240`, `scripts/evaluate.py` (empty)]
