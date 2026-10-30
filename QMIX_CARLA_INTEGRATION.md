# QMIX to CARLA integration

## What this checkpoint can execute

The Highway-Env trainer consumes the existing `HighwayV2XEnv` interface without
changing the adapter. A trained checkpoint therefore expects **25 float32 values
per agent** and uses four agents, 20 global-state values for training, and six
actions. The currently available project reference, `docs/env_spec.md`, describes
a proposal with approximately 56 local observation values and six global values
per CAV (24 total); it does not define every field's exact order or normalization.
Consequently, a QMIX checkpoint trained on the current Highway adapter is **not
directly loadable into a CARLA policy emitting the proposed 56-value vector**.
CARLA must either produce the same 25-value Highway-compatible observation, or
the team must agree on and train with a shared feature encoder/schema first.
The mixer state dimension does not constrain deployment because the mixer is not
used by the decentralized policy.

The requested `SpecV2X` attachment was not present in the repository. This guide
uses the available `docs/env_spec.md` and the observed Highway adapter settings;
any newer authoritative specification should supersede these proposed mappings.

## Parameters to transfer

Transfer only the **online shared agent network** from the `agent_q` entry in
`qmix_highway.pt` for decentralized execution. The checkpoint also stores:

- The target agent network, online and target mixers, optimizer state, and
  dimensions/training hyperparameters for resuming training.
- Agent and mixer weights trained jointly under centralized training.

Do not deploy the target network or mixer for normal action selection. The one
shared agent network is run once per CAV using that CAV's local observation and
fixed identity index (0 through 3). The four agents share weights but receive
different one-hot identity inputs. Keep the agent ordering consistent with
training (`cav_0`, `cav_1`, `cav_2`, `cav_3`).

## Exact trained network and tensor dimensions

With the current Highway adapter, a single local observation has shape `(5, 5)`.
The trainer flattens it in C row-major order to `(25,)`, and batches four agents
as `(batch, 4, 25)`. The shared agent network appends a four-value agent one-hot,
so its input is 29 values:

1. `Linear(29, 128)`, ReLU
2. `Linear(128, 128)`, ReLU
3. `Linear(128, 6)` (unmasked action values)

The online agent network output has shape `(batch, 4, 6)`. At inference, mask
invalid actions with negative infinity and choose the largest remaining Q-value.
Exploration is disabled for deployment.

The training-only mixer takes four selected per-agent Q-values `(batch, 4)` and
the existing adapter's global state `(batch, 20)`. Its first state-conditioned
state input is LayerNorm-normalized, its first mixing layer has 32 units, and its
second layer has one output. State-conditioned generated mixing weights use
`softplus` to stay nonnegative. It is trained with the agent network but is not
called during execution. This satisfies QMIX's monotonicity constraint.

## Observations and normalization

### Input schema used for the Highway-trained checkpoint

The active Highway-Env Kinematics configuration was inspected at runtime:

- Shape: `(5, 5)`; flatten to 25 values without reordering.
- Rows: ego vehicle first, then other visible vehicles in Highway-Env's
  `sorted` order; unused rows are zero/presence-masked by that observation type.
- Columns: `[presence, x, y, vx, vy]`.
- Coordinates and velocities are relative (`absolute=False`).
- Normalization and clipping are enabled. Highway-Env uses feature ranges
  `x: [-200, 200]`, `y: [-16, 16]`, `vx: [-80, 80]`, `vy: [-80, 80]`.
  Reproduce the same Highway-Env Kinematics normalization semantics; do not
  normalize these values a second time.

These are **not** the proposed CARLA observation blocks. In the proposal, the
block order is ego (about 8), route (about 6), six candidate rows (6 values per
candidate, including validity; 36 total), then six action-mask values. The listed
ego and route fields do not fully specify 8 and 6 ordered scalar slots, and the
normalization constants are not given. Those details must be frozen before
training a CARLA-compatible 56-input agent network. Do not silently pad, truncate,
or guess those features to use this 25-input checkpoint.

Action masks are passed separately to `act`; they are not appended to the
Highway-trained checkpoint's 25-value input.

### Global state used by training

The current adapter returns four concatenated records of
`[x, y, vx, vy, heading]`, in fixed agent order, for 20 float32 values total.
Those are native Highway values and are used only by the mixer during training.
The proposal instead describes six values per CAV `[position, velocity, heading,
goal distance]` (24 total); it does not set units, coordinate frame, or
normalization. A CARLA state with that alternative shape requires a newly
configured/retrained mixer, but does not change the local policy input contract.

## CARLA environment interface

Use CARLA **0.9.16** with synchronous world mode and
`fixed_delta_seconds = 0.05`. Enable Traffic Manager synchronous mode and seed
its random device with the episode seed. Make one policy decision every four
simulation ticks (5 Hz). The QMIX code itself has no CARLA dependency.

Implement a parallel multi-agent API with the following behavior:

- `reset(seed)` seeds spawn points, routes, goals, background traffic, and the
  Traffic Manager, then returns `(observations, info)`. `observations` is a
  dictionary keyed by the four stable agent IDs; each value must have exactly the
  trained local feature shape and ordering.
- `step(actions)` accepts one integer action in `[0, 5]` per agent, advances
  exactly four synchronous ticks, and returns
  `(observations, rewards, terminated, truncated, info)`. Each per-agent value
  is keyed by agent ID. A collision ends the episode as termination, all agents
  reaching their goals is successful termination, and the time limit is
  truncation. Preserve the final observation and info on the ending step.
- `state()` returns the concatenated centralized training state in fixed agent
  order. It is only needed by a trainer, not the deployed agent network.
- `action_masks()` returns a per-agent six-element boolean mask. At least one
  action must remain valid for every agent. In particular, mask lane changes
  without a legal adjacent lane.

At inference, pass only each agent's own observation and mask to the shared
network. Do not pass `state()` to the policy.

## Six action IDs and CARLA controls

Keep the specification's action IDs unchanged:

| ID | Meaning | Low-level CARLA controller behavior |
|---:|---|---|
| 0 | Keep speed | Retain current target speed and route-following control |
| 1 | Accelerate | Increase target speed by the configured increment |
| 2 | Decelerate | Decrease target speed by the configured increment |
| 3 | Emergency brake | Apply braking and suspend acceleration for this decision |
| 4 | Lane change left | Request a legal left-lane change from the route/PID controller |
| 5 | Lane change right | Request a legal right-lane change from the route/PID controller |

Use the local planner/PID to turn these meta-actions into throttle, brake, and
steer. Do not change their IDs or action-output order. This matches the action
semantics in the Highway adapter, which translates these IDs to Highway-Env
native meta-actions.

## Loading and executing the policy

The `QMIX` loader validates checkpoint dimensions. A minimal Python execution
loop for the current 25-value contract is:

```python
import numpy as np
from marl.qmix import QMIX

policy = QMIX.from_checkpoint("qmix_highway.pt", device="cpu")
observations, info = env.reset(seed=42)

while True:
    agent_ids = ("cav_0", "cav_1", "cav_2", "cav_3")
    local_obs = np.stack(
        [np.asarray(observations[agent]).reshape(-1) for agent in agent_ids]
    )
    masks_by_agent = env.action_masks()
    masks = np.stack([masks_by_agent[agent] for agent in agent_ids])
    actions = policy.act(local_obs, masks, epsilon=0.0)
    observations, rewards, terminated, truncated, info = env.step(
        {agent: int(actions[i]) for i, agent in enumerate(agent_ids)}
    )
    if any(terminated.values()) or any(truncated.values()):
        break
```

In CARLA, substitute the agreed 25-value-compatible observation producer and
CARLA environment methods. The example deliberately does not call `env.state()`
at execution.

## Centralized training vs decentralized execution

- **Centralized training (CTDE):** QMIX receives local observations at the shared
  agent network and the global state at the mixer. The mixer combines selected
  local Q-values into `Q_tot` and is used by TD updates and target calculation.
- **Decentralized execution:** each CAV runs the shared agent Q-network using
  only its local observation, its fixed identity, and its legal-action mask.
  Neither global state, mixer, target networks, replay buffer, nor optimizer is
  required in the vehicle.

## Metrics and known limits

The current Highway adapter supports episode reward, collision flags, speeds,
step count, and duration; the training script logs these and creates QMIX/VDN
learning plots. It does not expose success/goal flags, TTC, jerk, V2X message
counts, or selected teammate IDs. These are recorded as `NA`, not estimated.
CARLA should add those measurements to its `info` contract before claiming those
specification metrics.
