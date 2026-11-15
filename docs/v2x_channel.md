# V2X channel

## Purpose

`envs/v2x_channel.py` is a pure-NumPy model of the V2X broadcast layer described in
`docs/env_spec.md` section 4. It decides which messages each CAV receives under a
configurable communication range, packet loss and delay. It does **not** know which
vehicles are CAVs and which are background traffic: the caller passes the senders for
each decision step. It imports neither `carla` nor `highway_env`, adds no dependencies,
and is not wired into any environment on this branch.

The received messages are optionally converted by `messages_to_candidates(...)` into the
candidate row layout used by `envs/obs_builder.py`, so the existing grouping module can
consume them through the normal candidate interface.

## Message format

Every sender broadcasts six values:

```text
[x, y, vx, vy, heading, intent]
```

* `x, y` — world position in metres.
* `vx, vy` — world velocity in m/s.
* `heading` — world yaw in radians.
* `intent` — the sender's current action id, `0..5` (stored as a float inside the
  message). Actions are those of `envs/base.py` / `docs/env_spec.md` section 2.

Positions used for the range check always come from the message itself.

## Parameters and defaults

`V2XConfig` is a frozen dataclass:

| Field | Default | Validation | Robustness values |
|---|---|---|---|
| `range_m` | `50.0` | finite and `>= 0` | `30`, `100` |
| `delay_steps` | `0` | integer and `>= 0` | `1` to `3` decision steps |
| `loss_prob` | `0.0` | finite and `0 <= p <= 1` | `0.1` to `0.3` |

Invalid values raise `ValueError` at construction time. Booleans are rejected for
`delay_steps`; floats with a fractional part are rejected too.

## Semantics

* **Range.** Euclidean distance is measured at transmission time from the sender and
  receiver positions carried in their current messages. A distance equal to `range_m`
  is inside the range (`distance <= range_m`); anything strictly beyond is dropped.
* **Loss.** One independent Bernoulli draw per directed `(sender, receiver, step)`
  transmission. The draw happens at transmission time, before the delay queue. A sender
  never receives its own message. With `loss_prob=0` no random draw is made; with
  `loss_prob=1` every in-range message is dropped.
* **Delay.** `delay_steps` is measured in **decision steps** (not CARLA simulation
  ticks). A message sent at step `t` is queued with due step `t + delay_steps` and is
  delivered at `t + delay_steps` with `age = delay_steps`. Its six values are the stale
  values captured at step `t`. Nothing arrives before its due step.
* **Receivers.** Receivers are the senders supplied in the same `step()` call. The
  returned dictionary has an entry for every current sender (possibly an empty list).
  Pending messages are delivered to their addressed receiver even if that receiver is
  not broadcasting in the current call, so a caller can flush the channel with an empty
  sender list.
* **Reset.** `reset(seed=None)` clears all in-flight messages and all per-episode
  counters, restarts the step index at zero, and recreates the RNG. An explicit seed
  becomes the new replay seed; `reset(None)` replays the seed last used (initial or
  explicit). Messages still in flight at reset are discarded, never delivered.
* **Determinism.** The channel owns a `numpy.random.Generator` (PCG64) and never uses
  the global NumPy RNG. The same seed and the same sequence of sender lists give
  identical received messages, ages and statistics. Different seeds change the loss
  pattern (when `loss_prob > 0`).
* **Ids.** Sender ids must be unique within one `step()` call. Ids only need to be
  hashable; they are never used in numeric computations. Sorting ties in the conversion
  helper prefer numeric ids, then strings.
* **Counts.** Per step, `sent` counts directed transmissions considered, i.e.
  `n * (n - 1)` for `n` senders, excluding self. `dropped_by_range` and
  `dropped_by_loss` partition the drops among those transmissions. `delivered` counts
  messages actually arriving in that step, which may have been sent earlier when
  `delay_steps > 0`. The extra `in_flight` key is the cumulative number of accepted
  messages still queued after the step.

## API and minimal example

```python
import numpy as np

from envs.v2x_channel import (
    V2XChannel,
    V2XConfig,
    messages_to_candidates,
)

channel = V2XChannel(
    V2XConfig(range_m=50.0, delay_steps=1, loss_prob=0.1),
    seed=7,
)

# Once per decision step, outside the per-agent loop.
senders = [
    ("cav_0", np.array([0.0, 0.0, 10.0, 0.0, 0.0, 1.0])),
    ("cav_1", np.array([12.0, 0.0, 10.0, 0.0, 0.0, 0.0])),
]
received, stats = channel.step(senders)

# Per agent, convert its received messages to the candidate row layout.
ego_state = {"x": 0.0, "y": 0.0, "vx": 10.0, "vy": 0.0, "yaw": 0.0}
candidate_rows = messages_to_candidates(received["cav_0"], ego_state)
```

Public objects:

* `V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0)` — frozen, validated config.
* `V2XChannel(config=None, seed=None)` — owns the RNG, delay buffer and counters.
* `channel.reset(seed=None)` — reset the stream and discard in-flight messages.
* `channel.step(senders)` — returns `(received, stats)`.
* `channel.episode_stats()` — aggregated counters and mean age.
* `channel.step_index`, `channel.in_flight` — read-only diagnostics.
* `ReceivedMessage(sender_id, age, message)` — named tuple returned per delivery.
* `messages_to_candidates(received, ego, m=M_CANDIDATES)` — `(m, 6)` float32 candidate
  block.

`received` is `dict[receiver_id -> list[ReceivedMessage]]`. `stats` contains `sent`,
`dropped_by_range`, `dropped_by_loss`, `delivered` and `in_flight`.

`episode_stats()` returns:

```text
sent, delivered, dropped_by_range, dropped_by_loss, dropped,
mean_age, in_flight
```

`mean_age` is `sum(age) / delivered` in decision steps and is `0.0` when nothing was
delivered. All counters reset on `reset()`. These map to the "V2X messages received and
dropped" metric in `docs/env_spec.md` section 10.

## Conversion to the candidate layout

`messages_to_candidates(received, ego, m=M_CANDIDATES)` uses the same layout and
normalization as `envs/obs_builder.py`:

```text
[rel_x / 50, rel_y / 50, rel_vx / 30, rel_vy / 30,
 wrapped_rel_heading / pi, valid]
```

* `ego` is a mapping with `x, y, vx, vy` and `yaw` (or `heading`), or a sequence
  `[x, y, vx, vy, heading]`.
* Positions and relative velocities are rotated into the ego frame with
  `obs_builder.to_ego_frame`; the heading difference is wrapped to `[-pi, pi]`.
* Values are clipped to `[-1, 1]` and the block is float32.
* The nearest `m` messages to the ego are kept; ties are broken by sender id.
* Missing slots are all-zero with `valid = 0`; present rows have `valid = 1`.
* `intent` is not one of the six candidate features and is therefore not copied into
  the row. See the open questions below.

The channel itself applies the range check, so the helper does not re-filter by
`range_m`. Note that `obs_builder.CANDIDATE_RANGE` is 50 m while the channel range can be
100 m; messages between 50 m and 100 m will saturate the normalized relative position
columns at `+/-1`. This is an integration question, not changed here (see below).

## How it plugs in later

The channel is intentionally not wired into any env on this branch. A later adapter can
expose it through a `get_candidates(agent_id)` hook:

```python
def get_candidates(self, agent_id):
    senders = [
        (aid, self._v2x_message(aid))
        for aid in self._broadcasters()   # caller decides whether background vehicles count
    ]
    received, stats = self._channel.step(senders)  # once per decision step
    self._last_v2x_stats = stats
    ego = self._ego_state(agent_id)
    return messages_to_candidates(received.get(agent_id, []), ego)
```

`step()` advances the channel clock and consumes RNG, so it must be called exactly once
per decision step, not once per agent. A per-step cache can return the same `received`
mapping to every agent. The env remains responsible for deciding which vehicles
broadcast and for the shape of the ego state.

## Tests

`tests/test_v2x_channel.py` covers:

* config validation and frozen behaviour;
* inclusive range at exactly 30, 50 and 100 m, and rejection just beyond;
* empirical loss rates for 0.1, 0.2 and 0.3 over 20,000 directed messages, plus exact
  0 and 1 extremes;
* delay 0, 1, 2 and 3: arrival step, age, stale content and no early arrival;
* determinism, seed sensitivity, reset replay and in-flight discard;
* no self-reception, out-of-range exclusion, count invariants;
* conversion layout, ego-frame transform, zero padding and nearest-M selection;
* combined range + loss + delay behaviour.

Run:

```powershell
python -m pytest tests/test_v2x_channel.py -v
python -m pytest tests -k "not carla and not toy" -q
```

## Status

* Stage 1 — spec and contract reading, ambiguities/defaults recorded above.
* Stage 2 — implementation committed as `feat: V2X channel` (`7272ff3`).
* Stage 3 — tests committed as `test: V2X channel` (`90ebded`);
  `tests/test_v2x_channel.py` 37 passed; full non-CARLA/non-toy suite
  173 passed, 4 skipped, 33 deselected.
* Stage 4 — this document committed as `docs: V2X channel`.
* Scope respected: only `envs/v2x_channel.py`, `tests/test_v2x_channel.py` and
  `docs/v2x_channel.md`; no `docs/PROGRESS.md` edit, no env wiring, no changes to
  existing candidate code.

## Open questions for the CARLA partner

1. **Background vehicles.** Should scripted background vehicles broadcast V2X messages,
   or only the CAVs? The channel supports either; the caller currently decides.
2. **Intent in the candidate row.** The message carries the sender's current action
   (0..5), but `obs_builder` candidate rows have only six features and no intent column.
   Should intent be exposed to grouping (for example as an extra feature or a separate
   block), and if so what is the agreed layout/version?
3. **Delay units.** The spec table says "ticks" while the robustness column and this
   implementation use decision steps. Confirm that delay should be counted in 5 Hz
   decision steps rather than 20 Hz simulation ticks (and whether the channel is ticked
   once per decision or once per sim tick).
4. **Loss timing under delay.** Should loss be drawn at transmission time (current
   choice) or at delivery time after the delay?
5. **Range vs candidate range.** With `range_m=100`, should candidates beyond the 50 m
   `obs_builder.CANDIDATE_RANGE` be kept, clamped (current helper behaviour), or should
   the candidate normalization range change for the robustness runs?
