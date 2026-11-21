# Technical Audit Specification — V2X Teammate Selection + CARLA Integration

## Executive Summary

This report documents a read-only deep technical audit of the V2X-MARL repository conducted on `main` at commit `f1156fa8d1e541497cae270ff999356a0a811b35` (also `origin/main`). The repository contains a 56-D CARLA observation stack, VDN/QMIX/MAPPO learners, and a pure-NumPy V2X channel, but the critical research hooks are unimplemented:

- **Grouping module** (`grouping/*.py`) is 0 bytes — no radius, random, or learned selectors.
- **V2X integration** is isolated: `envs/v2x_channel.py` models range/loss/delay but is not wired into the CARLA environment.
- **Candidate source** uses raw CARLA world actors instead of V2X-delivered messages, meaning range/delay/loss cannot affect policy inputs.
- **No fixed-k padding**, no selected-ID logging, no deterministic tie-breaking.

The audit establishes 23 open decisions (D1–D10 from `V2X_INTEGRATION_DECISIONS.md` plus A1–A7 from the repository audit) that must be resolved before any research claim can be made. The minimum viable experiment requires freezing the interface, implementing three grouping variants with fixed `k=2`, integrating the V2X channel, adding scenarios and background traffic, and running paired-seed evaluations on held-out seeds/towns.

**Verified facts:**
- 56-D observation: ego(8) + route(6) + candidates(36, 6×6) + mask(6)
- V2X channel: range_m=50 default, delay_steps in decision steps, Bernoulli loss, tested 37/37 tests passing
- Grouping files empty; no selector implementation exists
- Local `main` and `origin/main` share commit `f1156fa`
- No `feat/v2x-channel` branch exists; V2X channel was implemented directly on `main`

**Unsafe claims to avoid:** Learned selection outperforms radius/random; V2X provides robustness; Highway results generalize to CARLA; full test suite passes without qualification.

---

## Exact Audited Git Baseline

| Item | Value |
|---|---|
| Working directory | `C:\Users\souro\Desktop\V2X-MARL` |
| Current branch | `main` |
| HEAD commit | `f1156fa8d1e541497cae270ff999356a0a811b35` |
| `origin/main` commit | `f1156fa8d1e541497cae270ff999356a0a811b35` (identical) |
| Local/remote agreement | Yes — local main points to same commit as origin/main |
| `feat/v2x-channel` branch | Does not exist locally or as remote-tracking branch |
| Other branches | `Qmix` (local), `feat/env-and-vdn`, `feat/phase1-core`, `feat/highway-adapter-56d`, `feat/highway-contract`, `test/qmix-on-carla` (all remote) |
| Uncommitted changes | None (8 untracked files under `results/`, all diagnostic/audit outputs) |
| Test results (non-CARLA, non-Highway) | 164 passed, 14 skipped, 16 deselected |

---

## Repository Architecture

The repository has three principal layers:

### 1. Environment Layer (`envs/`)
- **`carla_env.py`**: `CarlaMultiAgentEnv` — synchronous CARLA 0.9.16 adapter with 4 CAVs, 20 Hz simulation, 5 Hz decision rate, route planning, termination, reward, action masks.
- **`obs_builder.py`**: Pure-Numpy 56-D observation builder from ego/route/candidate/actor_mask records.
- **`base.py`**: Shared contract (M_CANDIDATES=6, CAND_FEATURES=6, N_ACTIONS=6, observation contract).
- **`reward.py`, `termination.py`, `metrics.py`, `safety.py`**: Reward, termination, metrics, collision monitoring.
- **`v2x_channel.py`**: Pure-NumPy V2X broadcast channel with range, Bernoulli loss, fixed delay, `messages_to_candidates()` conversion helper. **Not wired into any environment.**

### 2. MARL Layer (`marl/`)
- **`vdn.py`**, **`qmix.py`**, **`mappo.py`**: CTDE learners with masked action selection, replay buffers, normalization.
- **`networks.py`**, **`buffers.py`**: Shared network architectures and training buffers.

### 3. Grouping Layer (`grouping/`)
- **`radius.py`**, **`random_sel.py`**, **`learned_sel.py`**: All **0 bytes** — no implementation.
- **`__init__.py`**: Likely empty or missing.
- **`tests/test_grouping.py`**: 0 bytes — no tests.

### Disconnected Halves (from `V2X_INTEGRATION_DECISIONS.md`)
- **CAV state → V2XChannel → received messages → messages_to_candidates()**: Implemented but isolated.
- **CARLA actors → get_candidates() (raw world) → obs_builder → policy**: Used by the environment, no V2X involvement.

The design intent (from `docs/env_spec.md` §1) is to keep the environment generic and put the research in the grouping module, with fixed `k=2` across radius/random/learned variants. The current boundary stops before grouping and before the V2X channel is used.

---

## Verified CARLA Environment Contract

| Parameter | Exact value / behavior | Source |
|---|---|---|
| CARLA version | 0.9.16, synchronous mode | `docs/env_spec.md:9`, `carla_env.py:90-96` |
| Town/scenario | Town03 (default), random 150 m routes | `carla_env.py:283-375` |
| Simulation frequency | 20 Hz (`SIM_DT=0.05`) | `carla_env.py:52` |
| Decision frequency | 5 Hz (`TICKS_PER_STEP=4`) | `carla_env.py:53` |
| Simulation ticks per decision | 4 | `carla_env.py:53` |
| Number of controlled CAVs | 4 (`N_AGENTS=4`) | `carla_env.py:48`, `env_spec.md:12` |
| Agent IDs | `cav_0`, `cav_1`, `cav_2`, `cav_3` | `carla_env.py:81` |
| Maximum candidate slots | 6 (`M_CANDIDATES=6`) | `obs_builder.py:12`, `base.py:15` |
| Candidate discovery radius | 50 m (`CANDIDATE_RANGE=50.0`) | `obs_builder.py:25`, `base.py:CANDIDATE_RANGE` |
| Observation dimension | 56 (`OBS_DIM=56`) | `obs_builder.py:16`, `carla_env.py:50` |
| Ego feature dimension | 8 | `obs_builder.py:10`, `EGO=8` |
| Route feature dimension | 6 | `obs_builder.py:11`, `ROUTE=6` |
| Features per candidate | 6 (`CAND_FEATURES=6`) | `obs_builder.py:13` |
| Candidate slot capacity | 6 × 6 = 36 | `obs_builder.py:14`, `CAND = 6*6` |
| Action mask dimension | 6 (`MASK=6`) | `obs_builder.py:15` |
| Action names and indices | KEEP=0, ACCEL=1, DECEL=2, BRAKE=3, LANE_LEFT=4, LANE_RIGHT=5 | `base.py:30-36`, `carla_env.py:70` |
| Observation dtype | `np.float32` | `obs_builder.py:171,175` |
| Observation normalization | All features clipped to [−1, 1] via `_clip()` | `obs_builder.py:38-39, 111-118, 124-131, 156-162, 165-175` |
| Observation clipping | `.clip(-1.0, 1.0)` after `nan_to_num` | `obs_builder.py:173-175` |
| Padding convention | Zero padding with `valid=0` for fewer than 6 candidates | `obs_builder.py:144-163`, `test_obs_builder.py:63-69` |
| Candidate ordering | By (distance, actor_id) ascending | `obs_builder.py:142`, `carla_env.py:740` |
| Invalid candidate convention | `valid=0` row in candidate block; mask zeros for invalid slots | `obs_builder.py:138-141` |
| Reset return signature | `(obs_dict, info_dict)` | `carla_env.py:418-428` |
| Step return signature | `(obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict)` | `carla_env.py:595` |
| Termination conditions | Collision (any CAV → all), goal reached (all), timeout (300 steps), off-road/stuck (>10 s) | `carla_env.py:506-569`, `base.py:139-158` |
| Truncation conditions | 300 decision steps | `carla_env.py:569` |
| Ego features (indices 0-7) | speed(0), longitudinal_accel(1), heading_error(2), lateral_offset(3), lane_index(4), lane_count(5), distance_to_goal(6), progress_fraction(7) | `obs_builder.py:110-119` |
| Route features (indices 8-13) | next_waypoint_rel_x(8), next_waypoint_rel_y(9), speed_limit(10), at_intersection(11), traffic_light_state(12), distance_to_next(13) | `obs_builder.py:124-131` |
| Candidate slots (indices 14-49) | 6 slots × 6 features: [rel_x, rel_y, rel_vx, rel_vy, rel_heading, valid] | `obs_builder.py:144-163` |
| Action mask (indices 50-55) | 6 booleans: keep, accelerate, decelerate, brake, lane_left, lane_right | `obs_builder.py:165-171` |

**56-D observation layout (verified in code):**

```
Indices  0-7: EGO block (8 features)
  0: speed / V_MAX                (clip [0,1])
  1: longitudinal_accel / A_MAX   (clip [−1,1])
  2: heading_error / pi           (clip [−1,1])
  3: lateral_offset / lane_width   (clip [−1,1])
  4: lane_index / MAX_LANES       (clip [0,1])
  5: lane_count / MAX_LANES       (clip [0,1])
  6: distance_to_goal / D_MAX     (clip [0,1])
  7: progress_fraction           (clip [0,1])

Indices  8-13: ROUTE block (6 features)
  8: next_waypoint_rel_x / ROUTE_FRAME_MAX  (clip [−1,1])
  9: next_waypoint_rel_y / ROUTE_FRAME_MAX  (clip [−1,1])
 10: speed_limit / V_MAX                   (clip [0,1])
 11: at_intersection                       (0 or 1)
 12: traffic_light_feature                 (0, 0.5, or 1)
 13: distance_to_next / ROUTE_FRAME_MAX    (clip [0,1])

Indices 14-49: CANDIDATE block (6 × 6 = 36 features, stored row-major)
  Each candidate slot j (j=0..5):
    14+6j:   rel_x / CANDIDATE_RANGE        (clip [−1,1])
    15+6j:   rel_y / CANDIDATE_RANGE        (clip [−1,1])
    16+6j:   rel_vx / V_MAX                 (clip [−1,1])
    17+6j:   rel_vy / V_MAX                 (clip [−1,1])
    18+6j:   wrapped_rel_heading / pi       (clip [−1,1])
    19+6j:   valid                          (0 or 1; 0 = padded)

Indices 50-55: ACTION MASK (6 booleans clipped to float32)
  50: keep      (always True)
  51: accelerate
  52: decelerate
  53: brake
  54: lane_left  (False → masked)
  55: lane_right (False → masked)

(After nan_to_num + clip + float32 cast, shape (56,))
```

**Action mask is PART OF the 56-D observation** (indices 50-55), confirmed by `build_observation` concatenating `mask.astype(np.float32)` at index 165-171, and by `test_obs_builder.py:186-187` asserting `observation[EGO + ROUTE + 5] == 1.0` (the emergency brake mask element).

**Invalid/padding convention:** When fewer than 6 candidates are available, the remaining candidate slots are filled with zeros and `valid=0`. The valid flag in feature index 5 of each candidate slot (i.e., indices 19, 25, 31, 37, 43, 49) is set to 0 for padded slots. This is verified by `test_candidate_padding_sets_validity_to_zero` in `test_obs_builder.py:63-69`.

**Candidate feature details (from `obs_builder.py:133-141` and `v2x_channel.py:260-293`):**

| Feature | Meaning | Normalization | Clipping | Invalid/padding value |
|---|---|---|---|---|
| rel_x | Relative world x (ego frame) | `/ CANDIDATE_RANGE (50 m)` | `clip [−1,1]` | 0.0 (padded) |
| rel_y | Relative world y (ego frame) | `/ CANDIDATE_RANGE (50 m)` | `clip [−1,1]` | 0.0 (padded) |
| rel_vx | Relative world vx (ego frame) | `/ V_MAX (30 m/s)` | `clip [−1,1]` | 0.0 (padded) |
| rel_vy | Relative world vy (ego frame) | `/ V_MAX (30 m/s)` | `clip [−1,1]` | 0.0 (padded) |
| rel_heading | Wrapped relative heading difference | `/ pi` | `clip [−1,1]` | 0.0 (padded) |
| valid | Candidate validity flag | — | — | 0 = padded/invalid; 1 = valid |

The `obs_builder` independently re-filters candidates by distance ≤ `CANDIDATE_RANGE (50 m)` and `valid=True` at `obs_builder.py:138`. This is a second filter on top of whatever the caller (V2X or raw world) provides.

---

## Existing Candidate Pipeline

The data path from CARLA actors to the 56-D observation is:

1. **CARLA actors → state extraction**: `CarlaMultiAgentEnv.get_candidates(agent_id)` at `carla_env.py:702-741`
   - Enumerates every `vehicle.*` actor in the world
   - Filters out the ego vehicle and finished agents
   - Computes Euclidean distance from ego location
   - Keeps actors with `distance <= CANDIDATE_RANGE (50 m)`
   - Sorts by `(distance, actor_id)` ascending
   - Returns list of candidate dicts with `actor_id, x, y, vx, vy, yaw`

2. **Candidate discovery**: Raw world state — every vehicle within 50 m is a candidate, regardless of whether it's a CAV, background traffic, or another simulated vehicle. No V2X involvement.

3. **Candidate conversion → observation builder**: `build_observation(ego, route, candidates, action_mask)` at `obs_builder.py:84-175`
   - Receives the candidate list from `get_candidates()`
   - Re-filters by `distance <= CANDIDATE_RANGE` and `valid=True` (defensive duplicate filter)
   - Sorts by (distance, actor_id)
   - Keeps nearest `M_CANDIDATES (6)`; zero-pads remaining slots with `valid=0`
   - Converts each candidate to ego frame, normalizes features
   - Concatenates ego + route + candidate_block.reshape(-1) + mask → 56-D vector

4. **Key question**: Candidates are **not** derived from V2X messages in the current implementation. The `V2X_INTEGRATION_DECISIONS.md` explicitly identifies this as gap D7.

**Safest insertion point for V2X:** Between steps 2 and 3 — replace the raw `get_candidates()` call with V2X-delivered messages converted via `messages_to_candidates()`, while keeping `get_candidates()` as a debug/fallback source. The observation builder does not need to change its interface; only the caller needs to provide candidate records in the expected format.

**Exact call chain:**
```
CARLA actors → get_candidates(agent_id) → candidate dicts → build_observation() → 56-D observation → policy
                                                  ↑
                                          V2X integration point:
                                          messages_to_candidates(received[agent_id], ego)
                                          replaces/get_candidates() output
```

---

## Nour's V2X Channel Contract

The V2X channel exists at `envs/v2x_channel.py` and is fully implemented and tested. Key parameters:

| Parameter / behavior | Exact implementation |
|---|---|
| Class name | `V2XChannel` |
| Constructor signature | `__init__(self, config: Optional[V2XConfig] = None, seed: Optional[int] = None)` |
| Default communication range | `range_m: float = 50.0` (validated `>= 0`, finite) |
| Range unit | Metres (meters) |
| Default delay steps | `delay_steps: int = 0` (decision steps, not ticks) |
| Default packet-loss probability | `loss_prob: float = 0.0` (Bernoulli per directed sender-receiver-step) |
| Message format | `[x, y, vx, vy, heading, intent]` — 6 float values (Message indices X,Y,VX,VY,HEADING,INTENT) |
| Required message fields | 6 values; must be finite; reshaped to `(6,)` |
| Optional message fields | None — message is exactly 6 values |
| Broadcast API | `channel.step(senders)` where `senders = [(id, [x,y,vx,vy,heading,intent]), ...]` |
| Receive API | Returns `(received, stats)` where `received = dict[receiver_id -> list[ReceivedMessage]]` |
| Transmission timing | One `step()` call per decision step; not per-agent, not per-tick |
| Packet-loss parameter | Applied per directed `(sender, receiver, step)` Bernoulli draw, before delay queue |
| Delay parameter | `delay_steps` measured in **decision steps**; message due at `step_index + delay_steps`; age = `delay_steps` when delivered |
| Queue behavior | Pending messages stored in `_pending[due_step]`; delivered when `step_index` reaches due step; discarded on `reset()` |
| Message expiration | No explicit TTL; messages persist in queue until delivered or reset discards them |
| Message-age unit | Steps (integer); `age = current_step - sent_step` |
| Candidate conversion API | `messages_to_candidates(received, ego, m=M_CANDIDATES)` returns `(m, 6)` float32 candidate block |
| Candidate output shape | `(M_CANDIDATES, CAND_FEATURES)` = `(6, 6)` |
| Candidate output feature order | `[rel_x / range, rel_y / range, rel_vx / V_MAX, rel_vy / V_MAX, wrapped_rel_heading / pi, valid]` |
| Candidate capacity | Hard cap at `m = M_CANDIDATES` (default 6); excess messages discarded |
| Candidate sorting | By (distance, sender_id) ascending; ties broken by `_id_sort_key()` (numeric id first, then string) |
| Statistics API | `episode_stats()` → `{"sent", "delivered", "dropped_by_range", "dropped_by_loss", "dropped", "mean_age", "in_flight"}` |
| RNG implementation | `np.random.default_rng(seed)` (PCG64); own RNG, never uses global NumPy RNG |
| Seed behavior | Same seed + same sender list sequence → identical received messages, ages, stats; different seeds change loss pattern when `loss_prob > 0` |

**Critical distinction:** Communication range (`range_m`) and candidate-selection radius are different concepts. The V2X channel default range is 50 m, and `obs_builder.CANDIDATE_RANGE` is also 50 m. Messages between 50 m and 100 m will saturate normalized position at `±1` if range=100 m is used. The channel applies range check **before** the conversion helper, so the helper does not re-filter by range.

**Does the channel transmit once per simulation tick or once per decision step?**
- The channel docs and implementation require **one `step()` per decision step**, not per CARLA tick. The env must call `channel.step(senders)` once at the start of each decision step.

**Does the channel require explicit `step()` or `advance()` calls?**
- Yes, explicit `step(senders)` is required. The env is responsible for calling it once per decision step.

**Does a message become available immediately or after a delay?**
- With `delay_steps=0`, messages are available immediately in the same step they are sent.
- With `delay_steps>0`, messages are queued and delivered at `step_index + delay_steps`, with `age = delay_steps`.

**Is packet loss applied per broadcast, per receiver, or per message?**
- Per directed `(sender, receiver, step)` Bernoulli draw. A sender transmits one message; each receiver independently experiences loss.

**Does the channel return only received messages or also messages still in flight?**
- `channel.step()` returns `received` (messages delivered this step) and `stats` which includes `in_flight` (cumulative accepted messages still queued). The caller can also inspect `_pending` for in-flight messages.

**Can a vehicle receive without transmitting?**
- Yes. The channel is agnostic about who broadcasts; the caller decides.

**Are duplicate messages possible?**
- The channel ensures sender ids are unique within one `step()` call (raises `ValueError` if duplicated). However, the same sender could transmit the same message content across different steps.

**Are old messages removed?**
- `reset()` clears all in-flight messages and counters. Otherwise, messages are delivered when their due step is reached and remain in `_pending` until then.

**Does candidate conversion enforce the communication radius?**
- No. `messages_to_candidates()` does not re-filter by range; it assumes the caller has already filtered. The channel's range check is separate.

**Does it independently enforce the candidate-selection radius?**
- No. The channel's range check uses `range_m` from config; the candidate-selection radius is `CANDIDATE_RANGE` in the obs_builder. These can be different values.

**Does the channel return dictionaries, arrays, tuples, or custom objects?**
- Returns `(received, stats)` where `received` is `dict[receiver_id -> list[ReceivedMessage]]` and `stats` is `dict[str, int]`.
- `ReceivedMessage` is a `NamedTuple` of `(sender_id, age, message)` where `message` is `np.ndarray` of shape `(6,)`.

**Does it preserve actor IDs?**
- Yes. `ReceivedMessage.sender_id` preserves the sender's hashable ID.

**Are intent and message age included in the candidate features or kept as metadata?**
- `intent` is **not** one of the six candidate features in `messages_to_candidates()` or `obs_builder`. It is stored inside the message but dropped during conversion. Message `age` is available per `ReceivedMessage` but is not a candidate feature.

---

## Current Selector/Grouping Implementation Status

| Component | Status | Evidence |
|---|---|---|
| `grouping/radius.py` | **NOT IMPLEMENTED** | 0 bytes |
| `grouping/random_sel.py` | **NOT IMPLEMENTED** | 0 bytes |
| `grouping/learned_sel.py` | **NOT IMPLEMENTED** | 0 bytes |
| `grouping/__init__.py` | Unknown (not checked) | — |
| `tests/test_grouping.py` | **NOT IMPLEMENTED** | 0 bytes |
| Any fixed-k selection | **NOT IMPLEMENTED** | — |
| Any deterministic tie-breaking | **NOT IMPLEMENTED** | — |
| Any padding/mask for fewer candidates | **NOT IMPLEMENTED** | — |
| Any selected-ID logging | **NOT IMPLEMENTED** | — |

---

## Recommended Common Selector API

The selector must satisfy these requirements derived from the repository contracts and `V2X_INTEGRATION_DECISIONS.md`:

**Input:** A sequence/array of candidate records, each with at minimum `(distance, actor_id, [x, y, vx, vy, yaw, valid])` or the six-feature candidate row layout. Also receives `k` (fixed number of selections) and optionally `rng` or `seed`.

**Output:** A tuple of `(selected_candidates, selected_ids, mask)` where:
- `selected_candidates`: Sequence of exactly `k` candidate records in the six-feature layout `[rel_x, rel_y, rel_vx, rel_vy, rel_heading, valid]`, nearest first, padded with zero-rows and `valid=0` if fewer than `k` eligible candidates exist.
- `selected_ids`: Tuple of exactly `k` actor IDs (or `None` for padded slots).
- `mask`: Boolean array of length `k` where `True` = valid selection, `False` = padded slot.

**Candidate feature representation:** Must match the `obs_builder` six-feature layout exactly: `[rel_x / CANDIDATE_RANGE, rel_y / CANDIDATE_RANGE, rel_vx / V_MAX, rel_vy / V_MAX, wrapped_rel_heading / pi, valid]`. The `valid` field is 1.0 for a real candidate, 0.0 for a padded slot.

**Candidate ID representation:** Actor ID as stored in the candidate record (integer from CARLA `actor.id` or string from V2X `sender_id`). IDs are used for deterministic tie-breaking.

**Fixed-k application:** Every selector returns **exactly `k` selections**, regardless of how many eligible candidates exist. If fewer than `k` candidates are eligible, the remaining slots are padded with zeros and `valid=0`.

**How fewer than k candidates are handled:** Padded with zero-feature rows and `valid=0`. The `selected_ids` for padded slots are `None` (or a sentinel). The observation mask (separate from candidate valid flags) already handles this via `make_action_mask`, but the candidate-level valid flag is what `obs_builder` uses.

**Deterministic tie-breaking:** When candidates have equal distance, break ties by `actor_id` in ascending order (numeric ids first, then string ids, per `_id_sort_key` in `v2x_channel.py:236-240`). This is the established convention in both the V2X channel conversion and the `obs_builder` candidate sorting.

**How random selection is seeded:** Use a per-selector RNG (e.g., `np.random.default_rng(seed)`) or pass an `rng` object. The selector must not mutate the V2X channel's RNG. Per-reproducibility: same seed + same candidate set → same selected IDs.

**What happens if candidate IDs are duplicated:** Not explicitly addressed in current code. The selector should either deduplicate by ID prior to selection or use `(distance, actor_id)` as the sort key to ensure deterministic ordering even with duplicates.

**How invalid candidates are handled:** Candidates with `valid=0` or `distance > CANDIDATE_RANGE` are excluded from selection eligibility. The selector may still receive them from the caller but must not include them in the selected set.

**Six-slot observation contract with k=2:** When `k=2` selected teammates, the observation still reserves six candidate slots (indices 14-49). The two selected candidates occupy slots 0 and 1 (nearest first), and slots 2-5 are padded with `valid=0`. The observation shape remains exactly `(56,)`. This is verified by the existing test `test_observation_shape_dtype_and_block_layout` which passes with zero, one, two, and six candidates.

**Concrete examples:**

| Eligible candidates | Selected (k=2) | Observation slots 0-1 | Slots 2-5 |
|---|---|---|---|
| 0 | Both padded | `valid=0` for both | All `valid=0` |
| 1 | 1 real + 1 padded | Slot 0: real candidate; Slot 1: `valid=0` | All `valid=0` |
| 2 | 2 real candidates | Both real candidates | All `valid=0` |
| 6 | 2 of the 6 nearest | The 2 nearest; slots 2-5 padded | All `valid=0` |

---

## Radius Selector Specification

**Which distance feature should be used?** Euclidean distance between ego and candidate positions in world coordinates (or equivalently, the distance already computed by `get_candidates()` or `messages_to_candidates()`). This is the distance used in both `carla_env.py:718-721` and `obs_builder.py:137-138`.

**Is distance Euclidean or longitudinal?** Euclidean (Hypot of Δx and Δy). The `obs_builder` computes `math.hypot(cx - origin_x, cy - origin_y)` at `obs_builder.py:137`. The V2X channel also uses `math.hypot` at `v2x_channel.py:165-168`. Longitudinal distance (along ego's heading) is not used for candidate discovery.

**What is the default radius?** Not yet determined. The V2X channel default is `range_m=50 m`, and `CANDIDATE_RANGE=50 m` in the obs_builder. The radius baseline should use the same 50 m default unless a different value is explicitly configured.

**Is radius different from the V2X communication range?** Yes — these are different concepts even if they happen to have the same default value:
- **Communication range** (`range_m` in V2XConfig): Determines which messages are successfully received by the channel. Messages beyond this distance are dropped and never become candidates.
- **Candidate-selection radius**: Determines which received messages are eligible for the selector. A message may be successfully received (within communication range) but still be outside the candidate-selection radius, in which case it is not eligible.

The audit identifies this as **D5 - Range vs candidate normalization**: with `range_m=100 m` robustness test and `CANDIDATE_RANGE=50 m`, messages between 50-100 m saturate normalized positions at `±1` but are still "received." The radius baseline must use a explicitly configured candidate radius, distinct from the channel's communication range.

**How should candidates be sorted?** By (distance, actor_id) ascending, matching the established convention in both `obs_builder.py:142` and `carla_env.py:740`.

**What happens when more than k candidates are within range?** The selector selects exactly `k` candidates. If more than `k` are within the radius, the `k` nearest (by distance, then actor_id) are chosen. The remaining in-range candidates are discarded for this selection.

**How should deterministic ties be resolved?** By `actor_id` ascending (numeric first, then string), per the established `_id_sort_key` convention. This applies both within the radius filter and within the selector's final ordering.

**Proposed radius selector behavior:**
- Input: candidate records (with distance and actor_id), radius value, k
- Filter: keep only candidates with `distance <= radius`
- Sort remaining by (distance, actor_id) ascending
- Select first `k`
- If fewer than `k` remain, pad with zero-rows and `valid=0`
- Return `(selected_candidates, selected_ids, mask)` where mask has `True` for the first `min(k, eligible)` slots and `False` for padded slots

---

## Random Selector Specification

**Which candidates are eligible before random selection?** All candidates present in the input set, regardless of distance. The radius baseline is a separate concern — the random selector operates on whatever candidate set it receives. In the integrated experiment, the candidate set comes from received V2X messages (after range/delay/loss filtering).

**Whether selection is without replacement?** Yes. Each candidate can be selected at most once. Since the output is exactly `k` selections and there are `M >= k` candidates, selection is without replacement.

**Which RNG should be used?** A dedicated per-selector RNG, not the V2X channel's RNG. Proposed: `np.random.default_rng(seed)` where the seed is either:
- A fixed per-episode seed (for reproducibility), or
- Derived from the environment seed

**How per-episode/per-agent reproducibility can be achieved:** Fix the random seed per episode (e.g., `seed + agent_index` or a hash of the episode seed). Same seed + same candidate set → same selected IDs.

**Whether random selection should preserve candidate ordering after selection?** The output should preserve the original candidate ordering for the selected slots. That is, if candidates sorted by (distance, actor_id) are `[c0, c1, c2, c3, c4, c5]` and the random selector chooses indices `[2, 0]`, the output should be `[c2, c0]` (selected in the order they appear in the input, not sorted by index).

Actually, re-reading the design: the selector output `selected_candidates` should be in the same order as they appear in the input candidate list. This preserves the (distance, actor_id) ordering for the selected subset.

**How to avoid changing the channel's own random-number stream?** The selector must create its own `np.random.default_rng(seed)` instance. It must not use `np.random.seed()`, `random.seed()`, or the V2X channel's `_rng`.

**Proposed random selector behavior:**
- Input: candidate records (list), k, seed
- Create `rng = np.random.default_rng(seed)`
- Generate a permutation of `0..len(candidates)-1` using `rng.permutation`
- Select the first `k` indices (without replacement)
- Return selected candidates in their original order (i.e., `selected[i] = candidates[selected_idx[i]]`)
- If fewer than `k` candidates, pad as per the common API

---

## Learned Selector Interface Requirements

**Do not implement.** Identify the common interface the future learned selector must satisfy.

**Model outputs/scores needed:** The learned selector should produce a score or attention weight for each candidate, from which the top-k are chosen. The required inputs are:

- Candidate feature vectors (six-feature layout: `[rel_x, rel_y, rel_vx, rel_vy, rel_heading, valid]`)
- Candidate IDs (actor IDs)
- Optional: Ego features, route features, action mask
- Output: Scores/attention weights for each candidate, of shape `(n_candidates,)`

**Required interface methods:**
- `selector.select(candidates, k, rng)` → `(selected_candidates, selected_ids, mask)`
- Or: `selector.forward(candidates)` → `(scores,)` and a separate `top_k(scores, k)` function

**The learned selector must produce the same output format as the radius and random selectors** — the common API defined above. This ensures that the observation builder receives a consistent input format regardless of which selector is active.

**Candidate feature ordering for the learned selector:** Must match the `obs_builder` six-feature layout exactly. The selector receives candidate records in the format produced by `messages_to_candidates()` or `get_candidates()`, i.e., each record has `(distance, actor_id, {x, y, vx, vy, yaw, valid})` or the six normalized features.

**Top-k selection:** Choose the `k` candidates with highest scores. Ties broken by actor_id (ascending), consistent with the radius and random selectors.

**Must not include padded rows in the selection:** The selector selects exactly `k` from the eligible candidates. If there are fewer than `k` eligible candidates, the observation builder pads the remaining slots.

**No change to the 56-D observation contract:** Even with the learned selector, the observation remains exactly 56-D. Selected candidates occupy the first few of the 6 candidate slots, and the remaining slots are padded with `valid=0`.

---

## Exact V2X-to-CARLA Integration Points

Based on the real code, here is the proposed data path:

```
CARLA state extraction
→ V2X message transmission
→ V2X message reception
→ eligibility filtering
→ selector
→ selected candidate records
→ existing observation builder
→ 56-D policy observation.
```

**For every arrow:**

| Arrow | Existing function | Required input | Required output | New adapter needed? | Existing code to reuse |
|---|---|---|---|---|---|
| 1. CARLA state extraction | `get_candidates(agent_id)` at `carla_env.py:702-741` | Agent ID | List of candidate dicts `{actor_id, x, y, vx, vy, yaw}` | Reuse with modification (see below) | Candidate discovery logic, distance filtering at 50 m |
| 2. V2X message transmission | `V2XChannel.step(senders)` at `v2x_channel.py:129-199` | List of `(id, [x,y,vx,vy,heading,intent])` | `(received, stats)` dicts | **Yes** — env must call this once per decision step | Range check, Bernoulli loss, delay queue |
| 3. V2X message reception | Channel `step()` return `received[agent_id]` | Agent ID | List of `ReceivedMessage(sender_id, age, message)` | **Yes** — env must extract per-agent received messages | Channel API is already designed for this |
| 4. Eligibility filtering | `messages_to_candidates(received, ego, m=6)` at `v2x_channel.py:247-293` | Received messages, ego state | `(6, 6)` float32 candidate block | **No** — reuse as-is | Range normalization, ego-frame conversion, zero-padding to m slots |
| 5. Selector | To be implemented in `grouping/` | Candidate block, k=2, seed | `(selected_candidates, selected_ids, mask)` | **Yes** — new grouping module | Common API: fixed-k, deterministic tie-break, padding |
| 6. Observation builder | `build_observation(ego, route, candidates, action_mask)` at `obs_builder.py:84-175` | Selected candidates (6-feature layout), action mask | 56-D float32 observation | **No** — reuse as-is | Accepts any sequence of candidate records in the expected format |
| 7. Policy input | Policy forward pass | 56-D observation, action mask | Action selection | N/A | Existing policy architecture |

**Detailed integration answers:**

1. **Where should the V2X channel be instantiated?**
   - One `V2XChannel` per environment instance, created in `CarlaMultiAgentEnv.__init__()` (or lazily on first `reset()`).
   - Owned by the environment; reset with the episode seed in `reset()`.

2. **Who owns its lifecycle?**
   - The `CarlaMultiAgentEnv` instance. It creates the channel, calls `step()` once per decision, and calls `reset()` at episode reset.

3. **At which exact point should messages be broadcast?**
   - At the start of each decision step, **before** `channel.step(senders)` is called. The env builds sender messages from all broadcasting vehicles (CAVs only, or CAVs + background per D1 decision).
   - Sender format: `(agent_id, [x, y, vx, vy, heading, intent])` where position/velocity/yaw come from the vehicle's current CARLA state, and `intent` is the agent's current meta-action (or 0 = KEEP).

4. **At which exact point should the channel advance its delay queue?**
   - Once per decision step, at the start of the decision step, **before** observations are built. The env calls `channel.step(senders)` once, caches the `received` mapping, and reuses it for all agents' candidate conversion.

5. **How do we guarantee one V2X decision update per 5 Hz decision rather than per 20 Hz CARLA tick?**
   - The env's `step()` method already has a loop `for _ in range(TICKS_PER_STEP): self.world.tick()` at `carla_env.py:458-459` that advances the simulation by 4 ticks (5 Hz decision rate). The V2X channel `step()` must be called **once** after this loop completes, at the same point where the observation is built. The env should not call `channel.step()` inside the per-tick loop.

6. **How do we map message sender IDs to CARLA actor IDs?**
   - Sender IDs in the V2X channel are hashable values (strings like `"cav_0"` or integers like actor IDs). The env must ensure that sender IDs used in the V2X channel match the agent IDs used in the environment (`cav_0` through `cav_3`). For background vehicles, sender IDs should be distinct from CAV agent IDs. The V2X channel is agnostic; the caller (env) controls the ID space.

7. **How do we prevent stale or duplicate messages becoming candidates?**
   - The V2X channel's range check, loss model, and delay queue already handle this. Messages beyond `range_m` are dropped; loss may drop them; delayed messages carry stale values from the step they were sent. The `messages_to_candidates()` conversion then sorts by (distance, actor_id) and keeps the nearest 6. Stale messages from previous steps are in the delay queue and will have `age = delay_steps`; they are valid messages and may be selected — this is the intended behavior of the robustness test.
   - **Additional precaution**: The env should not mix candidates from `get_candidates()` (raw world) and `messages_to_candidates()` (V2X) in the same observation. Use one source exclusively.

8. **How do we ensure only actually received messages are eligible?**
   - The `received[agent_id]` dict from `channel.step()` contains only messages delivered to that agent in the current step (or from the delay queue). Messages never received (out of range, lost) are not present. This is the channel's guarantee.

9. **How do we apply the candidate-selection radius separately from communication range?**
   - The V2X channel uses `range_m` (default 50 m) to decide which messages are received. The candidate selector uses a (possibly different) radius to decide which received messages are eligible. These are independent:
     - A message at 70 m may be received if `range_m=100 m`, but excluded from candidates if the candidate radius is 50 m.
     - A message at 30 m is always received if `range_m=50 m` and included as a candidate.
   - The selector radius should be configured explicitly. For the main experiment with 50 m communication range, the natural radius is also 50 m, but the code must allow them to differ.

10. **How do we feed the selected records into the existing observation builder?**
    - The selector returns `selected_candidates` as a sequence of six-feature candidate rows (the same format that `obs_builder` expects). These are passed directly to `build_observation(ego, route, candidates=selected_candidates, action_mask=mask)`. The observation builder does not need to know whether candidates came from V2X or raw world; it only needs the six-feature layout.

11. **How do we preserve the exact 56-D observation shape?**
    - The observation builder always produces `(56,)` output. When `k=2` selected candidates are passed, the builder places them in the first two of six candidate slots (indices 14-15 within the candidate block), and the remaining four slots are padded with `valid=0`. The candidate block is always `6 × 6 = 36` features, concatenated with ego(8) + route(6) + mask(6) = 56. This is verified by `test_observation_shape_dtype_and_block_layout`.

12. **How do we log received and selected IDs without modifying the observation format?**
    - Add V2X stats and selected IDs to the `info` dict returned by `step()`, not to the observation vector. The `info` dict already contains per-episode keys like `success`, `min_ttc`, `jerk`, `town`, `seed`. Recommended additions:
      - `info["v2x"]` = `{"sent", "delivered", "dropped_by_range", "dropped_by_loss", "mean_age", "in_flight"}`
      - `info["selection"]` = `{"agent_id": {"received_ids": [...], "selected_ids": [...], "candidate_count": N, "selected_count": k, "selector_type": "radius|random|learned"}}`
    - Per-episode CSV columns can include these fields without changing the observation contract.

---

## Configuration and Logging Requirements

### Parameters needed for Milestones 1–3

| Parameter | Existing name | Default value | Source file | Unit | Configurable | Currently used | New parameter required? |
|---|---|---|---|---|---|---|---|
| Selector type | — | — | — | — | Yes (new) | No | Yes — e.g., `selector_type="radius"` |
| Fixed k | — | `k=2` (from spec) | `env_spec.md:118` | — | Yes | No (hard-coded in env logic) | Yes — passed to selector |
| Selector radius | — | Not yet configured | — | Metres | Yes (new) | No | Yes — e.g., `selector_radius=50.0` |
| Candidate radius | `CANDIDATE_RANGE` | `50.0` | `obs_builder.py:25` | Metres | Yes (env constant) | Yes | No — already exists |
| V2X communication range | `range_m` in `V2XConfig` | `50.0` | `v2x_channel.py:37` | Metres | Yes | Yes (channel config) | No — already exists |
| Packet-loss probability | `loss_prob` in `V2XConfig` | `0.0` | `v2x_channel.py:39` | — | Yes | Yes (channel config) | No — already exists |
| Delay steps | `delay_steps` in `V2XConfig` | `0` | `v2x_channel.py:38` | Decision steps | Yes | Yes (channel config) | No — already exists |
| Message lifetime/TTL | — | No explicit TTL | — | Steps | No | No | MISSING — may need TTL or rely on reset |
| Random seed | `seed` in `CarlaMultiAgentEnv` | `0` | `carla_env.py:82` | — | Yes | Yes (env seed) | Yes — for selector reproducibility |
| Scenario seed | `seed` in `CarlaMultiAgentEnv` | `0` | `carla_env.py:82` | — | Yes | Yes (env seed) | Yes — for reproducible spawns |
| Maximum candidate slots | `M_CANDIDATES` | `6` | `base.py:15`, `obs_builder.py:12` | — | Yes (constant) | Yes | No — already exists |
| Decision frequency | `TICKS_PER_STEP` | `4` | `carla_env.py:53` | — | Yes (env config) | Yes | No — already exists |
| Logging: episode ID | — | Not logged | — | — | No | No | Yes — recommended |
| Logging: scenario ID | `scenario_id` in `info` | `"free_drive"` | `carla_env.py:580` | — | Yes (new) | No (hard-coded) | Yes — recommended |
| Logging: seed | `seed` in `info` | `0` | `carla_env.py:581` | — | Yes | Yes | No — already exists |
| Logging: decision step | — | Not logged | — | — | No | No | Yes — recommended |
| Logging: received message IDs | — | Not logged | — | — | No | No | Yes — recommended |
| Logging: received sender IDs | — | Not logged | — | — | No | No | Yes — recommended |
| Logging: selected teammate IDs | — | Not logged | — | — | No | No | Yes — recommended |
| Logging: candidate count | — | Not logged | — | — | No | No | Yes — recommended |
| Logging: selected count | — | Not logged | — | — | No | No | Yes — recommended |
| Logging: message age | `mean_age` in `episode_stats()` | `0.0` default | `v2x_channel.py:201-211` | Decision steps | Yes (channel) | Yes (channel stats) | No — already exists in channel |
| Logging: packets sent | `sent` in `episode_stats()` | `0` default | `v2x_channel.py:201-211` | — | Yes (channel) | Yes (channel stats) | No — already exists |
| Logging: packets received | `delivered` in `episode_stats()` | `0` default | `v2x_channel.py:201-211` | — | Yes (channel) | Yes (channel stats) | No — already exists |
| Logging: packets dropped | `dropped` in `episode_stats()` | `0` default | `v2x_channel.py:201-211` | — | Yes (channel) | Yes (channel stats) | No — already exists |
| Logging: messages in flight | `in_flight` in `episode_stats()` | `0` default | `v2x_channel.py:201-211` | — | Yes (channel) | Yes (channel stats) | No — already exists |
| Logging: selector type | — | Not logged | — | — | No | No | Yes — recommended |
| Logging: selector radius | — | Not logged | — | Metres | No | No | Yes — recommended |
| Logging: fixed k | — | Not logged | — | — | No | No | Yes — recommended |

**No duplicate configuration keys:** The following values already have established homes and must not be duplicated:
- `CANDIDATE_RANGE` → `envs/obs_builder.py:25`
- `M_CANDIDATES` → `envs/base.py:15` / `envs/obs_builder.py:12`
- `V2XConfig.range_m` → `envs/v2x_channel.py:37`
- `V2XConfig.delay_steps` → `envs/v2x_channel.py:38`
- `V2XConfig.loss_prob` → `envs/v2x_channel.py:39`
- `OBS_DIM` → `envs/obs_builder.py:16`

### Logging fields table (proposed additions to `info` dict and per-episode CSV)

| Field | Suggested key | Type | Source |
|---|---|---|---|
| Episode ID | `info["episode_id"]` | `int` | Environment counter |
| Scenario ID | `info["scenario_id"]` | `str` | Environment (currently `"free_drive"`) |
| Seed | `info["seed"]` | `int` | Already in `info` |
| Agent ID | `info["per_agent"][agent_id]` | dict | Already in `info` |
| Decision step | `info["decision_step"]` | `int` | `env.step_count` |
| Received message IDs | `info["v2x"]["received_ids"]` | `list[str]` | Channel `received` per agent |
| Received sender IDs | `info["v2x"]["received_sender_ids"]` | `list` | Channel sender_ids from ReceivedMessage |
| Selected teammate IDs | `info["v2x"]["selected_ids"]` | `list[int]` | Selector output |
| Candidate count | `info["v2x"]["candidate_count"]` | `int` | `len(received[agent_id])` |
| Selected count | `info["v2x"]["selected_count"]` | `int` | `k` (fixed=2) |
| Message age | `info["v2x"]["mean_age"]` | `float` | `channel.episode_stats()["mean_age"]` |
| Packets sent | `info["v2x"]["sent"]` | `int` | `channel.episode_stats()["sent"]` |
| Packets received | `info["v2x"]["delivered"]` | `int` | `channel.episode_stats()["delivered"]` |
| Packets dropped | `info["v2x"]["dropped"]` | `int` | `channel.episode_stats()["dropped"]` |
| Messages in flight | `info["v2x"]["in_flight"]` | `int` | `channel.episode_stats()["in_flight"]` |
| Selector type | `info["v2x"]["selector_type"]` | `str` | `"radius"|"random"|"learned"` |
| Selector radius | `info["v2x"]["selector_radius"]` | `float` | Config value |
| Fixed k | `info["v2x"]["fixed_k"]` | `int` | `2` |

**Existing CSV/logging system check:** The current metrics system at `envs/metrics.py` tracks TTC and jerk. The training scripts at `scripts/train_qmix_highway.py` and `scripts/train_mappo_highway.py` write small CSVs. The `V2X_INTEGRATION_DECISIONS.md` Sec 10 recommends adding these fields to `info["v2x"]` and the per-episode CSV. The existing `info` contract (Contract 12.3) already has keys `success`, `min_ttc`, `jerk`, `scenario_id`, `town`, `seed` on the final step. The new V2X and selection fields can be added as nested dicts without breaking the contract, since the contract only requires the listed keys and does not prohibit additional keys.

---

## Milestone 1 Test Matrix

### Test names, fixtures, expected outputs

| # | Test name | Input fixture | Expected selected IDs | Expected output shape | Expected mask | Expected padding values | Expected deterministic behavior |
|---|---|---|---|---|---|---|---|
| M1.1 | Empty candidate set | `candidates=[]`, `k=2` | Both padded (no real IDs) | `(56,)` observation | Valid mask: `[False, False, False, False, False, False]` for candidate block; action mask per lane context | All candidate slots `valid=0` | N/A (no candidates to tie-break) |
| M1.2 | One candidate | `candidates=[candidate_1]`, `k=2` | `[candidate_1_id, None]` (second padded) | `(56,)` observation | Candidate valid mask: `[True, False, False, False, False, False]` (first slot valid, second padded) | Slot 0: real features `valid=1`; Slot 1: all-zero `valid=0` | Deterministic: same input → same output |
| M1.3 | Two candidates | `candidates=[c1, c2]` (distinct distances), `k=2` | Both selected, ordered by distance | `(56,)` observation | Candidate valid mask: `[True, True, False, False, False, False]` | Both slots `valid=1` with real features | Deterministic: same input → same selection |
| M1.4 | Exactly six candidates | `candidates=[c0,...,c5]`, `k=2` | Two nearest (by distance/ID), slots 2-5 padded | `(56,)` observation | Candidate valid mask: `[True, True, False, False, False, False]` | Slots 0-1 valid, slots 2-5 `valid=0` | Observation shape `(56,)` verified |
| M1.5 | More than six candidates | `candidates=[c0,...,c7]`, `k=2` | Two nearest of the 7, slots 2-5 padded | `(56,)` observation | Same as M1.4 | Same as M1.4 | Selects the 2 nearest regardless of total count |
| M1.6 | Duplicate IDs | `candidates=[c_with_id_8, c_with_id_8, c9]`, `k=2` | Two distinct IDs if possible, or deterministic pick | `(56,)` observation | Valid mask depends on selection | Padded slots have `valid=0` | Duplicate ID handling: either deduplicate or use (dist, ID) sort |
| M1.7 | Equal-distance candidates | `candidates` all at same distance but different IDs, `k=2` | Two smallest actor IDs (ascending) | `(56,)` observation | Valid mask: first two slots `True` | Remaining slots `valid=0` | Tie-break by actor_id ascending (established convention) |
| M1.8 | Radius boundary | Candidates at distances `[49.9, 50.1, 50.0]` m, radius=50 m, `k=2` | Only the 49.9 m and 50.0 m candidates selected (50.1 excluded) | `(56,)` observation | Valid mask: two True slots | 50.1 m candidate excluded; slots 2-5 padded | Radius filter: `distance <= radius` (inclusive) |
| M1.9 | Random selection reproducibility | Same candidate set, same seed, `k=2` | Same selected IDs across runs | `(56,)` observation | Same mask | Same padding | Same seed → same output (per Common API spec) |
| M1.10 | Random selection without replacement | `k=2` from `N=6` candidates, never selects same candidate twice | Two distinct candidates selected | `(56,)` observation | Two `True` slots, four `False` | No duplicate selections | Without replacement: permutation of indices, first k |
| M1.11 | Fixed-k behavior | `k=2` always, even with 0, 1, 3, or 6 candidates | Always exactly 2 selected (or padded) | `(56,)` observation | Mask always has exactly `k` `True` values (or `False` for padded) | Padded slots `valid=0` | Fixed-k: output size = k, regardless of input count |
| M1.12 | k greater than eligible count | `k=5` but only 2 candidates within radius | Both candidates selected, 3 slots padded | `(56,)` observation | Mask: first 2 `True`, last 3 `False` | 3 padded slots `valid=0` | Fixed-k: output size = k, excess eligibility ignored |
| M1.13 | Padding | Fewer than k candidates, e.g., 1 candidate with `k=2` | 1 real + 1 padded | `(56,)` observation | Valid mask: `[True, False, ...]` | Padded slot `valid=0` | Zero padding with `valid=0` |
| M1.14 | Validity mask | Mixed valid/invalid candidates, `k=2` | Selected valid candidates only | `(56,)` observation | Valid mask reflects which of the 6 candidate slots have `valid=1` | Invalid candidates excluded before selection | `valid=0` rows are excluded from selection; only `valid=1` candidates compete |
| M1.15 | Deterministic tie-breaking | Equal-distance candidates with IDs 8, 2, 15, `k=2` | IDs 2 and 8 selected (ascending) | `(56,)` observation | Valid mask: first two True | Remaining False | Tie-break by (distance, actor_id) ascending |

**Milestone 1 test execution:** These tests must be implemented in `tests/test_grouping.py` and should all pass before proceeding to Milestone 2. The tests verify the common selector API: fixed-k, deterministic tie-breaking, padding, validity mask, and handling of edge cases (empty, one, two, six+, duplicate IDs, equal distance).

---

## Milestone 2 Test Matrix

### Test names, fixtures, expected outputs

| # | Test name | Condition | Expected selected IDs | Expected output shape | Expected mask | Expected padding | Other |
|---|---|---|---|---|---|---|---|
| M2.1 | Perfect communication | `range_m=50`, `loss_prob=0`, `delay_steps=0`, all messages received | All sent messages received | `(56,)` per agent | Correct action mask | No NaNs/inf | V2X stats: delivered = sent |
| M2.2 | Message outside communication range | Message at 70 m, `range_m=50` | Not received; not in candidates | `(56,)` observation | Cand valid=0 for missing | Natural exclusion | `dropped_by_range` counted in stats |
| M2.3 | Message inside comm range but outside candidate radius | Message at 70 m, `range_m=100`, candidate radius=50 | Received but not eligible | `(56,)` observation | Candidate valid=0 (clipped at ±1) | Saturation at ±1 for normalized features | D5 issue: range vs candidate radius |
| M2.4 | Packet loss | `loss_prob=0.3`, 1000 directed transmissions | ~30% dropped | `(56,)` per agent | Some candidates missing | Some slots padded | Empirical rate ≈ 0.3 within 0.02 |
| M2.5 | Delayed message | `delay_steps=2`, message sent at step t, delivered at t+2 | Stale values (from step t-2) | `(56,)` observation | Age = 2 in ReceivedMessage | Values from old step | Verified by `test_delay_arrival_step_age_and_stale_content` |
| M2.6 | Missing message | Expected sender absent from `senders` list | No entry in `received[agent_id]` | `(56,)` observation | Candidate block has `valid=0` slots | Natural gap in candidate set | |
| M2.7 | Duplicate message | Same sender transmits identical message twice | Channel raises `ValueError` for duplicate IDs within one `step()` | Error or handled | — | — | Sender IDs must be unique per step |
| M2.8 | Sender-ID mapping | V2X sender ID `"cav_0"` maps to CARLA actor ID `123` | Consistent mapping across steps | `(56,)` observation | Consistent candidate IDs | Env must ensure ID consistency | Sender IDs in V2X must match env agent IDs |
| M2.9 | Candidate ordering | Candidates at different distances, sorted by (dist, ID) | Nearest first, then by ID | `(56,)` observation | Cand slots 0-1 nearest | Sorted by (dist, ID) | Verified by `test_candidates_are_sorted_by_distance_then_actor_id` |
| M2.10 | Candidate capacity | 10 candidates, `M_CANDIDATES=6` | Only 6 kept (nearest), 4 discarded | `(56,)` observation | Slots 2-5 `valid=0` | Excess discarded, not padded at builder level | `messages_to_candidates` keeps nearest m |
| M2.11 | Observation shape exactly `(56,)` | Any valid input | `observation.shape == (56,)` | `(56,)` | — | — | Verified by `test_observation_shape_dtype_and_block_layout` |
| M2.12 | No NaNs or infinities | Any input with finite values | `np.isfinite(observation).all()` | `(56,)` | — | — | `nan_to_num` + clip handles this |
| M2.13 | Correct action mask | Lane-left/right present/absent | Mask with correct booleans | `(6,)` per agent | `make_action_mask` output | Invalid lane changes fall back to KEEP | Verified by `test_lane_masks_keep_emergency_brake_valid` |
| M2.14 | Metadata not accidentally inserted into observation | V2X intent, message age not in candidate features | Observation has exactly 56 features, none are intent/age | `(56,)` | — | — | Intent is dropped in `messages_to_candidates`; age is metadata only |

**Milestone 2 test execution:** These tests integrate the V2X channel into the CARLA observation path. They should be run with a headless CARLA server or with the toy environment adapted to use V2X candidates. The tests verify that range/delay/loss propagate from the V2X channel into the observation, and that the 56-D contract is preserved.

---

## Milestone 3 Synthetic Experiment Design

### Reproducible synthetic fixture

**Setup:** Several vehicle IDs with known positions relative to ego.

| Vehicle ID | World position (x, y) m | Velocity (vx, vy) m/s | Heading rad | Intent (action) |
|---|---|---|---|---|
| `cav_0` (ego) | `(0.0, 0.0)` | `(10.0, 0.0)` | `0.0` | `0` (KEEP) |
| `cav_1` | `(5.0, 0.0)` | `(8.0, 0.0)` | `0.0` | `0` |
| `cav_2` | `(15.0, 0.0)` | `(6.0, 0.0)` | `0.0` | `0` |
| `cav_3` | `(30.0, 0.0)` | `(5.0, 0.0)` | `0.0` | `0` |
| `bkg_1` | `(-5.0, 3.0)` | `(0.0, 0.0)` | `0.0` | `0` |
| `bkg_2` | `(-3.0, -4.0)` | `(0.0, 0.0)` | `0.0` | `0` |

**Comparisons (4 conditions):**

| Condition | Communication range | Candidate radius | Packet loss | Delay | Expected result |
|---|---|---|---|---|---|
| 1. Perfect communication | 50 m | 50 m | 0% | 0 steps | All 5 vehicles (cav_1, cav_2, cav_3, bkg_1, bkg_2) within 50 m are received and become candidates. Radius selector selects k=2 nearest. |
| 2. 30% packet loss | 50 m | 50 m | 30% | 0 steps | Approximately 70% of messages received. Random selector with fixed seed produces reproducible subset. |
| 3. Two-decision-step delay | 50 m | 50 m | 0% | 2 steps | Messages carry stale values from 2 steps ago. Ages = 2. Selected candidates have stale positions. |
| 4. 30 m comm range + 50 m candidate radius | 30 m | 50 m | 0% | 0 steps | Only vehicles within 30 m are received. Of those, all are within 50 m candidate radius. Fewer candidates than with 100 m range. |

**Concrete example: radius vs random selectors from same received candidate set**

*Received candidates (after V2X channel, 50 m range, no loss, no delay):*
| Candidate | Distance m | Actor ID | Features |
|---|---|---|---|
| c_A | 5.0 | 8 | [rel_x=0.1, rel_y=0, rel_vx=..., ...] |
| c_B | 5.0 | 12 | [rel_x=0.1, rel_y=0, rel_vx=..., ...] |
| c_C | 10.0 | 5 | [rel_x=0.2, rel_y=0, ...] |
| c_D | 15.0 | 3 | [rel_x=0.3, rel_y=0, ...] |

- **Radius selector** (radius=7 m): Only c_A and c_B are within radius (distance ≤ 7). Selected IDs: `{8, 12}`. Slots 2-5 padded `valid=0`.
- **Random selector** (seed=42, k=2): All four candidates are eligible (within communication range). Random selection picks 2 of the 4. Possible selected IDs: e.g., `{5, 12}` or `{8, 3}`, depending on the RNG.
- **Demonstration:** From the same received set, radius selects `{8, 12}` (both at 5.0 m, within 7 m radius), while random selects a subset of all 4 candidates. The two selectors produce different selected candidate sets from the identical input.

*Another example with 6 eligible candidates:*
- Radius (radius=12 m): Selects candidates at 5.0 m and 10.0 m → IDs `{8, 5}`.
- Random (k=2 from 6): Selects any 2 of the 6. With a fixed seed, might select `{3, 12}`.
- **Key point:** Both selectors produce exactly 2 selected IDs, but the sets differ. The radius selector is distance-biased; the random selector is uniformly biased. Both preserve the 56-D observation shape.

### Milestone 3 synthetic demonstration goals

- Validate that different selectors produce different selected candidate sets from the same received message set.
- Verify observation shape `(56,)`, masks, padding, and deterministic behavior.
- Validate that the V2X channel's range/delay/loss parameters affect which candidates are available.
- **Do not claim** this demonstrates driving performance or MARL improvement. It only validates data flow and selector behavior.

---

## Risks, Contradictions, and Missing Information

| Issue | Evidence | Impact | Recommended resolution | Must be decided by |
|---|---|---|---|---|
| `main` vs Nour's V2X branch compatibility | V2X channel implemented directly on `main`; no `feat/v2x-channel` branch exists | No branch merge needed; V2X code is already on `main` | Proceed with V2X channel as-is on `main` | N/A (already on main) |
| Existing candidate radius vs V2X communication range | `CANDIDATE_RANGE=50 m` in obs_builder; `range_m=50 m` default in V2XConfig | Same default value but different semantic meanings | Configure them independently; for main experiment, use 50 m for both | Supervisor |
| 56-D observation contract vs k=2 selection | Observation has 6 candidate slots; k=2 selects 2, pads 4 with `valid=0` | Shape preserved at `(56,)`; verified by existing tests | Keep 6-slot layout with padding; selector returns exactly k=2 | Supervisor (contract decision D9) |
| Candidate feature ordering vs V2X candidate conversion | Both use `[rel_x/range, rel_y/range, rel_vx/V_MAX, rel_vy/V_MAX, wrapped_rel_heading/pi, valid]` | Same layout; `messages_to_candidates()` matches `obs_builder` | Reuse `messages_to_candidates()` without modification | N/A (already compatible) |
| CARLA actor IDs vs V2X sender IDs | CARLA actor IDs are integers (e.g., `123`); V2X sender IDs can be strings (e.g., `"cav_0"`) | Different ID spaces must be reconciled | Env ensures consistency: use same ID space (e.g., `cav_0` through `cav_3` for CAVs) | Supervisor (D1 decision) |
| Simulation ticks vs decision steps | CARLA: 20 Hz sim, 5 Hz decision (4 ticks/step). V2X: delay in decision steps. | Must call channel.step() once per decision, not per tick | Env's `step()` loop of 4 ticks → then channel.step() → then observation | Engineering (D6 decision) |
| Delay queue advancement frequency | Channel stepped once per decision step; env's tick loop is 4 ticks/step | Must not step channel inside per-tick loop | Call channel.step() once after the tick loop, before observation | Engineering (D6 decision) |
| Channel RNG vs selector RNG | V2X channel owns `np.random.default_rng(seed)`; selectors must use their own RNG | Must not share/mix RNG streams | Selectors create own `rng = np.random.default_rng(seed)`; never mutate channel's `_rng` | Engineering |
| Candidate ordering and deterministic ties | Convention: sort by (distance, actor_id) ascending, in both `obs_builder` and `v2x_channel` | Establishes predictable behavior across selectors | Enforce (distance, actor_id) sort key throughout the pipeline | Supervisor (established convention) |
| Observation builder can accept externally filtered candidates | `build_observation()` accepts any sequence of candidate records in the 6-feature layout | Must ensure caller provides candidates in the right format (post-`messages_to_candidates`) | Document that caller must convert via `messages_to_candidates()` before passing to `build_observation()` | Engineering |
| Current tests cover actual integration path | V2X channel tests pass in isolation; no test proves env consumes received messages | No end-to-end test of V2X→candidates→observation pipeline | Add integration tests before claiming validity | Supervisor |
| Whether a live CARLA server has ever validated the current code | Historical logs (`results/api_carla.txt`) use 2 agents, older commits; no 4-agent HEAD run | Cannot claim CARLA validation without a live server setup | Document as "pending CARLA validation"; set up compatible env (Python 3.10/3.11 + CARLA 0.9.16) | Engineering |

---

## Final Deliverable — Implementation-Ready Technical Specification

### 1. Executive Summary
See above.

### 2. Exact Audited Git Baseline
See above.

### 3. Repository Architecture
See above.

### 4. Verified CARLA Environment Contract
See above.

### 5. Exact 56-D Observation Layout
See above.

### 6. Existing Candidate Pipeline
See above.

### 7. Nour's V2X Channel Contract
See above.

### 8. Current Selector/Grouping Implementation Status
See above.

### 9. Recommended Common Selector API
See above.

### 10. Radius Selector Specification
See above.

### 11. Random Selector Specification
See above.

### 12. Learned Selector Interface Requirements
See above.

### 13. Exact V2X-to-CARLA Integration Points
See above.

### 14. Configuration Parameter Table
See above.

### 15. Logging Parameter Table
See above.

### 16. Milestone 1 Test Matrix
See above.

### 17. Milestone 2 Test Matrix
See above.

### 18. Milestone 3 Synthetic Experiment Design
See above.

### 19. Risks and Unresolved Supervisor Decisions
See table in section 11.

### 20. Recommended Implementation Order
1. **Freeze the interface:** Resolve the 10 D-decisions and 7 A-decisions recorded in `V2X_INTEGRATION_DECISIONS.md` and the repository audit. Record approved answers in `docs/env_spec.md`.
2. **Implement grouping module:** Implement `grouping/radius.py`, `grouping/random_sel.py`, `grouping/learned_sel.py` with the common selector API, fixed `k=2`, deterministic tie-breaking, zero padding, and validity masks. Add `tests/test_grouping.py` with the Milestone 1 test matrix.
3. **Integrate V2X into CARLA candidates:** 
   - Add `V2XChannel` to `CarlaMultiAgentEnv.__init__()`.
   - In `reset()`, create senders list from CAVs (config: `cavs_only` or `cavs_and_background`).
   - In `step()`, after the 4-tick loop, call `channel.step(senders)` once, cache `received`.
   - For each agent, convert `received[agent_id]` via `messages_to_candidates(ego_state)` to get candidate block.
   - Feed candidate block through the selector to get `selected_candidates`, `selected_ids`, `mask`.
   - Pass `selected_candidates` to `build_observation()` instead of `get_candidates()` output.
4. **Implement scenarios and background traffic:** Add scripted vehicles per scenario, set real `scenario_id`, seed spawn/routes/goals.
5. **Fix contract defects:** Correct per-agent/team termination; align comfort reward with frozen spec; document deviations.
6. **Build a runnable environment:** Pin Python/CARLA, install dependencies, run 4-agent smoke test, run full test suite.
7. **Build the matched experiment:** Same policy, reward, seeds, traffic, budget; only selector changes; log selected IDs and communication counters.
8. **Evaluate on held-out seeds/towns:** Report mean and 95% CI; compare against random and keep-speed baselines; do not tune on test.

### 21. Files that may need to be created or modified in a future implementation (do not create them)

| File path | Why it is needed | Existing function/class to reuse | Exact proposed responsibility | Risks changing public env contract? |
|---|---|---|---|---|
| `envs/carla_env.py` | Integrate V2X channel; call `channel.step()` per decision; use `messages_to_candidates()` output instead of `get_candidates()` | `CarlaMultiAgentEnv`, `get_candidates()`, `_extract_observation_inputs()`, `build_observation()` via `obs_builder` | Add V2X channel instance, sender construction, per-decision stepping, candidate conversion, selector feeding, info logging. Modify `step()` and `reset()` flow. **HIGH RISK** — changes the candidate source and may alter observation output if not done carefully. Must keep 56-D contract intact. |
| `grouping/radius.py` | Radius baseline selector implementation | Common selector API (proposed); `obs_builder` padding convention | Select up to k candidates within radius, deterministic tie-break by (dist, actor_id), pad to k, return (selected_candidates, selected_ids, mask). | LOW — only affects grouping output, not observation shape if common API observed. |
| `grouping/random_sel.py` | Random baseline selector implementation | Common selector API; seeded RNG | Seeded random sample without replacement, fixed k, padding when short, return (selected_candidates, selected_ids, mask). | LOW — same as above. |
| `grouping/learned_sel.py` | Learned selector interface | Common selector API; candidate feature layout | Attention scores over candidate features, top-k selection, no selection of padded rows, return (selected_candidates, selected_ids, mask). | LOW — same, if common API observed. |
| `grouping/__init__.py` | Package init for grouping module | — | Export selector functions/classes. | LOW — purely package infrastructure. |
| `tests/test_grouping.py` | Unit tests for all three selectors | Common selector API; Milestone 1 test matrix | Test empty/one/two/six+/duplicate/equal-distance cases; fixed-k; deterministic; padding; mask. | LOW — pure unit tests, no env changes. |
| `docs/env_spec.md` | Freeze interface decisions | — | Record approved answers for D1–D10, A1–A7. | MEDIUM — spec change requires supervisor sign-off. |
| `scripts/train.py` | Unified training entry point | `scripts/train_common.py`, `scripts/train_qmix_highway.py`, etc. | Single command to train with any selector config. | LOW — script addition, no env contract change. |
| `scripts/evaluate.py` | Unified evaluation entry point | `scripts/train.py` pattern | Evaluate trained policies with selected-ID logging. | LOW — same. |
| `scripts/make_report.py` | Report generation | `scripts/evaluate_comparison.py`, `plot_comparison.py` | Generate comparison plots and reports across selectors. | LOW — output generation only. |
| `configs/env.yaml` | Environment configuration | `V2XConfig`, `CarlaMultiAgentEnv` params | YAML-driven env setup including V2X params, selector type, scenario. | LOW — config-driven, no code change if keys match. |
| `configs/train.yaml` | Training configuration | Learner configs (vdn, qmix, mappo) | YAML-driven training hyperparameters, seed, step budget. | LOW — same. |
| `configs/eval.yaml` | Evaluation configuration | — | YAML-driven evaluation protocol, held-out seeds, CI metrics. | LOW — same. |

**No file should be created or modified as part of this audit.** The audit is read-only. These files are listed for future implementation reference only.

### 22. Exact commands to run tests and validate each milestone

**Milestone 1 — Grouping selector tests:**
```bash
python -m pytest tests/test_grouping.py -v
# Expected: all tests pass, covering empty/one/two/six+/duplicate/equal-distance cases
```

**Milestone 1a — V2X channel tests (already passing):**
```bash
python -m pytest tests/test_v2x_channel.py -v
# Expected: 37 passed (verified in audit)
```

**Milestone 2 — Integrated V2X→candidate→observation tests:**
```bash
# Run with toy env adapted for V2X, or with headless CARLA
python -m pytest tests -k "not carla and not highway and not toy" -v
# Expected: 164 passed (current baseline)
# Add new integration tests that verify:
# - V2X range/loss/delay propagate into observation
# - 56-D shape preserved with V2X candidates
# - Selector output feeds correctly to build_observation()
```

**Milestone 3 — Synthetic experiment validation:**
```bash
# Run a synthetic fixture script (to be written) that:
# 1. Creates known vehicle positions
# 2. Runs V2X channel step
# 3. Converts to candidates via messages_to_candidates
# 4. Applies radius and random selectors
# 5. Verifies observation shape (56,), masks, padding, determinism
# 6. Demonstrates radius vs random selecting different IDs
python -c "
from envs.v2x_channel import V2XChannel, V2XConfig, messages_to_candidates
import numpy as np
import math

# Synthetic fixture
ego = {'x': 0.0, 'y': 0.0, 'vx': 10.0, 'vy': 0.0, 'yaw': 0.0}
config = V2XConfig(range_m=50.0, delay_steps=0, loss_prob=0.0)
channel = V2XChannel(config, seed=42)

# Perfect communication: 3 vehicles within 50 m
senders = [
    ('cav_1', np.array([5.0, 0.0, 8.0, 0.0, 0.0, 0.0])),
    ('cav_2', np.array([10.0, 0.0, 6.0, 0.0, 0.0, 0.0])),
    ('cav_3', np.array([20.0, 0.0, 5.0, 0.0, 0.0, 0.0])),
]
received, stats = channel.step(senders)

# Convert for ego
candidates = messages_to_candidates(received['cav_0'], ego)
print('Candidates shape:', candidates.shape)  # (6, 6)

# Radius selector (k=2, radius=7 m) — manual implementation
# Would select candidates within 7 m of ego

# Random selector (k=2, seed=42) — manual implementation
# Would select 2 of 3 randomly

# Verify observation shape with selected candidates
from envs.obs_builder import build_observation, M_CANDIDATES, CAND_FEATURES, EGO, ROUTE, MASK
import numpy as np

# Build observation with 2 selected candidates
route = {'next_waypoint_x': 10.0, 'next_waypoint_y': 0.0,
         'speed_limit': 15.0, 'at_intersection': False,
         'traffic_light_state': 'none', 'distance_to_next': 10.0}
action_mask = np.ones(MASK, dtype=True)

selected = candidates[:2]  # first 2 (nearest) as if selected by selector
obs = build_observation(
    ego={'x': 0.0, 'y': 0.0, 'vx': 10.0, 'vy': 0.0, 'yaw': 0.0,
         'speed': 10.0, 'longitudinal_accel': 1.0, 'heading_error': 0.0,
         'lateral_offset': 0.0, 'lane_width': 4.0, 'lane_index': 1.0,
         'lane_count': 3.0, 'distance_to_goal': 80.0, 'progress_fraction': 0.4},
    route,
    selected,
    action_mask,
)
print('Observation shape:', obs.shape)  # (56,)
print('Observation finite:', np.isfinite(obs).all())
print('Observation in [-1,1]:', (obs >= -1.0).all() and (obs <= 1.0).all())
"
```

### 23. What is already implemented vs what remains to be implemented

| Category | Already implemented | Remains to be implemented |
|---|---|---|
| 56-D observation stack | ✅ Full implementation in `obs_builder.py`; tested in `test_obs_builder.py` | — |
| V2X channel (isolated) | ✅ Full implementation in `v2x_channel.py`; 37/37 tests passing in `test_v2x_channel.py` | — |
| V2X channel integration into CARLA | ❌ NOT implemented | Wire `V2XChannel` into `CarlaMultiAgentEnv`; call `step()` per decision; convert received messages to candidates; feed selected candidates to observation builder. |
| Candidate discovery | ✅ `get_candidates()` in `carla_env.py` returns raw world actors within 50 m | ❌ Replace/reorient to use V2X-delivered messages as primary source; add optional debug mode for raw state. |
| Three grouping selectors (radius, random, learned) | ❌ ALL empty (0 bytes) | Implement `radius.py`, `random_sel.py`, `learned_sel.py` with common API, fixed k=2, deterministic tie-breaking, padding, validity masks. |
| Fixed-k padding and selected-ID logging | ❌ NOT implemented | Add padding to exactly k slots with `valid=0`; log selected teammate IDs in `info` dict and CSV. |
| V2X metrics (sent, delivered, dropped, age) | ✅ Channel has `episode_stats()` with all counters | ❌ Expose these in `info` dict and per-episode CSV from the environment. |
| Scenario diversity (intersection, merge, lane change) | ❌ Only random Town03 routes; `scenario_id="free_drive"` | ❌ Add scripted background vehicles, intersection/merge/lane-change scenarios, real scenario IDs. |
| Reproducible CARLA 4-agent runs | ❌ CARLA missing from this environment; Python 3.13.3 + no GPU | ❌ Set up compatible environment (Python 3.10/3.11 + CARLA 0.9.16 wheel); verify 4-agent determinism. |
| Training/evaluation scripts | ❌ `scripts/train.py`, `evaluate.py`, `make_report.py` are empty | ❌ Implement unified train/eval/report scripts with config-driven selector choice. |
| Config files | ❌ `configs/` empty | ❌ Create `env.yaml`, `train.yaml`, `eval.yaml`, `scenarios.yaml`. |
| Documentation of interface decisions | ✅ `V2X_INTEGRATION_DECISIONS.md` with 17 decisions | ❏ Freeze decisions in `docs/env_spec.md`; resolve all OPEN questions. |
| Test suite portability | ✅ 164 non-CARLA/non-Highway tests pass | ❏ Fix `tests/test_highway_adapter.py` collection; add grouping tests; add integrated V2X tests. |
| Uncertainty analysis across seeds | ❌ Not done | ❏ Run paired seeds across strategies; report mean and 95% CI; test whether differences exceed seed noise. |

---

## Checklist

### Verified facts
- Local `main` and `origin/main` share commit `f1156fa8d1e541497cae270ff999356a0a811b35`
- V2X channel: `envs/v2x_channel.py` with `V2XConfig`, `V2XChannel`, `ReceivedMessage`, `messages_to_candidates()`
- V2X channel tests: 37/37 passing (`tests/test_v2x_channel.py`)
- 56-D observation: ego(8) + route(6) + candidates(36, 6×6) + mask(6), dtype float32, clipped [−1,1]
- Observation contract verified: `test_observation_shape_dtype_and_block_layout` passes
- Candidate sorting: by (distance, actor_id) ascending, established in both `obs_builder` and `v2x_channel`
- Action mask: part of 56-D observation at indices 50-55; `make_action_mask()` produces 6 booleans
- Grouping module: all files 0 bytes; no selector implementation exists
- No `feat/v2x-channel` branch exists; V2X channel was committed directly to `main`
- Repository has 164 passed non-CARLA/non-Highway tests

### Proposed decisions
- **D1 (Who broadcasts?):** CAVs only for main experiment; background as ablation (per `V2X_INTEGRATION_DECISIONS.md:65`)
- **D2 (Is intent used?):** Do not add to 56-D observation; log separately if needed (per `V2X_INTEGRATION_DECISIONS.md:77-79`)
- **D3 (Delay units?):** Keep decision steps at 5 Hz (per `V2X_INTEGRATION_DECISIONS.md:89`)
- **D4 (Loss timing?):** Keep transmission-time Bernoulli loss (per `V2X_INTEGRATION_DECISIONS.md:101`)
- **D5 (Range vs normalization?):** 50 m main experiment; versioned 100 m for robustness (per `V2X_INTEGRATION_DECISIONS.md:113-115`)
- **D6 (When to step channel?)** Once per decision, cached, after the 4-tick loop (per `V2X_INTEGRATION_DECISIONS.md:125-127`)
- **D7 (Candidate source?)** Received messages from V2X channel; raw state only for debug (per `V2X_INTEGRATION_DECISIONS.md:137-139`)
- **D8 (Selection semantics?)** Fixed k=2, deterministic tie-break by (dist, actor_id), padded, logged (per `V2X_INTEGRATION_DECISIONS.md:149-151`)
- **D9 (Policy input?)** Selected-only main mode; all-plus-mask ablation (per `V2X_INTEGRATION_DECISIONS.md:161`)
- **D10 (Metrics?)** V2X counts/age, selected IDs, group size, per-term rewards in `info["v2x"]` and CSV (per `V2X_INTEGRATION_DECISIONS.md:173`)

### Unresolved questions
- D1–D10 all marked OPEN in `V2X_INTEGRATION_DECISIONS.md:269-279` — must be explicitly approved by supervisor before implementation
- A1–A7 from repository audit also pending approval
- CARLA environment compatibility (Python 3.13.3 + CARLA 0.9.16) is unverified
- No live CARLA validation has been performed at this commit

### Safe first implementation step
Implement the common selector API in `grouping/`:
1. `grouping/radius.py`: radius baseline with fixed k, deterministic tie-break, padding
2. `grouping/random_sel.py`: random baseline with seeded RNG, fixed k, padding
3. `tests/test_grouping.py`: test matrix covering all edge cases

These three files are pure Python, require no CARLA or V2X, and establish the research hook that the entire experiment builds on. They can be implemented and tested immediately in the current environment.

### What must not be changed in Nour's work
- `envs/v2x_channel.py` — do not modify the V2X channel implementation, its API, its defaults, or its tests. It is verified and working in isolation.
- `docs/v2x_channel.md` — do not edit the V2X channel documentation.
- `tests/test_v2x_channel.py` — do not modify the V2X channel tests.
- The 56-D observation contract — do not change `OBS_DIM`, candidate feature layout, normalization, or clipping.
- The environment contract in `envs/base.py` — do not change `M_CANDIDATES`, `CAND_FEATURES`, `N_ACTIONS`, or required info keys.

### What cannot be claimed until live CARLA validation
- That the V2X channel provides robustness to delay and packet loss in a driving context.
- That learned teammate selection outperforms radius or random selection.
- That the full 4-agent CARLA stack runs end-to-end.
- That the Highway 25-D checkpoint generalizes to CARLA 56-D.
- That any selection strategy improves driving metrics (collision rate, success rate, travel time).
- That the test suite "passes" without qualifying that CARLA/Highway tests were skipped or deselected.
- That the 15,000-step comparison claims are reproducible from this checkout.