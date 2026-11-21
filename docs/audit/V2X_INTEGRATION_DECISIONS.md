# V2X Integration Decisions

- Date: 2026-11-16
- Branch / commit: `main` / `d4b5674`
- Purpose: turn the V2X and teammate-selection design into a small set of explicit, testable decisions that can be approved before implementation.
- Read with: `docs/v2x_channel.md`, `docs/env_spec.md`, `docs/audit/REPOSITORY_AUDIT.md`.

## 1. Current state

The repository has two disconnected halves:

```text
IMPLEMENTED BUT ISOLATED
  CAV state -> V2XChannel -> received messages -> messages_to_candidates()

IMPLEMENTED AND USED BY THE ENV
  CARLA actors -> get_candidates() (raw world state) -> obs_builder -> policy
```

Consequences:

- `envs/v2x_channel.py` correctly models range, Bernoulli loss, fixed delay, and intent, but nothing calls it.
- `envs/carla_env.py:get_candidates()` enumerates every `vehicle.*` within 50 m using exact simulator positions, velocities, and yaw (`envs/carla_env.py:702-741`).
- `envs/obs_builder.py` builds the 56-D observation from those raw candidates; there is no intent or message-age feature (`envs/obs_builder.py:133-175`).
- No grouping module exists (`grouping/*.py` are 0 bytes).
- No selected-teammate IDs, V2X counts, or message ages are exposed in `info` or the training CSV.
- The research question therefore has no treatment variable at present.

The five open questions already documented by Nour in `docs/v2x_channel.md:215-231` are reproduced below as D1-D5. D6-D10 are the integration decisions that follow from this audit.

## 2. Target data flow

```text
per decision step t:
  1. Build each broadcaster's message:
       [x, y, vx, vy, heading, intent]
  2. Step V2XChannel once:
       received, stats = channel.step(senders)
  3. For each ego:
       candidates = messages_to_candidates(received[ego], ego_state)
       optionally add intent and age as extra features
  4. Grouping:
       selected = selector(candidates, k=2, mask)
  5. Observation:
       obs = build_observation(ego, route, selected, action_mask)
  6. Policy:
       action = policy(obs, action_mask)
  7. Log:
       received/dropped/mean age, selected IDs, group size, per-term reward
```

The environment must call `channel.step()` exactly once per decision step, not once per agent. The environment must also decide which actors broadcast and how background vehicles are modelled.

## 3. Decision matrix

Status legend: **CURRENT** = what the code does now; **RECOMMENDED** = the proposed default; **APPROVAL** = whether the supervisor/Nour must explicitly sign off before implementation.

### D1 - Who broadcasts V2X messages?

| Field | Content |
|---|---|
| Current | No V2X use. `get_candidates()` returns every `vehicle.*` actor in range, including other CAVs and any map traffic. |
| Contract implication | Spec 4 says each CAV broadcasts; it does not clearly say whether background vehicles broadcast. The channel is agnostic. |
| Research implication | If background vehicles broadcast perfect state, the selector may rely on non-cooperative actors. If only CAVs broadcast, candidate supply may collapse when few CAVs are nearby. This changes both the treatment and the baseline. |
| Recommended | Make it an explicit config: `v2x_broadcasters = cavs_only | cavs_and_background`. Default for the main experiment: `cavs_only`, because the research question is about teammate selection among cooperating CAVs. Run `cavs_and_background` as an ablation only if the supervisor wants mixed-traffic communication. |
| Alternatives | CAVs only; all vehicles; CAVs plus a labelled subset of connected background vehicles. |
| Approval | **YES** - supervisor and Nour. |
| Code / integration impact | `reset()` must know which actors are broadcasters; `step()` must build the sender list; scenario setup must expose a stable actor-to-role map. |

### D2 - Is `intent` exposed to the selector or policy?

| Field | Content |
|---|---|
| Current | The channel message includes `intent` (action 0..5), but `messages_to_candidates()` drops it because the six candidate features are `[rel_x, rel_y, rel_vx, rel_vy, rel_heading, valid]` (`docs/v2x_channel.md:137-153`). |
| Contract implication | Spec 3 says candidates have six features; spec 4 says messages include intent. There is no agreed slot for intent in the 56-D observation. |
| Research implication | Intent is a plausible reason why learned selection could beat radius/random: the selector can choose a teammate whose manoeuvre matters. Dropping intent weakens the learned selector and changes the scientific question. |
| Recommended | Add intent as a separate, versioned feature block rather than silently changing the six-feature candidate row. Preserve the 56-D contract for the baseline comparison; introduce a 62-D or 68-D `contract_v2` only with explicit approval. For the minimum experiment, keep 56-D and log intent separately if the selector cannot consume it. |
| Alternatives | Keep 56-D and ignore intent; encode intent in an unused candidate feature (not recommended, ambiguous); add a separate intent block and version the observation. |
| Approval | **YES** - supervisor. This is an observation-contract change. |
| Code / integration impact | `obs_builder` versioning; checkpoint incompatibility; tests for layout; selector input schema. |

### D3 - Delay units

| Field | Content |
|---|---|
| Current | `V2XConfig.delay_steps` is measured in decision steps; the channel is stepped once per decision. Tests cover 0-3 decision steps. |
| Contract implication | Spec 4 says "ticks" in the table but the robustness column describes decision steps; the implementation and docs choose decision steps. |
| Research implication | 20 Hz ticks would make a 1-3 tick delay only 50-150 ms, which is much weaker than 5 Hz decision-step delay (200-600 ms). The choice changes the difficulty of the robustness condition. |
| Recommended | Keep **decision steps** at the 5 Hz decision rate. Document it as 200/400/600 ms for 1/2/3 steps. Do not mix tick and decision-step units in the same experiment. |
| Alternatives | 20 Hz ticks; milliseconds with a conversion table. |
| Approval | **YES** - confirm in the spec. |
| Code / integration impact | Env must tick the channel once per decision; tests should assert arrival step and age. |

### D4 - Loss timing under delay

| Field | Content |
|---|---|
| Current | Bernoulli loss is drawn at transmission time, before the delay queue (`docs/v2x_channel.md:50-53`). |
| Contract implication | The spec does not define whether loss happens at send or delivery. |
| Research implication | Transmission-time loss is simpler and gives a stable per-message fate; delivery-time loss would model a second failure mode. The difference matters for interpreting age and drop counts. |
| Recommended | Keep loss at transmission time for the main experiment. Record the choice in the spec and in every results table. |
| Alternatives | Loss at delivery; separate send/delivery loss probabilities. |
| Approval | **YES** - confirm in the spec. |
| Code / integration impact | No code change if the current semantics are approved; tests already cover the current behavior. |

### D5 - Range versus candidate normalization

| Field | Content |
|---|---|
| Current | Channel default range is 50 m; `obs_builder.CANDIDATE_RANGE` is 50 m. At a 100 m robustness range, messages between 50 m and 100 m would saturate relative position at +/-1 (`docs/v2x_channel.md:155-158`). |
| Contract implication | The spec proposes 30/50/100 m range tests and a 50 m candidate range. |
| Research implication | If the robustness range is 100 m but normalization stays 50 m, the policy cannot distinguish 60 m from 100 m; the comparison is partly invalid. |
| Recommended | Either keep the main experiment at 50 m, or introduce a versioned observation with `CANDIDATE_RANGE=100` for the 100 m condition. Do not silently saturate. |
| Alternatives | Keep 50 m only; rescale to 100 m for all conditions; add a distance feature independent of the normalized position. |
| Approval | **YES** - supervisor. |
| Code / integration impact | Observation constants and checkpoint compatibility; tests for scaling and clipping. |

### D6 - When is the channel stepped?

| Field | Content |
|---|---|
| Current | Not integrated. |
| Contract implication | The channel docs require one `channel.step()` per decision step, not per agent (`docs/v2x_channel.md:160-180`). |
| Research implication | Per-agent stepping would advance the clock and consume RNG multiple times per decision, changing delay/loss semantics and breaking comparability across k or agent count. |
| Recommended | Step the channel exactly once at the start of each decision step, cache the `received` mapping, and reuse it for all agents. Make the ordering explicit: actions at `t` -> world ticks -> messages at `t+1` -> observations at `t+1`. |
| Alternatives | Step at the end of the previous decision (equivalent if documented); step per simulator tick (not recommended). |
| Approval | No for engineering; **YES** for the timing diagram in the spec. |
| Code / integration impact | `CarlaMultiAgentEnv.step()` structure; determinism tests; reset ordering. |

### D7 - Candidate source: raw world state or received messages?

| Field | Content |
|---|---|
| Current | Raw world actors. This is the single largest gap between the implemented environment and the research design. |
| Contract implication | Spec 4 explicitly says "The candidate set comes from the messages actually received, so delay and loss affect the grouping variants naturally" (`docs/env_spec.md:55`). |
| Research implication | With raw state, delay/loss/range do not affect the policy at all. Any robustness or V2X-overhead claim would be false. With received messages, the selector's input is exactly the communication condition being studied. |
| Recommended | Use received messages for all reported experiments. Keep a `candidate_source = v2x | ground_truth_debug` switch for debugging only; never mix the two in a comparison. |
| Alternatives | Raw state with a separate V2X metric (not sufficient for the research question); hybrid where raw state is used for safety but messages for selection (confounded). |
| Approval | **YES** - supervisor. This is a core research-validity decision. |
| Code / integration impact | `get_candidates()` or a new candidate provider; reset/step channel ownership; tests for range/loss/delay propagation into observations. |

### D8 - Selection semantics and fixed `k`

| Field | Content |
|---|---|
| Current | No grouping. All candidate rows are passed to the observation; no selected IDs are produced. |
| Contract implication | Spec 5: candidates -> grouping -> `k=2` selected teammates; `k` is fixed across radius, random, and learned. |
| Research implication | Without fixed `k`, a comparison confounds group size with selection quality. Without selected IDs, the experiment cannot show what was selected or whether the selector learned anything. |
| Recommended | Every selector returns exactly `k=2` indices, with zero padding and a validity mask when fewer than `k` candidates exist. Tie-break deterministically by `(distance, actor_id)`. Log selected IDs and whether each slot is padded. |
| Alternatives | Variable `k` (not for the main question); soft attention weights without discrete selection; top-k with replacement (not recommended). |
| Approval | **YES** - supervisor and Nour. |
| Code / integration impact | New `grouping` API; observation consumption of selected rows; CSV/info schema; tests for padding and determinism. |

### D9 - Does the policy see all candidates or only selected teammates?

| Field | Content |
|---|---|
| Current | The policy sees all six candidate rows, regardless of selection. |
| Contract implication | Spec 5 says grouping chooses the `k` teammates; the observation description still has six candidate slots. The intended consumption is ambiguous. |
| Research implication | If the policy sees all six candidates, selection has no effect on the policy input and cannot change behavior. If it sees only the selected `k`, selection is the treatment. A middle option (all candidates plus a selection mask) tests attention/selection without removing information. |
| Recommended | Define two modes: `selected_only` (policy sees `k` rows plus padding/mask) for the causal selection test, and `all_plus_mask` (all six rows plus a selected mask) as an ablation. The main experiment should use `selected_only`, because otherwise the treatment is not actually applied. |
| Alternatives | All candidates always (selection only affects reward/logging); attention-weighted features; selected-only with a separate summary vector. |
| Approval | **YES** - supervisor. This determines whether the experiment tests selection at all. |
| Code / integration impact | 56-D observation layout; `build_observation` input; checkpoint compatibility; tests that selected-only changes the observation. |

### D10 - Which V2X and selection metrics are mandatory?

| Field | Content |
|---|---|
| Current | Channel has `sent`, `delivered`, `dropped_by_range`, `dropped_by_loss`, `mean_age`, `in_flight`. The env/trainer does not expose or log them. The Highway CSV marks V2X and selected IDs as `NA`. |
| Contract implication | Spec 10 requires V2X messages received/dropped and selected teammate IDs. |
| Research implication | Without these metrics, the project cannot test H3 (communication overhead trade-off) or audit whether a strategy actually selected different teammates. |
| Recommended | Add to `info["v2x"]` and the per-episode CSV: sent, delivered, dropped_by_range, dropped_by_loss, mean_age, selected IDs per agent, group size, padded slots, and selector name. Also log per-term reward shares. |
| Alternatives | Only aggregate drop rate; only selected IDs; no logging (not acceptable for the research question). |
| Approval | No for engineering; **YES** for the final CSV schema. |
| Code / integration impact | `info` contract addition; CSV columns; tests; plotting/report scripts. |

## 4. Additional decisions surfaced by the audit

These are not part of the ten above but are required for a valid experiment.

| ID | Decision | Recommendation | Approval |
|---|---|---|---|
| A1 | Background traffic in CARLA | Spawn scripted background vehicles per scenario, not only the four CAVs; make density configurable and seeded. | Supervisor |
| A2 | Scenario definitions | Implement intersection, merge, and dense lane change with explicit start/goal regions; replace `scenario_id="free_drive"`. | Supervisor |
| A3 | Team termination | End the team episode only when all agents reach the goal or the team terminates; do not let the first agent's goal end a trainer episode. | Engineering + supervisor |
| A4 | Reward comfort term | Align with the frozen spec or explicitly amend it; log per-term shares because the current comfort term dominates. | Supervisor |
| A5 | Observation versioning | Any change to 56-D layout or candidate features must bump a version and invalidate old checkpoints. | Engineering |
| A6 | Reproducibility | Pin Python/CARLA/gymnasium/highway-env versions and record them in README and run metadata. | Engineering |
| A7 | Held-out evaluation | Reserve test seeds and towns before tuning; report mean and 95% CI. | Supervisor |

## 5. Recommended minimum integration sequence

### Step 1 - Freeze the interface

- Approve D1-D10 and record the answers in `docs/env_spec.md`.
- Choose `candidate_source=v2x`, `broadcasters=cavs_only`, `delay_steps` in decision steps, transmission-time loss, and `selected_only` mode for the main experiment.
- Decide whether intent is added to the observation or only logged.

### Step 2 - Implement grouping in isolation

- `grouping/radius.py`: select up to `k` candidates inside a radius, deterministic tie-break.
- `grouping/random_sel.py`: seeded random sample without replacement, fixed `k`, padding when short.
- `grouping/learned_sel.py`: attention scores over candidate features + mask, top-k, no selection of padded rows.
- `tests/test_grouping.py`: fixed-k, no duplicates, padding, determinism, mask safety, all-invalid input behavior.

### Step 3 - Integrate the channel

- Create one `V2XChannel` per environment instance.
- Reset it with the episode seed.
- Build sender messages once per decision.
- Step once, cache received messages, convert per ego.
- Expose V2X stats in `info`.
- Add an integration test: a candidate beyond range must disappear; a lost message must disappear; a delayed message must carry stale values.

### Step 4 - Integrate selection into observations

- Feed only the selected `k` rows in the main mode.
- Keep a stable six-slot layout with padding and mask, so the observation dimension does not change.
- Log selected IDs and padding.
- Add a test proving two selectors can produce different observations from the same candidate set.

### Step 5 - Build the matched experiment

- Same algorithm, reward, traffic, scenario, seeds, step budget, and evaluation protocol for radius/random/learned.
- Only the selector changes.
- Pair seeds across strategies.
- Evaluate on held-out seeds/towns.
- Report collision rate, success rate, travel time, speed, min TTC, jerk, V2X received/dropped/age, selected IDs, and per-term reward shares.
- Report mean and 95% CI; do not claim significance from a single seed or a single scenario.

## 6. Interface sketch (for discussion; not implemented)

```python
# Once per decision step, before observations.
senders = [
    (agent_id, message_for(agent_id))
    for agent_id in broadcasters
]
received, v2x_stats = channel.step(senders)

for agent_id in agents:
    ego = ego_state(agent_id)
    candidates = messages_to_candidates(received[agent_id], ego)
    selected, selected_ids, padded = selector.select(
        candidates,
        ego=ego,
        k=2,
    )
    obs = build_observation(
        ego=ego,
        route=route_state(agent_id),
        candidates=selected,
        action_mask=action_mask(agent_id),
    )
```

Requirements for this sketch:

- `channel.step()` is called once per decision.
- `received[agent_id]` may be empty; the selector must pad to `k`.
- `selected` uses the same six-feature candidate layout as `obs_builder`.
- `selected_ids` and `padded` are logged, not silently dropped.
- The selector never sees raw world state in the reported experiment.

## 7. Decision checklist

| ID | Question | Recommended answer | Owner | Approval | Status |
|---|---|---|---|---|---|
| D1 | Who broadcasts? | CAVs only for main; background as ablation | Nour | Supervisor | OPEN |
| D2 | Is intent used? | Add a versioned intent block or log separately | Nour/Sourour | Supervisor | OPEN |
| D3 | Delay units? | Decision steps at 5 Hz | Nour | Supervisor | OPEN |
| D4 | Loss timing? | Transmission time | Nour | Supervisor | OPEN |
| D5 | Range vs normalization? | 50 m main; versioned 100 m for robustness | Nour/Sourour | Supervisor | OPEN |
| D6 | When to step channel? | Once per decision, cached | Nour | Engineering | OPEN |
| D7 | Candidate source? | Received messages; raw state only for debug | Sourour/Nour | Supervisor | OPEN |
| D8 | Selection semantics? | Fixed `k=2`, deterministic, padded, logged | Nour | Supervisor | OPEN |
| D9 | Policy input? | Selected-only main; all-plus-mask ablation | Sourour/Nour | Supervisor | OPEN |
| D10 | Metrics? | V2X counts/age, selected IDs, group size, per-term rewards | Sourour | Supervisor schema | OPEN |

## 8. What must not happen

- Do not claim V2X robustness while candidates come from raw world state.
- Do not compare radius/random/learned with different `k`, different candidate counts, or different communication budgets.
- Do not silently change the 56-D observation and reuse old checkpoints.
- Do not report a single-seed result as evidence of superiority.
- Do not tune on the held-out test seeds/towns.
- Do not treat the 25-D Highway checkpoint as a CARLA policy.
- Do not log selected IDs as `NA` and then make a selection claim.
