

**## 12. Contract v1 (frozen; overrides §9 and §6 where they differ)**



**\*\*Precedence:\*\* this section wins over §9 and §6. Any change needs a message to the other person and a spec update first.**



**\*\*12.1 API\*\***

**- `reset(seed=None) -> (obs, info)`; `obs` is `dict\[agent\_id -> np.ndarray(obs\_dim,)]`.**

**- `step(actions) -> (obs, rewards, terminated, truncated, info)`; `rewards`, `terminated` and `truncated` are dicts keyed by agent id (float / bool / bool).**

**- Attributes on every env: `agents` (list of ids), `obs\_dim`, `n\_actions`, plus the methods `state()` and `action\_masks()`.**

**- Actions are the 6 meta-actions of §2. The toy env may expose `n\_actions=3`, and the toy and highway envs may use their own `obs\_dim`. Only CARLA is fixed at 56.**



**\*\*12.2 Shared termination (team objective)\*\***

**- A collision by any CAV ends the episode for all agents: `terminated\[a]=True` for every `a`, and `info\["collision"]=True`.**

**- All agents at their goal with no collision: `terminated` all True, `success=True`.**

**- Timeout (300 decision steps): `truncated` all True.**

**- A stuck agent for more than 10 s: episode ends and `info\["stuck"]=True`.**

**- Per-agent details go in `info\["per\_agent"]\[agent\_id]` (for example who collided). They are never used for termination.**



**\*\*12.3 Info keys\*\***

**- Every step: `collision` (bool), `goal\_reached` (bool, True only when all agents have reached their goals), `steps` (int).**

**- Last step of an episode, additionally: `success` (bool), `min\_ttc` (float, seconds), `jerk` (float, mean), `scenario\_id`, `town`, `seed`.**

**- Names are singular: `collision` and `min\_ttc`, not `collisions` and `ttc\_min`.**



**\*\*12.4 Reward\*\***

**- Per-agent weights are those of §6: progress +1.0, collision -10, TTC penalty -0.5, comfort -0.05, goal +10 (once, on arrival), idle -0.1. `RewardCalculator` defaults must match.**

**- The team reward used by VDN and QMIX is `mean(r\_i)` plus a -10 team collision penalty, reported in `info\["team\_reward"]`.**

**- A vehicle that reached its goal is frozen and earns no further progress reward.**



**\*\*12.5 Observation\*\***

**- The CARLA observation is 56-D (ego 8, route 6, candidates 6x6, mask 6).**

**- Highway-env is a development fixture. A 25-D observation is accepted for now. A highway adapter producing the 56-D layout is a Phase 1 task for B.**

**- Models trained on 25-D do not transfer to CARLA, so QMIX and VDN are retrained on CARLA with 56-D.**



**\*\*12.6 Reproducibility\*\***

**- The same seed gives the same starts, routes and goals. Different seeds give different starts.**

