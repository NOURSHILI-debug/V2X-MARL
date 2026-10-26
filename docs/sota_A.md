# SOTA rows (Person A): VDN, QMIX, MAPPO

| Method | Paradigm | Credit assignment | Grouping / communication | Scalability | Code available |
|---|---|---|---|---|---|
| VDN (Sunehag et al., 2017) | Value-based, CTDE | Additive: Q_tot = sum Q_i | None (all agents share one team reward) | Good, but limited expressiveness | PyMARL / EPyMARL |
| QMIX (Rashid et al., 2018) | Value-based, CTDE | Monotonic mixing network conditioned on global state | None | Good, standard MARL baseline | PyMARL / EPyMARL |
| MAPPO (Yu et al., 2022) | Policy-gradient, CTDE (centralized critic) | Shared team value / advantage | None | Good, strong on cooperative benchmarks | marlbenchmark/on-policy |

Fill later: results in our CARLA scenarios (after reproduction).
