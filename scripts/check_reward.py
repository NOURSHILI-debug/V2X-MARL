"""Compare scripted keep and forced-crash returns on paired CARLA seeds.

Run with CARLA 0.9.16 already running: ``python scripts/check_reward.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.carla_env import CarlaMultiAgentEnv, KEEP
from scripts.test_termination import _setup_forced_collision


def _episode(env, seed, crash):
    if crash:
        from envs.carla_env import BRAKE, ACCEL, MAX_STEPS

        front_id, rear_id = _setup_forced_collision(env, seed=seed)
        actions = {front_id: BRAKE, rear_id: ACCEL}
    else:
        env.reset(seed=seed)
        actions = {agent_id: KEEP for agent_id in env.agents}
        MAX_STEPS = env.max_steps

    total = 0.0
    terms = {}
    diagnostics = {
        "agent_steps": 0,
        "absolute_a_long": 0.0,
        "absolute_jerk": 0.0,
        "absolute_comfort": 0.0,
        "absolute_reward": 0.0,
    }
    for _ in range(MAX_STEPS):
        _, rewards, terminated, truncated, info = env.step(actions)
        total += float(info["team_reward"])
        for agent_id, details in info["per_agent"].items():
            for name, value in details.get("reward_terms", {}).items():
                terms[name] = terms.get(name, 0.0) + float(value)
            diagnostics["agent_steps"] += 1
            diagnostics["absolute_a_long"] += abs(float(details.get("a_long", 0.0)))
            diagnostics["absolute_jerk"] += abs(float(details.get("jerk", 0.0)))
            diagnostics["absolute_comfort"] += abs(
                float(details.get("reward_terms", {}).get("comfort", 0.0))
            )
            diagnostics["absolute_reward"] += abs(float(rewards[agent_id]))
        if info["collision"]:
            terms["team_collision"] = terms.get("team_collision", 0.0) - 10.0
        if info["collision"] or all(terminated.values()) or any(truncated.values()):
            return total, terms, diagnostics
    raise RuntimeError(f"seed {seed}: episode did not finish within {MAX_STEPS} steps")


def main():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "UNAVAILABLE"
    env = None
    completed = False
    error = None
    try:
        env = CarlaMultiAgentEnv(town="Town03", n_agents=2)
        print(f"git_sha={sha} seeds=0..9 CARLA=live-server", flush=True)
        returns = {"keep": [], "crash": []}
        sums = {"keep": {}, "crash": {}}
        diagnostics = {
            name: {
                "agent_steps": 0,
                "absolute_a_long": 0.0,
                "absolute_jerk": 0.0,
                "absolute_comfort": 0.0,
                "absolute_reward": 0.0,
            }
            for name in ("keep", "crash")
        }
        for seed in range(10):
            for name, crash in (("keep", False), ("crash", True)):
                episode_return, terms, episode_diagnostics = _episode(env, seed, crash)
                returns[name].append(episode_return)
                for term, value in terms.items():
                    sums[name][term] = sums[name].get(term, 0.0) + value
                for key, value in episode_diagnostics.items():
                    diagnostics[name][key] += value
        completed = True
    except BaseException as exc:
        error = exc
    finally:
        if env is not None:
            env.close()

    if completed:
        for name in ("keep", "crash"):
            mean_return = sum(returns[name]) / len(returns[name])
            print(f"{name} mean team return: {mean_return:.6f}")
            print(f"{name} per-term sums: {sums[name]}")
            comfort_share = (
                diagnostics[name]["absolute_comfort"]
                / diagnostics[name]["absolute_reward"]
                if diagnostics[name]["absolute_reward"] else 0.0
            )
            print(f"{name} comfort share of total |reward|: {comfort_share:.1%}")
            print(f"{name} comfort exceeds 30%: {comfort_share > 0.30}")
        keep_steps = diagnostics["keep"]["agent_steps"]
        mean_a_long = diagnostics["keep"]["absolute_a_long"] / keep_steps if keep_steps else 0.0
        mean_jerk = diagnostics["keep"]["absolute_jerk"] / keep_steps if keep_steps else 0.0
        print(f"keep mean |a_long| per agent-step: {mean_a_long:.6f} m/s^2")
        print(f"keep mean |jerk| per agent-step: {mean_jerk:.6f} m/s^3")
        print(f"keep beats crash: {sum(returns['keep']) / 10 > sum(returns['crash']) / 10}")
        print("STATUS COMPLETE", flush=True)
    else:
        print("STATUS INCOMPLETE", flush=True)
        if error is not None:
            raise SystemExit(1) from error


if __name__ == "__main__":
    main()
