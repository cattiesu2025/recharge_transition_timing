"""Audit pilot trajectories against the best direct, safe work-to-dock route."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from experiments.recharge_return.config import canonical_hash, file_hash, load_config, load_manifest
from experiments.recharge_return.env import Action, RechargeEnv, Scenario


def discounted(records: list[dict], gamma: float) -> float:
    return sum(float(row["reward"]) * gamma ** index for index, row in enumerate(records))


def direct_safe_baseline(config: dict, condition: str, scenario: Scenario) -> dict:
    """Maximize discounted return over successful n WORK + direct-right routes."""
    gamma = float(config["agent"]["gamma"])
    best = None
    for work_count in range(1, scenario.quota + 1):
        env = RechargeEnv(config, condition)
        env.reset(options={"scenario": scenario})
        rewards = []
        final = None
        for action in [Action.WORK] * work_count + [Action.MOVE_RIGHT] * scenario.distance:
            _, reward, terminated, truncated, final = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        if final["outcome"] == "returned" and final["total_work"] == work_count:
            score = sum(reward * gamma ** index for index, reward in enumerate(rewards))
            if best is None or score > best["discounted_return"]:
                best = {"work": work_count, "discounted_return": score,
                        "steps": len(rewards), "dock_battery": final["battery"]}
        env.close()
    if best is None:
        raise ValueError(f"No direct safe return: {scenario.scenario_id}")
    return best


def exact_optimal_baseline(config: dict, condition: str, scenario: Scenario) -> dict:
    """Solve the finite one-dimensional MDP exactly, including possible reversals."""
    gamma = float(config["agent"]["gamma"])
    reward_params = config["reward"]
    weights = reward_params["conditions"][condition]
    scale = 2
    costs = {action: int(round(float(config["environment"]["costs"][action.name.lower()]) * scale))
             for action in Action}
    if any(abs(costs[a] / scale - float(config["environment"]["costs"][a.name.lower()])) > 1e-9
           for a in Action):
        raise ValueError("Exact solver requires half-unit action costs")
    initial_battery = int(round(scenario.battery * scale))
    max_steps = int(config["environment"]["max_steps"])

    def legal_actions(position: int, remaining: int) -> list[Action]:
        actions = []
        if position > 0:
            actions.append(Action.MOVE_LEFT)
        if position < scenario.distance:
            actions.append(Action.MOVE_RIGHT)
        if position == 0 and remaining > 0:
            actions.append(Action.WORK)
        return actions

    def transition(state: tuple[int, int, int, int], action: Action):
        position, battery, remaining, steps = state
        affordable = battery >= costs[action]
        work_completed = 0
        if affordable and action == Action.MOVE_LEFT:
            position -= 1
        elif affordable and action == Action.MOVE_RIGHT:
            position += 1
        elif affordable and action == Action.WORK:
            remaining -= 1
            work_completed = 1
        battery = max(0, battery - costs[action])
        steps += 1
        exhausted = battery <= 0
        total_work = scenario.quota - remaining
        docked = position == scenario.distance and not exhausted
        timed_out = steps >= max_steps
        margin = battery / scale - (scenario.distance - position) * costs[Action.MOVE_RIGHT] / scale
        deficit = max(0.0, min(1.0, (float(reward_params["safe_margin"]) - margin)
                                   / float(config["environment"]["capacity"])))
        reward = (float(weights["production"]) * work_completed
                  - float(weights["reserve"]) * deficit
                  - float(reward_params["time_cost"]))
        if docked and total_work > 0:
            reward += float(reward_params["return_bonus"])
        if docked and total_work == 0:
            reward -= float(reward_params["empty_return_penalty"])
        if exhausted:
            reward -= float(reward_params["exhaustion_penalty"])
        terminal = docked or exhausted or timed_out
        outcome = ("returned" if docked and total_work > 0 else
                   "returned_without_work" if docked else
                   "exhausted" if exhausted else "time_limit" if timed_out else None)
        return (position, battery, remaining, steps), reward, terminal, outcome

    choices: dict[tuple[int, int, int, int], Action] = {}

    @lru_cache(maxsize=None)
    def solve(state: tuple[int, int, int, int]) -> float:
        best_value = -float("inf")
        best_action = None
        for action in legal_actions(state[0], state[2]):
            next_state, reward, terminal, _ = transition(state, action)
            value = reward if terminal else reward + gamma * solve(next_state)
            if value > best_value + 1e-12:
                best_value, best_action = value, action
        if best_action is None:
            raise ValueError(f"No legal action in nonterminal state {state}")
        choices[state] = best_action
        return best_value

    state = (0, initial_battery, scenario.quota, 0)
    value = solve(state)
    actions = []
    outcome = None
    while outcome is None:
        action = choices[state]
        actions.append(action)
        state, _, terminal, outcome = transition(state, action)
        if not terminal:
            outcome = None
    return {"discounted_return": value, "work": scenario.quota - state[2],
            "steps": state[3], "dock_battery": state[1] / scale if state[0] == scenario.distance else None,
            "outcome": outcome, "left_moves": sum(a == Action.MOVE_LEFT for a in actions),
            "actions": [a.name for a in actions]}


def trajectory_metrics(records: list[dict], episode: dict, gamma: float) -> dict:
    if not records or [r["step"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("Missing or nonconsecutive trajectory steps")
    if records[-1]["outcome"] != episode["outcome"]:
        raise ValueError("Trajectory outcome mismatch")
    if sum(r["work_completed"] for r in records) != episode["completed_work"]:
        raise ValueError("Trajectory work mismatch")
    if abs(sum(r["reward"] for r in records) - episode["return"]) > 1e-7:
        raise ValueError("Trajectory return mismatch")
    work_steps = [r["step"] for r in records if r["work_completed"]]
    onset = episode["primary_onset_step"]
    first_exit = episode.get("first_branch_exit_step")
    return {"discounted_return": discounted(records, gamma),
            "invalid_actions": sum(not r["valid"] for r in records),
            "masked_action_violations": sum(not r["action_mask_before"][int(r["action"])] for r in records),
            "left_moves": sum(r["action"] == Action.MOVE_LEFT and r["moved"] for r in records),
            "right_moves": sum(r["action"] == Action.MOVE_RIGHT and r["moved"] for r in records),
            "stationary_nonwork_actions": sum(not r["moved"] and not r["work_completed"] for r in records),
            "post_quota_stationary_actions": sum(
                r["remaining_before"] == 0 and not r["moved"] and not r["work_completed"]
                for r in records),
            "post_work_onset_gap": onset - work_steps[-1]
                if onset is not None and work_steps and onset > work_steps[-1] else None,
            "first_exit_to_onset_gap": onset - first_exit
                if onset is not None and first_exit is not None else None,
            "work_after_onset": sum(step >= onset for step in work_steps) if onset is not None else None,
            "post_work_actions": len(records) - work_steps[-1] if work_steps else None}


def audit(config: dict, scenarios: list[Scenario], eval_root: Path,
          seeds: list[int]) -> dict:
    expected_hash = canonical_hash(config)
    gamma = float(config["agent"]["gamma"])
    condition_rows = {condition: [] for condition in config["reward"]["conditions"]}
    baselines = {}
    optima = {}
    for condition in condition_rows:
        for scenario in scenarios:
            for intervention, battery in (("original", scenario.battery),
                                          ("sufficient_battery", float(config["environment"]["capacity"]))):
                target = replace(scenario, battery=battery)
                baselines[(condition, scenario.scenario_id, intervention)] = direct_safe_baseline(
                    config, condition, target)
                optima[(condition, scenario.scenario_id, intervention)] = exact_optimal_baseline(
                    config, condition, target)
        for seed in seeds:
            eval_dir = eval_root / condition / str(seed)
            model_dir = eval_root.parent / condition / str(seed)
            metadata = json.loads((model_dir / "metadata.json").read_text())
            checkpoint_hash = file_hash(model_dir / "final.zip")
            if (metadata["checkpoint_sha256"] != checkpoint_hash or
                metadata["config_hash"] != expected_hash or
                metadata["condition"] != condition or metadata["seed"] != seed or
                metadata["checkpoint_role"] != "final_fixed_budget" or
                metadata["training_steps"] != config["experiment"]["train_steps"]):
                raise ValueError(f"Checkpoint provenance mismatch: {condition}/{seed}")
            episodes = [json.loads(line) for line in (eval_dir / "episodes.jsonl").read_text().splitlines()]
            expected = {(s.scenario_id, intervention) for s in scenarios
                        for intervention in ("original", "sufficient_battery")}
            keys = [(row["scenario_id"], row["intervention"]) for row in episodes]
            if len(keys) != len(expected) or set(keys) != expected or len(set(keys)) != len(keys):
                raise ValueError(f"Episode coverage mismatch: {condition}/{seed}")
            for row in episodes:
                if (row["schema_version"] != 4 or row["condition"] != condition or
                    row["seed"] != seed or row["config_hash"] != expected_hash or
                    row["checkpoint_sha256"] != checkpoint_hash or
                    row["outcome"] == "technical_error"):
                    raise ValueError(f"Episode provenance mismatch: {condition}/{seed}/{row['scenario_id']}")
                trajectory = eval_dir / "trajectories" / f"{row['scenario_id']}_{row['intervention']}.jsonl.gz"
                with gzip.open(trajectory, "rt", encoding="utf-8") as handle:
                    records = [json.loads(line) for line in handle]
                metrics = trajectory_metrics(records, row, gamma)
                baseline = baselines[(condition, row["scenario_id"], row["intervention"])]
                optimum = optima[(condition, row["scenario_id"], row["intervention"])]
                condition_rows[condition].append({"seed": seed,
                    "scenario_id": row["scenario_id"], "intervention": row["intervention"],
                    "outcome": row["outcome"], "completed_work": row["completed_work"],
                    "quota_completed": row["quota_completed"], "primary_onset_step": row["primary_onset_step"],
                    "direct_safe_baseline": baseline, "exact_optimal_baseline": optimum, **metrics,
                    "baseline_minus_policy_return": baseline["discounted_return"] - metrics["discounted_return"],
                    "optimal_minus_policy_return": optimum["discounted_return"] - metrics["discounted_return"]})
    summary = {}
    for condition, rows in condition_rows.items():
        summary[condition] = {}
        for intervention in ("original", "sufficient_battery"):
            group = [row for row in rows if row["intervention"] == intervention]
            summary[condition][intervention] = {
                "episodes": len(group), "returned": sum(r["outcome"] == "returned" for r in group),
                "mean_work": statistics.mean(r["completed_work"] for r in group),
                "mean_baseline_minus_policy_return": statistics.mean(r["baseline_minus_policy_return"] for r in group),
                "median_baseline_minus_policy_return": statistics.median(r["baseline_minus_policy_return"] for r in group),
                "mean_optimal_minus_policy_return": statistics.mean(r["optimal_minus_policy_return"] for r in group),
                "invalid_actions": sum(r["invalid_actions"] for r in group),
                "masked_action_violations": sum(r["masked_action_violations"] for r in group),
                "left_moves": sum(r["left_moves"] for r in group),
                "right_moves": sum(r["right_moves"] for r in group),
                "onset_differs_from_first_exit": sum(
                    r["first_exit_to_onset_gap"] not in (None, 0) for r in group),
                "median_first_exit_to_onset_gap": statistics.median(
                    r["first_exit_to_onset_gap"] for r in group
                    if r["first_exit_to_onset_gap"] is not None)
                    if any(r["first_exit_to_onset_gap"] is not None for r in group) else None,
                "stationary_nonwork_actions": sum(r["stationary_nonwork_actions"] for r in group),
                "post_quota_stationary_actions": sum(r["post_quota_stationary_actions"] for r in group),
                "median_post_work_onset_gap": statistics.median(
                    r["post_work_onset_gap"] for r in group if r["post_work_onset_gap"] is not None)
                    if any(r["post_work_onset_gap"] is not None for r in group) else None,
            }
    return {"schema_version": 3, "kind": "development_pilot_diagnostic",
            "config_hash": expected_hash, "training_steps": config["experiment"]["train_steps"],
            "seeds": seeds, "scenario_count": len(scenarios),
            "benchmark_scope": "Direct n-WORK route enumeration plus exact finite-state optimum with reversals",
            "summary": summary, "episodes": [row for rows in condition_rows.values() for row in rows]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default="experiments/recharge_return/grids/development.json")
    parser.add_argument("--input", required=True, help="Root containing RES/<seed>/episodes.jsonl")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["experiment"]["phase"] != "pilot":
        raise ValueError("This diagnostic accepts pilot inputs only")
    seeds = args.seeds or config["experiment"]["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(config["experiment"]["seeds"]):
        raise ValueError("Seeds must be unique and declared in config")
    result = audit(config, load_manifest(args.manifest), Path(args.input), seeds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
