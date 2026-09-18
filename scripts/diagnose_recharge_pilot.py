"""Audit pilot trajectories against the best direct, safe work-to-dock route."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from dataclasses import replace
from pathlib import Path

from experiments.recharge_return.config import canonical_hash, file_hash, load_config, load_manifest
from experiments.recharge_return.env import Action, RechargeEnv, Scenario


def discounted(records: list[dict], gamma: float) -> float:
    return sum(float(row["reward"]) * gamma ** index for index, row in enumerate(records))


def direct_safe_baseline(config: dict, condition: str, scenario: Scenario) -> dict:
    """Maximize discounted return over successful n WORK + distance FORWARD routes."""
    gamma = float(config["agent"]["gamma"])
    best = None
    for work_count in range(1, scenario.quota + 1):
        env = RechargeEnv(config, condition)
        env.reset(options={"scenario": scenario})
        rewards = []
        final = None
        for action in [Action.WORK] * work_count + [Action.FORWARD] * scenario.distance:
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
    return {"discounted_return": discounted(records, gamma),
            "invalid_actions": sum(not r["valid"] for r in records),
            "wait_actions": sum(r["action"] == Action.WAIT for r in records),
            "turn_actions": sum(r["action"] in (Action.LEFT, Action.RIGHT) for r in records),
            "post_work_onset_gap": onset - work_steps[-1]
                if onset is not None and work_steps and onset > work_steps[-1] else None,
            "work_after_onset": sum(step >= onset for step in work_steps) if onset is not None else None,
            "post_work_actions": len(records) - work_steps[-1] if work_steps else None}


def audit(config: dict, scenarios: list[Scenario], eval_root: Path,
          seeds: list[int]) -> dict:
    expected_hash = canonical_hash(config)
    gamma = float(config["agent"]["gamma"])
    condition_rows = {condition: [] for condition in config["reward"]["conditions"]}
    baselines = {}
    for condition in condition_rows:
        for scenario in scenarios:
            for intervention, battery in (("original", scenario.battery),
                                          ("sufficient_battery", float(config["environment"]["capacity"]))):
                baselines[(condition, scenario.scenario_id, intervention)] = direct_safe_baseline(
                    config, condition, replace(scenario, battery=battery))
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
                if (row["schema_version"] != 2 or row["condition"] != condition or
                    row["seed"] != seed or row["config_hash"] != expected_hash or
                    row["checkpoint_sha256"] != checkpoint_hash or
                    row["outcome"] == "technical_error"):
                    raise ValueError(f"Episode provenance mismatch: {condition}/{seed}/{row['scenario_id']}")
                trajectory = eval_dir / "trajectories" / f"{row['scenario_id']}_{row['intervention']}.jsonl.gz"
                with gzip.open(trajectory, "rt", encoding="utf-8") as handle:
                    records = [json.loads(line) for line in handle]
                metrics = trajectory_metrics(records, row, gamma)
                baseline = baselines[(condition, row["scenario_id"], row["intervention"])]
                condition_rows[condition].append({"seed": seed,
                    "scenario_id": row["scenario_id"], "intervention": row["intervention"],
                    "outcome": row["outcome"], "completed_work": row["completed_work"],
                    "quota_completed": row["quota_completed"], "primary_onset_step": row["primary_onset_step"],
                    "direct_safe_baseline": baseline, **metrics,
                    "baseline_minus_policy_return": baseline["discounted_return"] - metrics["discounted_return"]})
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
                "invalid_actions": sum(r["invalid_actions"] for r in group),
                "wait_actions": sum(r["wait_actions"] for r in group),
                "turn_actions": sum(r["turn_actions"] for r in group),
                "median_post_work_onset_gap": statistics.median(
                    r["post_work_onset_gap"] for r in group if r["post_work_onset_gap"] is not None)
                    if any(r["post_work_onset_gap"] is not None for r in group) else None,
            }
    return {"schema_version": 1, "kind": "development_pilot_diagnostic",
            "config_hash": expected_hash, "training_steps": config["experiment"]["train_steps"],
            "seeds": seeds, "scenario_count": len(scenarios),
            "benchmark_scope": "Best successful route among n WORK then direct FORWARD, n=1..quota",
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
