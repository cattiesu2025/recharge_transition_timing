"""CLI for feasibility probes, fixed-budget training, freezing, and evaluation."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent import DoubleDQN, train_condition
from .config import (canonical_hash, file_hash, load_config, load_manifest,
                     verify_frozen_inputs, write_freeze_bundle)
from .env import Action, RechargeEnv, Scenario, rollout
from .events import DetectorConfig, auxiliary_events, detect_onset


PACKAGE = Path(__file__).resolve().parent


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def reference_policy(env: RechargeEnv) -> int:
    """A deterministic threshold controller for task feasibility only."""
    safe = float(env.config["reward"]["safe_margin"])
    if env.position == (1, 4):
        if env.remaining > 0 and env.battery - env._cost(Action.WORK) - env.minimum_energy_to_dock() >= safe:
            return Action.WORK
        return Action.FORWARD if env.direction == 0 else Action.LEFT
    if env.direction == 0:
        return Action.FORWARD
    return Action.LEFT


def classify(result: dict[str, Any], onset: dict[str, Any]) -> str:
    outcome = result["outcome"]
    observed = onset["observed"]
    if outcome == "returned" and not observed:
        return "returned_without_confirmed_onset"
    if outcome == "returned_without_work":
        return outcome
    if outcome == "exhausted":
        return "exhausted_after_confirmed_return" if observed else "exhausted_before_confirmed_return"
    if outcome == "time_limit" and not observed:
        return "administratively_censored"
    return outcome


def evaluate_one(config: dict[str, Any], condition: str, seed: int,
                 scenario: Scenario, intervention: str, model: DoubleDQN,
                 checkpoint_hash: str, output: Path) -> dict[str, Any]:
    env = RechargeEnv(config, condition)
    result = rollout(env, scenario, lambda obs: int(model.predict(obs, deterministic=True)[0]))
    records = result["records"]
    detector = DetectorConfig(int(config["detector"]["window_steps"]),
                              int(config["detector"]["progress_moves"]))
    primary = detect_onset(records, detector).to_dict()
    detectors = {"primary": primary}
    for k in config["detector"]["sensitivity_progress_moves"]:
        detectors[f"progress_{k}"] = detect_onset(records,
            DetectorConfig(detector.window_steps, int(k))).to_dict()
    onset_record = next((r for r in records if r["step"] == primary["onset_step"]), None)
    aux = auxiliary_events(records, scenario.distance + 1)
    trajectory = output / "trajectories" / f"{scenario.scenario_id}_{intervention}.jsonl.gz"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(trajectory, "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    row = {"schema_version": 3, "condition": condition, "seed": seed,
           "scenario_id": scenario.scenario_id, "intervention": intervention,
           "outcome": result["outcome"], "terminal_category": classify(result, primary),
           "primary_observed": primary["observed"],
           "primary_onset_step": primary["onset_step"],
           "primary_confirmation_step": primary["confirmation_step"],
           "candidate_count": len(primary["candidates"]),
           "detectors": detectors, **aux, "completed_work": result["final"]["total_work"],
           "task_success": result["outcome"] == "returned",
           "docked": result["final"]["docked"],
           "quota_completed": result["final"]["remaining"] == 0,
           "returned_work": result["final"]["total_work"] if result["outcome"] == "returned" else 0,
           "dock_battery": result["final"]["battery"] if result["final"]["docked"] else None,
           "exhausted": result["outcome"] == "exhausted",
           "walk_steps": sum(r["action"] == Action.FORWARD and r["moved"] for r in records),
           "work_steps": sum(r["work_completed"] for r in records),
           "onset_battery": onset_record["battery"] if onset_record else None,
           "onset_minimum_energy_to_dock": onset_record["minimum_energy_to_dock"] if onset_record else None,
           "onset_margin": onset_record["margin"] if onset_record else None,
           "onset_remaining": onset_record["remaining"] if onset_record else None,
           "return": result["return"], "checkpoint_sha256": checkpoint_hash,
           "config_hash": canonical_hash(config), "trajectory": str(trajectory)}
    env.close()
    return row


def command_probe(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    scenarios = load_manifest(args.manifest)
    rows = []
    for scenario in scenarios:
        for intervention, battery in (("original", scenario.battery),
                                      ("sufficient_battery", float(config["environment"]["capacity"]))):
            target = replace(scenario, battery=battery)
            env = RechargeEnv(config)
            result = rollout(env, target, lambda _: reference_policy(env))
            rows.append({"scenario_id": scenario.scenario_id, "intervention": intervention,
                         "outcome": result["outcome"], "completed_work": result["final"]["total_work"],
                         "dock_battery": result["final"]["battery"] if result["final"]["docked"] else None,
                         "onset": detect_onset(result["records"], DetectorConfig(**{
                             "window_steps": config["detector"]["window_steps"],
                             "progress_moves": config["detector"]["progress_moves"]})).to_dict()})
            env.close()
    _jsonl(Path(args.output), rows)
    print(args.output)


def command_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if config["experiment"]["phase"] == "formal" and not args.freeze_manifest:
        raise SystemExit("Formal training requires --freeze-manifest")
    if config["experiment"]["phase"] == "formal" and args.steps is not None:
        raise SystemExit("Formal training must use the frozen step budget")
    if args.freeze_manifest:
        verify_frozen_inputs(args.freeze_manifest, [args.config])
    if args.seed not in config["experiment"]["seeds"] and not args.allow_smoke_seed:
        raise SystemExit("Seed not declared in config")
    print(train_condition(config, args.condition, args.seed, args.output, args.device, args.steps))


def command_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    scenarios = load_manifest(args.manifest)
    if config["experiment"]["phase"] == "formal" and not args.freeze_manifest:
        raise SystemExit("Formal evaluation requires --freeze-manifest")
    if args.freeze_manifest:
        verify_frozen_inputs(args.freeze_manifest, [args.config, args.manifest])
    checkpoint = Path(args.checkpoint)
    metadata = json.loads(checkpoint.with_name("metadata.json").read_text())
    checkpoint_hash = file_hash(checkpoint)
    if metadata["checkpoint_sha256"] != checkpoint_hash or metadata["config_hash"] != canonical_hash(config):
        raise ValueError("Checkpoint hash or config hash mismatch")
    if metadata["condition"] != args.condition or metadata["seed"] != args.seed:
        raise ValueError("Checkpoint condition or seed mismatch")
    if (config["experiment"]["phase"] == "formal" and
        (metadata["checkpoint_role"] != "final_fixed_budget" or
         metadata["training_steps"] != config["experiment"]["train_steps"])):
        raise ValueError("Formal evaluation requires the frozen final-budget checkpoint")
    model = DoubleDQN.load(checkpoint, device="cpu")
    output = Path(args.output)
    rows = []
    candidates = []
    errors = 0
    for scenario in scenarios:
        for intervention, battery in (("original", scenario.battery),
                                      ("sufficient_battery", float(config["environment"]["capacity"]))):
            target = replace(scenario, battery=battery)
            try:
                row = evaluate_one(config, args.condition, args.seed, target,
                                   intervention, model, checkpoint_hash, output)
            except Exception as error:
                errors += 1
                row = {"schema_version": 3, "condition": args.condition, "seed": args.seed,
                       "scenario_id": scenario.scenario_id, "intervention": intervention,
                       "outcome": "technical_error", "terminal_category": "technical_error",
                       "primary_observed": False, "primary_onset_step": None,
                       "detectors": {}, "error_type": type(error).__name__, "error_message": str(error),
                       "config_hash": canonical_hash(config), "checkpoint_sha256": checkpoint_hash}
            rows.append(row)
            for label, detection in row["detectors"].items():
                for candidate in detection["candidates"]:
                    candidates.append({"condition": args.condition, "seed": args.seed,
                                       "scenario_id": scenario.scenario_id,
                                       "intervention": intervention, "detector": label, **candidate})
    _jsonl(output / "episodes.jsonl", rows)
    _jsonl(output / "candidates.jsonl", candidates)
    print(output / "episodes.jsonl")
    if errors:
        raise SystemExit(f"Evaluation recorded {errors} technical errors")


def command_freeze(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if config["experiment"]["phase"] == "formal" and not args.confirm_calibrated:
        raise SystemExit("Formal freeze requires --confirm-calibrated")
    if config["experiment"]["phase"] == "formal" and not (args.pip_lock and args.python_lock):
        raise SystemExit("Formal freeze requires --pip-lock and --python-lock")
    print(json.dumps(write_freeze_bundle(args.config, [args.development_manifest,
          args.held_out_manifest, *([args.pip_lock] if args.pip_lock else []),
          *([args.python_lock] if args.python_lock else [])], args.protocol, args.output), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    default_config = str(PACKAGE / "configs" / "pilot_no_wait_600k.yaml")
    dev = str(PACKAGE / "grids" / "development.json")
    held = str(PACKAGE / "grids" / "held_out.json")
    p = sub.add_parser("probe")
    p.add_argument("--config", default=default_config)
    p.add_argument("--manifest", default=dev)
    p.add_argument("--output", required=True)
    p.set_defaults(func=command_probe)
    p = sub.add_parser("train")
    p.add_argument("--config", default=default_config)
    p.add_argument("--condition", choices=["RES", "BAL", "PROD"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--freeze-manifest")
    p.add_argument("--allow-smoke-seed", action="store_true")
    p.set_defaults(func=command_train)
    p = sub.add_parser("evaluate")
    p.add_argument("--config", default=default_config)
    p.add_argument("--manifest", default=held)
    p.add_argument("--condition", choices=["RES", "BAL", "PROD"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--freeze-manifest")
    p.set_defaults(func=command_evaluate)
    p = sub.add_parser("freeze")
    p.add_argument("--config", default=default_config)
    p.add_argument("--development-manifest", default=dev)
    p.add_argument("--held-out-manifest", default=held)
    p.add_argument("--protocol", default=str(PACKAGE / "protocol.md"))
    p.add_argument("--output", required=True)
    p.add_argument("--confirm-calibrated", action="store_true")
    p.add_argument("--pip-lock")
    p.add_argument("--python-lock")
    p.set_defaults(func=command_freeze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
