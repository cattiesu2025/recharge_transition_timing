#!/usr/bin/env python3
"""Audit denominators and aggregate matched return-onset estimates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.recharge_return.config import canonical_hash, load_config, load_manifest  # noqa: E402


KEYS = ("condition", "seed", "scenario_id", "intervention")
CONDITIONS = ("RES", "BAL", "PROD")
INTERVENTIONS = ("original", "sufficient_battery")


def read_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("episodes.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{number}") from error
    if not rows:
        raise ValueError("No episodes.jsonl rows found")
    return rows


def validate_completeness(rows: list[dict[str, Any]], seeds: list[int], ids: list[str],
                          expected_config_hash: str | None = None) -> dict[str, Any]:
    expected = {(c, s, i, v) for c in CONDITIONS for s in seeds for i in ids for v in INTERVENTIONS}
    keys = [tuple(row.get(k) for k in KEYS) for row in rows]
    counts = Counter(keys)
    actual = set(keys)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    duplicates = sorted(k for k, count in counts.items() if count > 1)
    errors = sum(row.get("outcome") == "technical_error" for row in rows)
    schema_errors = sum(row.get("schema_version") != 3 for row in rows)
    config_mismatches = sum(row.get("config_hash") != expected_config_hash for row in rows) if expected_config_hash else 0
    return {"complete": not (missing or unexpected or duplicates or errors or schema_errors or config_mismatches),
            "expected_rows": len(expected), "actual_rows": len(rows),
            "missing": [dict(zip(KEYS, k)) for k in missing],
            "unexpected": [dict(zip(KEYS, k)) for k in unexpected],
            "duplicates": [dict(zip(KEYS, k)) for k in duplicates],
            "technical_errors": errors, "schema_errors": schema_errors,
            "config_mismatches": config_mismatches}


def seed_contrasts(rows: list[dict[str, Any]], seeds: list[int], minimum: int) -> list[dict[str, Any]]:
    lookup = {(r["condition"], r["seed"], r["scenario_id"]): r for r in rows
              if r["intervention"] == "original" and r.get("outcome") != "technical_error"}
    ids = sorted({r["scenario_id"] for r in rows if r["intervention"] == "original"})
    result = []
    for seed in seeds:
        differences = []
        for scenario in ids:
            res = lookup.get(("RES", seed, scenario))
            prod = lookup.get(("PROD", seed, scenario))
            if res and prod and res["primary_observed"] and prod["primary_observed"]:
                differences.append(int(prod["primary_onset_step"]) - int(res["primary_onset_step"]))
        result.append({"seed": seed, "joint_events": len(differences),
                       "estimable": len(differences) >= minimum,
                       "median_prod_minus_res_steps": median(differences) if differences else None})
    return result


def bootstrap_seed_median(values: list[float], samples: int, seed: int) -> dict[str, float]:
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Bootstrap requires finite seed estimates")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.median(data[rng.integers(0, len(data), (samples, len(data)))], axis=1)
    return {"estimate": float(np.median(data)),
            "ci_lower": float(np.quantile(draws, 0.025)),
            "ci_upper": float(np.quantile(draws, 0.975))}


def exact_sign_test(values: list[float]) -> dict[str, Any]:
    positive = sum(v > 0 for v in values)
    negative = sum(v < 0 for v in values)
    n = positive + negative
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n if n else 0.5
    return {"positive": positive, "negative": negative, "ties": len(values) - n,
            "two_sided_p": min(1.0, 2 * tail)}


def aggregate(config: dict[str, Any], manifest: list[Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = [int(s) for s in config["experiment"]["seeds"]]
    integrity = validate_completeness(rows, seeds, [s.scenario_id for s in manifest],
                                      canonical_hash(config))
    contrasts = seed_contrasts(rows, seeds, int(config["evaluation"]["minimum_joint_events"]))
    event_groups = defaultdict(list)
    outcomes = Counter()
    detectors = defaultdict(list)
    for row in rows:
        key = (row["condition"], row["intervention"])
        event_groups[key].append(row)
        outcomes[(*key, row["terminal_category"])] += 1
        for name, detection in row.get("detectors", {}).items():
            detectors[(*key, name)].append(detection)
    rates = []
    for (condition, intervention), group in sorted(event_groups.items()):
        rates.append({"condition": condition, "intervention": intervention,
                      "episodes": len(group),
                      "confirmed_returns": sum(bool(r["primary_observed"]) for r in group),
                      "task_successes": sum(bool(r.get("task_success")) for r in group),
                      "dock_arrivals": sum(bool(r.get("docked")) for r in group),
                      "empty_returns": sum(r.get("outcome") == "returned_without_work" for r in group),
                      "quota_completions": sum(bool(r.get("quota_completed")) for r in group),
                      "exhaustions": sum(bool(r.get("exhausted")) for r in group),
                      "completed_work_mean": sum(r.get("completed_work", 0) for r in group) / len(group),
                      "returned_work_mean": sum(r.get("returned_work", 0) for r in group) / len(group),
                      "dock_battery_mean": (sum(r["dock_battery"] for r in group if r.get("dock_battery") is not None)
                                             / sum(r.get("dock_battery") is not None for r in group))
                                             if any(r.get("dock_battery") is not None for r in group) else None})
    detector_rates = [{"condition": c, "intervention": v, "detector": name,
                       "episodes": len(group), "events": sum(bool(d["observed"]) for d in group)}
                      for (c, v, name), group in sorted(detectors.items())]
    all_estimable = len(contrasts) == len(seeds) and all(r["estimable"] for r in contrasts)
    formal = config["experiment"]["phase"] == "formal"
    estimate = None
    sign = None
    if formal and integrity["complete"] and all_estimable:
        values = [float(r["median_prod_minus_res_steps"]) for r in contrasts]
        estimate = bootstrap_seed_median(values, int(config["evaluation"]["bootstrap_samples"]),
                                         int(config["evaluation"]["bootstrap_seed"]))
        estimate["supports_R1"] = estimate["ci_lower"] > 0
        sign = exact_sign_test(values)
    return {"schema_version": 3, "phase": config["experiment"]["phase"],
            "integrity": integrity, "all_seeds_estimable": all_estimable,
            "seed_contrasts": contrasts, "confirmatory_prod_minus_res_steps": estimate,
            "exact_sign_test": sign, "event_and_task_counts": rates,
            "terminal_counts": [{"condition": c, "intervention": v, "category": o, "count": n}
                                for (c, v, o), n in sorted(outcomes.items())],
            "detector_counts": detector_rates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = aggregate(load_config(args.config), load_manifest(args.manifest),
                        read_rows(Path(args.input)))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output / "summary.json")


if __name__ == "__main__":
    main()
