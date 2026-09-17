"""Config, scenario, and freeze validation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from .env import Scenario


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"experiment", "environment", "reward", "detector", "agent", "evaluation"}:
        raise ValueError("Config must contain the six protocol sections")
    if set(value["reward"]["conditions"]) != {"RES", "BAL", "PROD"}:
        raise ValueError("Conditions must be RES, BAL, PROD")
    if value["experiment"]["phase"] not in {"pilot", "formal"}:
        raise ValueError("Phase must be pilot or formal")
    if value["experiment"]["phase"] == "formal" and len(value["experiment"].get("seeds", [])) != 20:
        raise ValueError("Formal protocol requires exactly 20 declared seeds")
    if len(set(value["experiment"].get("seeds", []))) != len(value["experiment"].get("seeds", [])):
        raise ValueError("Declared seeds must be unique")
    if not 0 < float(value["environment"]["capacity"]):
        raise ValueError("Capacity must be positive")
    if int(value["detector"]["window_steps"]) < int(value["detector"]["progress_moves"]):
        raise ValueError("Detector progress exceeds window")
    return value


def load_manifest(path: str | Path) -> list[Scenario]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty list")
    scenarios = [Scenario.from_dict(row) for row in rows]
    if len({s.scenario_id for s in scenarios}) != len(scenarios):
        raise ValueError("Duplicate scenario id")
    for s in scenarios:
        if not 4 <= s.distance <= 9 or s.battery <= 0 or s.quota <= 0:
            raise ValueError(f"Invalid scenario: {s.scenario_id}")
    return scenarios


def write_freeze_bundle(config: str | Path, manifests: list[str | Path],
                        protocol: str | Path, output: str | Path) -> dict[str, Any]:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    files = [Path(config), *(Path(p) for p in manifests), Path(protocol)]
    if len({p.name for p in files}) != len(files):
        raise ValueError("Freeze inputs need unique names")
    hashes = {}
    for path in files:
        shutil.copyfile(path, target / path.name)
        hashes[path.name] = file_hash(target / path.name)
    bundle = {"schema_version": 1, "files": hashes}
    (target / "manifest.sha256.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return bundle


def verify_frozen_inputs(bundle: str | Path, paths: list[str | Path]) -> None:
    expected = json.loads(Path(bundle).read_text(encoding="utf-8"))["files"]
    for path in map(Path, paths):
        if expected.get(path.name) != file_hash(path):
            raise ValueError(f"Frozen input hash mismatch: {path}")
