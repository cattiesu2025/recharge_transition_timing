"""Rescore fixed v0.7.2 checkpoints on expanded validation; never train."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch

from .grid_run import GridDoubleDQN, SEEDS, sha
from .grid_selection import evaluate, select_best, validation_layouts, write_json
from scripts.sweep_grid_routes import BATTERIES, CONDITIONS, generated_tasks, layouts

SOURCE = Path("outputs/recharge_return_v0.7.2_grid_selection")
ROOT = Path("outputs/recharge_return_v0.7.3_expanded_validation")
KINDS = ("aligned", "detour", "scattered")


def signature(tasks):
    return tuple(sorted(tuple(p) for p in tasks.values()))


def scattered_tasks(count, seed):
    pool = [(x, y) for x in range(10) for y in range(10)
            if (x, y) != (0, 0) and not (x >= 5 and y >= 5)]
    rng = np.random.default_rng(seed)
    points = sorted(pool[i] for i in rng.choice(len(pool), count, replace=False))
    return {chr(65 + i): p for i, p in enumerate(points)}


def manifest(smoke=False):
    seen = {signature(t) for t in [*layouts().values(), *validation_layouts().values()]}
    result = dict(version="v0.7.3", phase="smoke" if smoke else "posthoc_development_pilot",
                  batteries=BATTERIES, selection_metric="mean_undiscounted_episode_return",
                  tie_break="earliest_step", development_layouts=layouts(), accepted_seeds={})
    for split, start in (("validation", 9300), ("audit", 9400)):
        scenes, accepted = {}, {}
        for kind in KINDS:
            for count in (4, 6):
                seeds = []
                seed = start
                while len(seeds) < (1 if smoke else 10):
                    if seed >= start + 10000:
                        raise RuntimeError("Geometric de-duplication exhausted seed range")
                    tasks = scattered_tasks(count, seed) if kind == "scattered" else generated_tasks(kind, count, seed)
                    key = signature(tasks)
                    if key not in seen:
                        seen.add(key)
                        seeds.append(seed)
                        scenes[f"{kind}_{count}_{seed}"] = tasks
                    seed += 1
                accepted[f"{kind}_{count}"] = seeds
        result[f"{split}_layouts"] = scenes
        result["accepted_seeds"][split] = accepted
    return result


def read_json(path):
    return json.loads(path.read_text())


def require(test, message):
    if not test:
        raise ValueError(message)


def verify_source(source, condition, seed):
    """Validate all twelve candidates before deserializing any model."""
    meta = read_json(source / "metadata.json")
    spec = read_json(source / "specification.json")
    selection = read_json(source / "selection.json")
    require(meta["version"] == "v0.7.2" and meta["phase"] == "development_pilot"
            and meta["status"] == "complete", "Requires completed v0.7.2 pilot, not smoke")
    require(meta["condition"] == condition and meta["seed"] == seed, "Condition/seed mismatch")
    require(meta["training_steps"] == 600000 and meta["updates"] == 148750, "Incomplete training")
    hashes = {name: sha(source / name) for name in ("metadata.json", "specification.json", "selection.json")}
    require(hashes["specification.json"] == meta["specification_sha256"] == selection["specification_sha256"],
            "Specification hash mismatch")
    require(hashes["selection.json"] == meta["selection_sha256"], "Selection hash mismatch")
    require(spec["version"] == "v0.7.2" and spec["train_steps"] == 600000
            and spec["checkpoint_interval"] == 50000, "Unexpected source specification")
    # Includes the original config/environment/agent/evaluator and generation rules.
    for name, digest in spec["source_hashes"].items():
        require(sha(name) == digest, f"Source implementation changed: {name}")
    candidates = selection["candidates"]
    require([c["steps"] for c in candidates] == list(range(50000, 600001, 50000)), "Incomplete candidate schedule")
    require(selection["metric"] == "mean_undiscounted_episode_return"
            and selection["tie_break"] == "earliest_step", "Unexpected original selection rule")
    require(select_best(candidates) == selection["selected"], "Original selection inconsistent")
    for c in candidates:
        name = f"checkpoints/step_{c['steps']:06d}.zip"
        require(c["checkpoint"] == name, "Unexpected checkpoint path")
        hashes[name] = sha(source / name)
        require(hashes[name] == c["checkpoint_sha256"], "Checkpoint hash mismatch")
        with zipfile.ZipFile(source / name) as archive:
            require(archive.testzip() is None, "Corrupt checkpoint ZIP")
            data = json.loads(archive.read("data"))
        require(data["num_timesteps"] == c["steps"] and data["seed"] == seed
                and data["_n_updates"] == c["updates"] == (c["steps"] - 5000) // 4,
                "Checkpoint step/seed/update mismatch")
    for role, c in (("best", selection["selected"]), ("final", candidates[-1])):
        hashes[f"{role}.zip"] = sha(source / f"{role}.zip")
        require(hashes[f"{role}.zip"] == meta[f"{role}_sha256"] == c["checkpoint_sha256"], "Role hash mismatch")
    require(meta["best_steps"] == selection["selected"]["steps"], "Best-step mismatch")
    return spec, candidates, hashes


def run(condition, seed, source, output, smoke=False):
    require(condition in CONDITIONS and seed in SEEDS, "Undeclared condition/seed")
    if output.exists():
        raise FileExistsError(output)
    require(not output.resolve().is_relative_to(source.resolve())
            and not source.resolve().is_relative_to(output.resolve()), "Source/output must be disjoint")
    require(not smoke or "smoke" in str(output), "Smoke output must contain smoke")
    spec, old_candidates, hashes = verify_source(source, condition, seed)
    new_manifest = manifest(smoke)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", new_manifest)
    provenance = dict(source=str(source), input_sha256=hashes, condition=condition, seed=seed,
                      original_best_steps=read_json(source / "metadata.json")["best_steps"],
                      source_hashes={p: sha(p) for p in (
                          "experiments/recharge_return/grid_reselect.py",
                          "experiments/recharge_return/expanded_validation.md",
                          "experiments/recharge_return/grid_selection.py")})
    write_json(output / "provenance.json", provenance)
    torch.set_num_threads(1)
    chosen_candidates = [old_candidates[0], old_candidates[-1]] if smoke else old_candidates
    candidates = []
    for old in chosen_candidates:
        checkpoint = source / old["checkpoint"]
        require(sha(checkpoint) == old["checkpoint_sha256"], "Input changed during evaluation")
        model = GridDoubleDQN.load(checkpoint, device="cpu")
        summary = evaluate(model, spec["config"], condition, new_manifest["validation_layouts"],
                           output / "validation" / f"step_{old['steps']:06d}", old["checkpoint_sha256"])
        del model
        candidates.append(dict(steps=old["steps"], updates=old["updates"], checkpoint=old["checkpoint"],
                               checkpoint_sha256=old["checkpoint_sha256"], **summary))
        print(f"{condition}/{seed} {old['steps']}: validation mean={summary['mean_return']:.6f}", flush=True)
    best = select_best(candidates)
    selection = dict(metric=new_manifest["selection_metric"], tie_break="earliest_step",
                     manifest_sha256=sha(output / "manifest.json"), candidates=candidates, selected=best)
    # Lock selection BEFORE reading/evaluating audit or development outcomes.
    write_json(output / "selection.json", selection)
    summaries = {}
    for role, c in (("best", best), ("final", candidates[-1])):
        dest = output / f"{role}.zip"
        shutil.copyfile(source / c["checkpoint"], dest)
        require(sha(dest) == c["checkpoint_sha256"], "Copied model hash mismatch")
        model = GridDoubleDQN.load(dest, device="cpu")
        for split, scenes in (("audit", new_manifest["audit_layouts"]), ("development", new_manifest["development_layouts"])):
            summaries[f"{split}_{role}"] = evaluate(model, spec["config"], condition, scenes,
                output / f"{split}_{role}", c["checkpoint_sha256"], events=True)
        del model
    for name, digest in hashes.items():
        require(sha(source / name) == digest, "Source modified during run")
    metadata = dict(version="v0.7.3", phase=new_manifest["phase"], status="complete",
                    condition=condition, seed=seed, training_performed=False,
                    original_training_steps=600000, best_steps=best["steps"],
                    original_best_steps=provenance["original_best_steps"],
                    best_sha256=sha(output / "best.zip"), final_sha256=sha(output / "final.zip"),
                    manifest_sha256=sha(output / "manifest.json"),
                    provenance_sha256=sha(output / "provenance.json"), selection_sha256=sha(output / "selection.json"),
                    torch_version=torch.__version__, summaries=summaries)
    write_json(output / "metadata.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.condition, args.seed, args.source or SOURCE / args.condition / str(args.seed),
        args.output or ROOT / args.condition / str(args.seed), args.smoke)
