import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from experiments.recharge_return import grid_reselect as gr


def test_expanded_layouts():
    m = gr.manifest()
    assert m == gr.manifest()
    seen = {gr.signature(t) for t in [*gr.layouts().values(), *gr.validation_layouts().values()]}
    for split in ("validation", "audit"):
        scenes = m[f"{split}_layouts"]
        assert len(scenes) == 60
        assert len(scenes) * len(m["batteries"]) == 2220
        for kind in gr.KINDS:
            for count in (4, 6):
                subset = [t for name, t in scenes.items() if name.startswith(f"{kind}_{count}_")]
                assert len(subset) == 10
                for tasks in subset:
                    key = gr.signature(tasks)
                    assert len(key) == len(set(key)) == count and key not in seen
                    seen.add(key)
                    assert all(0 <= x < 10 and 0 <= y < 10 and (x, y) != (0, 0)
                               and not (x >= 5 and y >= 5) for x, y in key)
    assert len(gr.manifest(True)["validation_layouts"]) == 6


@pytest.fixture
def source(tmp_path):
    p = tmp_path / "source"
    (p / "checkpoints").mkdir(parents=True)
    spec = dict(version="v0.7.2", train_steps=600000, checkpoint_interval=50000,
                source_hashes={"experiments/recharge_return/grid_env.py": gr.sha("experiments/recharge_return/grid_env.py")}, config={})
    gr.write_json(p / "specification.json", spec)
    candidates = []
    for step in range(50000, 600001, 50000):
        name = f"checkpoints/step_{step:06d}.zip"
        with zipfile.ZipFile(p / name, "w") as z:
            z.writestr("data", json.dumps(dict(num_timesteps=step, seed=4100, _n_updates=(step - 5000) // 4)))
        candidates.append(dict(steps=step, updates=(step - 5000) // 4, checkpoint=name,
                               checkpoint_sha256=gr.sha(p / name), mean_return=float(step)))
    best = candidates[-1]
    gr.write_json(p / "selection.json", dict(metric="mean_undiscounted_episode_return", tie_break="earliest_step",
        candidates=candidates, selected=best, specification_sha256=gr.sha(p / "specification.json")))
    for role in ("best", "final"):
        shutil.copyfile(p / best["checkpoint"], p / f"{role}.zip")
    gr.write_json(p / "metadata.json", dict(version="v0.7.2", phase="development_pilot", status="complete",
        condition="RES", seed=4100, training_steps=600000, updates=148750, best_steps=600000,
        specification_sha256=gr.sha(p / "specification.json"), selection_sha256=gr.sha(p / "selection.json"),
        best_sha256=best["checkpoint_sha256"], final_sha256=best["checkpoint_sha256"]))
    return p


def test_input_guards(source):
    _, candidates, hashes = gr.verify_source(source, "RES", 4100)
    assert len(candidates) == 12 and len(hashes) == 17
    with pytest.raises(ValueError, match="Condition/seed"):
        gr.verify_source(source, "BAL", 4100)
    (source / candidates[0]["checkpoint"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="Checkpoint hash"):
        gr.verify_source(source, "RES", 4100)


def test_lock_selection_before_audit(source, tmp_path, monkeypatch):
    output = tmp_path / "smoke_result"
    calls = []
    monkeypatch.setattr(gr.GridDoubleDQN, "load", lambda *a, **k: object())
    def fake_eval(model, config, condition, scenes, folder, digest, events=False):
        if events:
            assert (output / "selection.json").exists()
        else:
            assert not (output / "selection.json").exists()
        calls.append((events, len(scenes)))
        folder.mkdir(parents=True)
        return dict(mean_return=1.0, episodes=len(scenes)*37)
    monkeypatch.setattr(gr, "evaluate", fake_eval)
    original = {p: gr.sha(p) for p in source.rglob("*") if p.is_file()}
    meta = gr.run("RES", 4100, source, output, smoke=True)
    assert meta["best_steps"] == 50000  # Ties do not look at audit or prior best.
    assert meta["phase"] == "smoke" and meta["status"] == "complete"
    assert calls == [(False, 6), (False, 6), (True, 6), (True, 9), (True, 6), (True, 9)]
    assert all(gr.sha(p) == h for p, h in original.items())
    with pytest.raises(FileExistsError):
        gr.run("RES", 4100, source, output, True)
    with pytest.raises(ValueError, match="disjoint"):
        gr.run("RES", 4100, source, source / "smoke_child", True)
    with pytest.raises(ValueError, match="Smoke"):
        gr.run("RES", 4100, source, tmp_path / "wrong", True)


def test_reselect_pbs():
    pairs = set()
    for i in range(1, 11):
        env = dict(os.environ, PBS_ARRAY_INDEX=str(i), PBS_O_WORKDIR=str(Path.cwd()), RECHARGE_VALIDATE_MAPPING_ONLY="1")
        r = subprocess.run(["bash", "scripts/katana_recharge_grid_v0.7.3.pbs"], env=env, capture_output=True, text=True)
        if i == 10:
            assert r.returncode != 0
        else:
            assert r.returncode == 0
            pairs.add(tuple(r.stdout.split()[:2]))
    assert len(pairs) == 9
