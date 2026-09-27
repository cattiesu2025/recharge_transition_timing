import json
import os
import random
import subprocess

import numpy as np
import pytest
import torch

from experiments.recharge_return import grid_selection as gs


def test_validation_split_and_selection():
    assert len(gs.validation_layouts()) == 8
    assert len(gs.BATTERIES) == 37
    a = dict(steps=100000, mean_return=2, onset=100)
    b = dict(steps=50000, mean_return=2, onset=0)
    assert gs.select_best([a, b]) is b
    assert gs.select_best([a, dict(b, mean_return=-4)]) is a
    with pytest.raises(ValueError):
        gs.select_best([dict(a, mean_return=float("nan"))])


def test_rng_restored_even_on_error():
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    expected = (random.random(), np.random.random(), torch.rand(1))
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    with pytest.raises(RuntimeError), gs.preserve_rng():
        random.random(), np.random.random(), torch.rand(1)
        raise RuntimeError()
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(1), expected[2])


def test_loop_erasure():
    points = [(0, 0), (1, 0), (2, 0), (1, 0), (0, 0), (0, 1)]
    records = [dict(before=a, position=b, work=0) for a, b in zip(points, points[1:])]
    assert gs.loop_moves(records) == 4
    records.insert(2, dict(before=(2, 0), position=(2, 0), work=1))
    assert gs.loop_moves(records) == 0


def test_budget_guards(tmp_path):
    for steps, interval, smoke in [(6000, 3000, False), (6001, 3000, True), (6000, 3000, True)]:
        with pytest.raises(ValueError):
            gs.train_eval("RES", 4100, tmp_path / "run", steps, interval, smoke)
    assert not (tmp_path / "run").exists()
    existing = tmp_path / "smoke_existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        gs.train_eval("RES", 4100, existing, 6000, 3000, True)
    assert list(existing.iterdir()) == []


def test_checkpoints_do_not_change_training(tmp_path, monkeypatch):
    # Actual evaluator/model, small scene set and short learning warmup.
    monkeypatch.setattr(gs, "BATTERIES", [24])
    monkeypatch.setattr(gs, "validation_layouts", lambda: {"one": gs.layouts()["original_6"]})
    cfg = gs.specification()["config"]
    cfg["agent"]["learning_starts"] = 4
    torch.set_num_threads(1)
    reference = gs.make_model(cfg, "RES", 4100, 24)
    reference.learn(24, log_interval=None)
    weights = {k: v.clone() for k, v in reference.policy.state_dict().items()}
    replay_size = reference.replay_buffer.size()
    observations = reference.replay_buffer.observations[:replay_size].copy()
    reference.get_env().close()
    model = gs.make_model(cfg, "RES", 4100, 24)
    callback = gs.SelectionCallback(tmp_path, cfg, "RES", 8, 24)
    model.learn(24, callback=callback, log_interval=None)
    assert [c["steps"] for c in callback.candidates] == [8, 16, 24]
    assert [c["updates"] for c in callback.candidates] == [1, 3, 5]
    for k, v in model.policy.state_dict().items():
        assert torch.equal(v, weights[k])
    np.testing.assert_array_equal(model.replay_buffer.observations[:replay_size], observations)
    for c in callback.candidates:
        assert gs.sha(tmp_path / c["checkpoint"]) == c["checkpoint_sha256"]
    loaded = gs.GridDoubleDQN.load(tmp_path / callback.candidates[-1]["checkpoint"])
    assert loaded._n_updates == model._n_updates
    for k, v in loaded.policy.state_dict().items():
        assert torch.equal(v, weights[k])
    rows = json.loads((tmp_path / "validation/step_000024/episodes.json").read_text())
    assert len(rows) == 1 and "final_entry" not in rows[0]
    model.get_env().close()


def test_pbs_mapping():
    pairs = set()
    for i in range(1, 10):
        env = dict(os.environ, PBS_ARRAY_INDEX=str(i), PBS_O_WORKDIR=os.getcwd(), RECHARGE_VALIDATE_MAPPING_ONLY="1")
        result = subprocess.run(["bash", "scripts/katana_recharge_grid_v0.7.2.pbs"], env=env,
                                capture_output=True, text=True, check=True)
        pairs.add(tuple(result.stdout.split()[:2]))
    assert len(pairs) == 9
