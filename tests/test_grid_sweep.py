import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import yaml

from scripts.audit_grid_routes import CONFIG, route, simulate
from scripts.sweep_grid_routes import layouts, generated_tasks, separation
from experiments.recharge_return.grid_env import GridRechargeEnv, GridTrainingEnv, DELTAS
from experiments.recharge_return.grid_run import GridDoubleDQN


def test_layout_generation_and_density():
    ls=layouts()
    assert len(ls)==9
    for name,tasks in ls.items():
        assert len(set(tasks.values()))==len(tasks)
        assert all(not(x>=5 and y>=5) for x,y in tasks.values())
        if name.startswith("aligned"):
            points=sorted(tasks.values())
            assert all(a[1]<=b[1] for a,b in zip(points,points[1:]))
        if name.startswith("detour"):
            assert sum(x<4 for x,y in tasks.values())==len(tasks)//2
    assert generated_tasks("detour",6,9101)==generated_tasks("detour",6,9101)


def test_tie_aware_separation():
    assert separation([20,21],[34,35])=="positive"
    assert separation([20,21],[21,22])=="overlap"
    assert separation([20,None],[35])=="missing_event"
    assert separation([30],[20])=="negative"


@pytest.mark.parametrize("condition",["RES","BAL","PROD"])
def test_grid_simulator_matches_route_reference(condition):
    cfg=yaml.safe_load(CONFIG.read_text())
    env=GridRechargeEnv(cfg,condition)
    for tasks in layouts().values():
        for battery in (24,40,60):
            actions=route(tuple(tasks), (1,0), tasks)
            reference=simulate(actions,battery,cfg,tasks)
            obs,_=env.reset(options=dict(tasks=tasks,battery=battery))
            score=0.
            for i,(action,target) in enumerate(actions):
                aid=4 if action=="WORK" else DELTAS.index((target[0]-env.pos[0],target[1]-env.pos[1]))
                assert obs[-5+aid]==1
                obs,reward,term,trunc,info=env.step(aid)
                score+=cfg["agent"]["gamma"]**i*reward
                assert env.observation_space.contains(obs)
                ref=reference["records"][i]
                for key in ("position","battery","work","deficit","common","outcome"):
                    assert info[key]==ref[key]
                if term or trunc: break
            assert score==pytest.approx(reference["scores"][condition])
            assert env.work==reference["work"]
    env.close()


def test_matching_training_scenarios_and_grid_masks():
    cfg=yaml.safe_load(CONFIG.read_text())
    a=GridTrainingEnv(GridRechargeEnv(cfg,"RES"),4100)
    b=GridTrainingEnv(GridRechargeEnv(cfg,"PROD"),4100)
    for _ in range(4):
        obs,_=a.reset(); b.reset()
        assert a.current==b.current
        assert list(obs[-5:])==[0,1,0,1,0]
        assert all(x in (1,3) for x in GridDoubleDQN._random_valid_actions(np.tile(obs,(100,1))))
        with pytest.raises(ValueError): a.step(0)
    a.close(); b.close()


def test_grid_array_mapping():
    import os
    script="scripts/katana_recharge_grid_v0.7.pbs"
    for i in range(1,10):
        env=dict(os.environ,PBS_O_WORKDIR=str(Path.cwd()),PBS_ARRAY_INDEX=str(i),RECHARGE_VALIDATE_MAPPING_ONLY="1")
        result=subprocess.run(["bash",script],env=env,capture_output=True,text=True,check=True)
        c=("RES","BAL","PROD")[(i-1)//3]; seed=4100+(i-1)%3
        assert f"condition={c} seed={seed} steps=600000" in result.stdout


def test_saved_sweep_reproduces_v062_and_has_all_cells():
    root=Path("outputs/recharge_return_v0.7.0_sweep")
    prior=Path("outputs/recharge_return_v0.6.2_grid_audit/summary.json")
    if not root.exists() or not prior.exists():
        pytest.skip("Generated local audits not present")
    rows=json.loads((root/"summary.json").read_text())["rows"]
    assert len(rows)==333
    assert len({(r["layout"],r["battery"]) for r in rows})==333
    old=json.loads(prior.read_text())
    for r in old["summaries"]:
        new=next(x for x in rows if x["layout"]=="original_6" and x["battery"]==r["battery"])
        for c in ("RES","BAL","PROD"):
            for key in ("score","work","battery","onset","outcomes"):
                assert new["best_all"][c][key]==r["best_all"][c][key]
