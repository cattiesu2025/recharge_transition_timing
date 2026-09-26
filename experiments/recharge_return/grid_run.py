"""Train/evaluate separate grid pilot, preserving all declared seeds and scenes."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import stable_baselines3
import torch
import yaml

from .agent import DoubleDQN, MetricsCallback
from .grid_env import GridRechargeEnv, GridTrainingEnv
from scripts.audit_grid_routes import CONFIG, detect
from scripts.sweep_grid_routes import BATTERIES, CONDITIONS, layouts

SEEDS=(4100,4101,4102)
TRAIN_STEPS=600000
ROOT=Path("outputs/recharge_return_v0.7.0_grid_pilot")


class GridDoubleDQN(DoubleDQN):
    mask_slice=slice(-5,None)
    action_count=5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specification():
    return dict(version="v0.7.0",phase="development_pilot",config=yaml.safe_load(CONFIG.read_text()),
                layouts=layouts(),batteries=BATTERIES,seeds=SEEDS,train_steps=TRAIN_STEPS,
                action_order=["UP","DOWN","LEFT","RIGHT","WORK"],observation_size=109,
                source_hashes={p:sha(p) for p in [str(CONFIG),
                    "experiments/recharge_return/grid_env.py","experiments/recharge_return/grid_run.py",
                    "experiments/recharge_return/agent.py","scripts/audit_grid_routes.py",
                    "scripts/sweep_grid_routes.py","experiments/recharge_return/grid_sweep_plan.md"]})


def evaluate(model,config,condition,output,checkpoint_hash):
    env=GridRechargeEnv(config,condition)
    rows=[]
    with gzip.open(output/"trajectories.jsonl.gz","wt") as handle:
        for name,tasks in layouts().items():
            for battery in BATTERIES:
                obs,_=env.reset(options=dict(tasks=tasks,battery=battery))
                records=[]
                while not env.done:
                    action,_=model.predict(obs,deterministic=True)
                    obs,_,_,_,record=env.step(int(action))
                    records.append(record)
                events={str(s):detect(records,s) for s in (4,5)}
                row=dict(layout=name,initial_battery=battery,condition=condition,
                    checkpoint_sha256=checkpoint_hash,work=env.work,remaining_battery=env.battery,
                    outcome=records[-1]["outcome"],steps=env.steps,events=events,
                    last_work=max((r["step"] for r in records if r["work"]),default=None),
                    discounted_return=sum(config["agent"]["gamma"]**i*r["reward"] for i,r in enumerate(records)),
                    invalid_actions=sum(not r["mask_before"][r["action_id"]] for r in records))
                rows.append(row)
                handle.write(json.dumps(dict(layout=name,battery=battery,records=records))+"\n")
    env.close()
    (output/"episodes.json").write_text(json.dumps(rows,indent=2)+"\n")
    return rows


def train_eval(condition,seed,output,steps,smoke):
    if seed not in SEEDS or condition not in CONDITIONS or steps<=0:
        raise ValueError("Undeclared seed/condition/budget")
    if steps!=TRAIN_STEPS and not smoke:
        raise ValueError("Nonfinal budget requires --smoke")
    if smoke and "smoke" not in str(output):
        raise ValueError("Smoke output path must contain smoke")
    output.mkdir(parents=True,exist_ok=False)
    spec=specification()
    (output/"specification.json").write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n")
    cfg=spec["config"]
    a=cfg["agent"]
    torch.set_num_threads(1)
    env=GridTrainingEnv(GridRechargeEnv(cfg,condition),seed)
    model=GridDoubleDQN("MlpPolicy",env,learning_rate=a["learning_rate"],
        buffer_size=a["replay_capacity"],learning_starts=a["learning_starts"],batch_size=a["batch_size"],
        gamma=a["gamma"],train_freq=a["train_frequency"],gradient_steps=a["gradient_steps"],
        replay_buffer_kwargs={"handle_timeout_termination":False},target_update_interval=a["target_update_interval"],
        exploration_fraction=min(1.,a["epsilon_decay_steps"]/steps),exploration_initial_eps=a["epsilon_start"],
        exploration_final_eps=a["epsilon_end"],max_grad_norm=a["gradient_clip"],
        policy_kwargs={"net_arch":a["hidden_sizes"]},seed=seed,device="cpu",verbose=0)
    callback=MetricsCallback(output/"training_metrics.jsonl")
    model.learn(total_timesteps=steps,callback=callback,log_interval=None)
    model.save(output/"final.zip")
    metadata=dict(condition=condition,seed=seed,training_steps=model.num_timesteps,
                  checkpoint_role="smoke" if smoke else "final_fixed_budget",episodes=callback.completed,
                  checkpoint_sha256=sha(output/"final.zip"),specification_sha256=sha(output/"specification.json"),
                  sb3_version=stable_baselines3.__version__,torch_version=torch.__version__)
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    model.get_env().close()
    del model
    reloaded=GridDoubleDQN.load(output/"final.zip",device="cpu")
    rows=evaluate(reloaded,cfg,condition,output,metadata["checkpoint_sha256"])
    print(condition,seed,metadata["checkpoint_role"],steps,dict(Counter(r["outcome"] for r in rows)),flush=True)


def aggregate(root):
    all_rows=[]
    specs=set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            folder=root/condition/str(seed)
            meta=json.loads((folder/"metadata.json").read_text())
            assert meta["condition"]==condition and meta["seed"]==seed
            assert meta["checkpoint_role"]=="final_fixed_budget" and meta["training_steps"]==TRAIN_STEPS
            assert sha(folder/"final.zip")==meta["checkpoint_sha256"]
            assert sha(folder/"specification.json")==meta["specification_sha256"]
            specs.add(meta["specification_sha256"])
            rows=json.loads((folder/"episodes.json").read_text())
            assert len(rows)==len(layouts())*len(BATTERIES)
            assert {(r["layout"],r["initial_battery"]) for r in rows}=={(l,b) for l in layouts() for b in BATTERIES}
            assert all(r["condition"]==condition and r["checkpoint_sha256"]==meta["checkpoint_sha256"] and r["invalid_actions"]==0 for r in rows)
            all_rows.extend(dict(seed=seed,**r) for r in rows)
    assert len(specs)==1,"Mixed pilot specifications"
    summary=[]
    for layout in layouts():
        for battery in BATTERIES:
            group=[r for r in all_rows if r["layout"]==layout and r["initial_battery"]==battery]
            stats={c:dict(outcomes=dict(Counter(r["outcome"] for r in group if r["condition"]==c)),
                          work_mean=statistics.mean(r["work"] for r in group if r["condition"]==c),
                          event_count=sum(r["events"]["4"]["onset"] is not None for r in group if r["condition"]==c)) for c in CONDITIONS}
            contrasts=[]
            for seed in SEEDS:
                pair={r["condition"]:r for r in group if r["seed"]==seed}
                a=pair["RES"]["events"]["4"]["onset"]
                b=pair["PROD"]["events"]["4"]["onset"]
                contrasts.append(dict(seed=seed,prod_minus_res=None if a is None or b is None else b-a))
            summary.append(dict(layout=layout,battery=battery,statistics=stats,contrasts=contrasts))
    result=dict(kind="development_pilot_no_confirmatory_inference",rows=len(all_rows),summary=summary)
    with (root/"pilot_summary.json").open("x") as handle:
        json.dump(result,handle,indent=2)
    print(root/"pilot_summary.json")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition",choices=CONDITIONS)
    p.add_argument("--seed",type=int,choices=SEEDS)
    p.add_argument("--steps",type=int,default=TRAIN_STEPS)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--output",type=Path)
    p.add_argument("--aggregate",type=Path)
    args=p.parse_args()
    if args.aggregate:
        aggregate(args.aggregate)
    elif args.condition is None or args.seed is None:
        p.error("--condition and --seed are required")
    else:
        dest=args.output or (ROOT/args.condition/str(args.seed))
        train_eval(args.condition,args.seed,dest,args.steps,args.smoke)
