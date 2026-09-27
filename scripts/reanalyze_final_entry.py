"""Post-hoc final-entry event reanalysis; leaves original evaluations intact."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

from scripts.audit_grid_routes import detect, distance


def final_entry(records, side=4, net_progress=2):
    if side not in (4,5) or net_progress not in (1,2,3):
        raise ValueError("Unsupported candidate parameter")
    if not records or records[-1]["outcome"] is None:
        raise ValueError("Complete terminated or truncated trajectory required")
    inside=lambda p: p[0]>=10-side and p[1]>=10-side
    result=dict(onset=None,entry=None,progress_met_step=None,reason="no_final_entry")
    if not inside(records[-1]["position"]):
        return result
    entries=[i for i,r in enumerate(records) if not inside(r["before"]) and inside(r["position"])]
    if not entries:
        return result
    i=entries[-1]
    assert all(inside(r["position"]) for r in records[i:])
    result.update(entry=records[i]["step"],reason="insufficient_net_progress")
    base=distance(records[i]["position"])
    for r in records[i:]:
        if r["battery"]>0 and base-distance(r["position"])>=net_progress:
            result.update(onset=records[i]["step"],progress_met_step=r["step"],reason="observed")
            break
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contrasts(rows, field):
    out=[]
    for seed in (4100,4101,4102):
        for family in ("all","original","aligned","detour"):
            rr=[r for r in rows if r["seed"]==seed and (family=="all" or r["layout"].startswith(family))]
            d={(r["condition"],r["layout"],r["battery"]):r for r in rr}
            keys=sorted({(r["layout"],r["battery"]) for r in rr})
            diffs=[]; success=[]
            for l,b in keys:
                a,z=d["RES",l,b],d["PROD",l,b]
                x,y=a[field]["onset"],z[field]["onset"]
                if x is not None and y is not None:
                    diffs.append(y-x)
                    if a["outcome"]==z["outcome"]=="returned": success.append(y-x)
            out.append(dict(seed=seed,family=family,total=len(keys),joint=len(diffs),
                median=median(diffs) if diffs else None,
                signs=dict(Counter("positive" if x>0 else "negative" if x<0 else "tie" for x in diffs)),
                successful_joint=len(success),successful_median=median(success) if success else None))
    return out


def run(root,output):
    if output.exists(): raise FileExistsError(output)
    rows=[]; hashes={}; specifications=set()
    for c in ("RES","BAL","PROD"):
        for seed in (4100,4101,4102):
            d=root/c/str(seed)
            meta=json.loads((d/"metadata.json").read_text())
            spec=json.loads((d/"specification.json").read_text())
            assert (meta["condition"],meta["seed"],meta["training_steps"],meta["checkpoint_role"])==(c,seed,600000,"final_fixed_budget")
            assert digest(d/"final.zip")==meta["checkpoint_sha256"]
            assert digest(d/"specification.json")==meta["specification_sha256"]
            specifications.add(meta["specification_sha256"])
            for f in ("metadata.json","specification.json","final.zip","episodes.json","trajectories.jsonl.gz"):
                hashes[str(d/f)]=digest(d/f)
            ep=json.loads((d/"episodes.json").read_text())
            expected={(l,b) for l in spec["layouts"] for b in spec["batteries"]}
            lookup={(r["layout"],r["initial_battery"]):r for r in ep}
            assert len(ep)==333 and set(lookup)==expected
            seen=set()
            with gzip.open(d/"trajectories.jsonl.gz","rt") as handle:
                for line in handle:
                    t=json.loads(line); key=(t["layout"],t["battery"])
                    assert key not in seen; seen.add(key)
                    e=lookup[key]; rec=t["records"]
                    assert e["condition"]==c and e["checkpoint_sha256"]==meta["checkpoint_sha256"]
                    assert [r["step"] for r in rec]==list(range(1,len(rec)+1))
                    assert rec[-1]["outcome"]==e["outcome"] and all(r["outcome"] is None for r in rec[:-1])
                    assert sum(r["work"] for r in rec)==e["work"]
                    assert {str(s):detect(rec,s) for s in (4,5)}==e["events"]
                    cf=e["events"]["4"]["confirmation"]
                    oldexit=cf is not None and any(r["step"]>cf and (r["position"][0]<6 or r["position"][1]<6) for r in rec)
                    oldwork=cf is not None and any(r["step"]>cf and r["work"] for r in rec)
                    row=dict(condition=c,seed=seed,layout=key[0],battery=key[1],outcome=e["outcome"],work=e["work"],
                             old=e["events"]["4"],old_exit=oldexit,old_later_work=oldwork)
                    for side in (4,5):
                        for k in (1,2,3): row[f"s{side}_k{k}"]=final_entry(rec,side,k)
                    rows.append(row)
            assert seen==expected
    assert len(specifications)==1 and len(rows)==2997
    counts={}
    for c in ("RES","BAL","PROD"):
        g=[r for r in rows if r["condition"]==c]
        counts[c]=dict(total=len(g),old_events=sum(r["old"]["onset"] is not None for r in g),
            new_events=sum(r["s4_k2"]["onset"] is not None for r in g),
            recovered=sum(r["old"]["onset"] is None and r["s4_k2"]["onset"] is not None for r in g),
            lost=sum(r["old"]["onset"] is not None and r["s4_k2"]["onset"] is None for r in g),
            changed_time=sum(r["old"]["onset"] is not None and r["s4_k2"]["onset"] is not None and r["old"]["onset"]!=r["s4_k2"]["onset"] for r in g),
            by_outcome={o:dict(total=sum(r["outcome"]==o for r in g),observed=sum(r["outcome"]==o and r["s4_k2"]["onset"] is not None for r in g)) for o in sorted({r["outcome"] for r in g})})
    cs={f:contrasts(rows,f) for f in ("old","s4_k1","s4_k2","s4_k3","s5_k1","s5_k2","s5_k3")}
    result=dict(kind="posthoc_pilot_final_entry_reanalysis",version="v0.7.1",input_sha256=hashes,
                primary="s4_k2",counts=counts,contrasts=cs,rows=rows)
    output.mkdir(parents=True)
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# 最终进入候选定义重算（探索性）","","v0.7.0已学习pilot的离线重算，旧定义与输入不变。主定义4×4、进入后净推进2格、之后不再离区。",
           "","|条件|旧事件|新事件|恢复|丢失|已有事件改时|","|---|---:|---:|---:|---:|---:|"]
    for c,v in counts.items(): lines.append(f"|{c}|{v['old_events']}|{v['new_events']}|{v['recovered']}|{v['lost']}|{v['changed_time']}|")
    lines += ["","|定义|seed|布局族|共同事件/分母|PROD−RES中位步差|双方有工作到站共同事件|该子集中位差|","|---|---|---|---|---|---|---|"]
    for f,items in cs.items():
        for r in items:
            lines.append(f"|{f}|{r['seed']}|{r['family']}|{r['joint']}/{r['total']}|{r['median']}|{r['successful_joint']}|{r['successful_median']}|")
    lines += ["","旧确认后退出的全部案例：","","|条件/seed|布局/电量|旧onset|新onset|确认后再工作|终局|","|---|---|---|---|---|---|"]
    for r in rows:
        if r["old_exit"]:
            lines.append(f"|{r['condition']}/{r['seed']}|{r['layout']}/{r['battery']}|{r['old']['onset']}|{r['s4_k2']['onset']}|{r['old_later_work']}|{r['outcome']}|")
    lines += ["","无事件不赋无限时间；零工作到站与耗尽独立保留。此定义在观察结果后提出，不用于确认性推断；回溯边界不能代表在线意图。"]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(counts,indent=2))
    print(json.dumps({f:[r for r in v if r['family']=='all'] for f,v in cs.items()},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=Path("outputs/recharge_return_v0.7.0_grid_pilot"))
    p.add_argument("--output",type=Path,default=Path("outputs/recharge_return_v0.7.1_final_entry"))
    a=p.parse_args(); run(a.input,a.output)
