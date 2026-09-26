"""Predeclared battery/layout sweep of a finite route class."""
import argparse
from collections import Counter
import hashlib
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.audit_grid_routes import CONFIG, TASKS, best, route, simulate

BATTERIES = list(range(24, 61))
CONDITIONS = ("RES", "BAL", "PROD")


def generated_tasks(kind, count, seed):
    rng = np.random.default_rng(seed)
    if kind == "aligned":
        pool = []
        for k in range(4):
            pool.extend([(k+1,k), (k+1,k+1)])
        pool.extend((x,4) for x in range(5,10))
        points = [pool[i] for i in rng.choice(len(pool), count, replace=False)]
    elif kind == "detour":
        pools = [[(x,y) for x in range(4) for y in range(6,10)],
                 [(x,y) for x in range(6,10) for y in range(4)]]
        points = [pool[i] for pool in pools for i in rng.choice(len(pool),count//2,replace=False)]
    else:
        raise ValueError(kind)
    return {chr(65+i): p for i,p in enumerate(sorted(points))}


def layouts():
    result = {"original_6": TASKS}
    for kind in ("aligned", "detour"):
        for count in (4,6):
            for seed in (9101,9102):
                result[f"{kind}_{count}_{seed}"] = generated_tasks(kind,count,seed)
    return result


def separation(a, b):
    if not a or not b or None in a or None in b:
        return "missing_event"
    if min(b) > max(a):
        return "positive"
    if max(b) < min(a):
        return "negative"
    return "overlap"


def enhanced_best(rows, condition):
    w = best(rows,condition)
    lower = [r["scores"][condition] for r in rows if r["scores"][condition] < w["score"]-1e-10]
    w["strict_runner_up_gap"] = w["score"]-max(lower) if lower else None
    different = [r["scores"][condition] for r in rows if r["work"] not in w["work"]]
    w["different_work_gap"] = w["score"]-max(different) if different else None
    ids = set(w["route_ids"])
    w["representatives"] = [dict(id=r["id"], work=r["work"], battery=r["battery"],
                                  terms=r["terms"], last_work=r["last_work"], events=r["events"])
                              for r in rows if r["id"] in ids]
    return w


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    config = yaml.safe_load(CONFIG.read_text())
    manifest = dict(version="v0.7.0", batteries=BATTERIES, layouts=layouts(),
                    config=config, config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest())
    content = json.dumps(manifest,indent=2,sort_keys=True)+"\n"
    (output/"manifest.json").write_text(content)
    manifest_hash = hashlib.sha256(content.encode()).hexdigest()
    summaries = []
    for layout, tasks in layouts().items():
        plans = []
        seen = set()
        for n in range(len(tasks)+1):
            for order in permutations(tasks,n):
                for axes in ((0,1),(1,0)):
                    actions = route(order,axes,tasks)
                    if tuple(actions) in seen:
                        continue
                    seen.add(tuple(actions))
                    plans.append(dict(id=f"{''.join(order) or 'empty'}_{axes[0]}",
                                      order=order, axes=axes, actions=actions))
        (output/f"{layout}_plans.json").write_text(json.dumps(plans)+"\n")
        # Compact full enumeration: one row per battery/route; -1 means absent event.
        names = ["battery", "route_index", "work", "remaining", "outcome", "steps", "last_work",
                 "production", "reserve", "common", "RES", "BAL", "PROD", "onset4", "onset5"]
        matrix = []
        for battery in BATTERIES:
            rows = []
            for i,plan in enumerate(plans):
                r = simulate(plan["actions"],battery,config,tasks)
                last = max((x["step"] for x in r["records"] if x["work"]),default=None)
                r.pop("records")
                r.update(id=plan["id"],last_work=last)
                rows.append(r)
                outcome = {"returned":1,"returned_without_work":0,"exhausted":-1,"time_limit":-2}[r["outcome"]]
                matrix.append([battery,i,r["work"],r["battery"],outcome,r["steps"],last or -1,
                               *[r["terms"][k] for k in ("production","reserve","common")],
                               *[r["scores"][c] for c in CONDITIONS],
                               *[r["events"][str(s)]["onset"] or -1 for s in (4,5)]])
            win = {c:enhanced_best(rows,c) for c in CONDITIONS}
            safe = [r for r in rows if r["outcome"]=="returned"]
            safe_win = {c:enhanced_best(safe,c) for c in CONDITIONS} if safe else None
            contrasts = {str(side): {f"{b}-{a}": separation(win[a]["onset"][str(side)],win[b]["onset"][str(side)])
                         for a,b in (("RES","PROD"),("RES","BAL"),("BAL","PROD"))} for side in (4,5)}
            summaries.append(dict(layout=layout,battery=battery,route_count=len(rows),
                                  outcomes=dict(Counter(r["outcome"] for r in rows)),
                                  best_all=win,best_successful=safe_win,contrasts=contrasts))
        np.savez_compressed(output/f"{layout}_routes.npz",columns=np.array(names),data=np.array(matrix))
        (output/f"{layout}_summary.json").write_text(json.dumps(summaries[-37:],indent=2)+"\n")
        print(layout,"completed",len(plans)*37,"route/battery evaluations",flush=True)
    result = dict(kind="finite_route_sweep_not_learned_policy",manifest_sha256=manifest_hash,rows=summaries)
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    lines = ["# 多布局连续电量扫描", "", "全部24–60电量保留；有限路线类最佳，非全局最优或学习证据。",
             "严格正分离：PROD所有并列最佳的最早ONSET晚于RES所有并列最佳的最晚ONSET。重叠不代表等价。",
             "", "| 布局 | 正分离电量 | 反向电量 | 重叠数 | 缺事件数 |", "|---|---|---|---|---|"]
    for name in layouts():
        rs = [r for r in summaries if r["layout"]==name]
        counts = Counter(r["contrasts"]["4"]["PROD-RES"] for r in rs)
        pos = [r["battery"] for r in rs if r["contrasts"]["4"]["PROD-RES"]=="positive"]
        neg = [r["battery"] for r in rs if r["contrasts"]["4"]["PROD-RES"]=="negative"]
        lines.append(f"|{name}|{pos}|{neg}|{counts['overlap']}|{counts['missing_event']}|")
    lines += ["", "## 每格最佳集合", "", "| 布局 | 电量 | RES任务/onset | BAL任务/onset | PROD任务/onset | RES严格次优回报差 |", "|---|---|---|---|---|---|"]
    for r in summaries:
        values = [f"{r['best_all'][c]['work']}/{r['best_all'][c]['onset']['4']}" for c in CONDITIONS]
        lines.append(f"|{r['layout']}|{r['battery']}|"+"|".join(values)+f"|{r['best_all']['RES']['strict_runner_up_gap']}|")
    (output/"report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("outputs/recharge_return_v0.7.0_sweep"))
    run(p.parse_args().output)
