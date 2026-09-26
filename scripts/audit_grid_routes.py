"""Finite route-class audit; not a global planner or learned-policy evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import permutations
import json
from pathlib import Path

import yaml

TASKS = {"A": (1, 2), "B": (3, 1), "C": (2, 5),
         "D": (5, 3), "E": (4, 8), "F": (8, 4)}
DOCK = (9, 9)
BATTERIES = (24, 32, 40, 60)
CONFIG = Path("experiments/recharge_return/configs/pilot_line_masked_600k.yaml")


def distance(p):
    return abs(9 - p[0]) + abs(9 - p[1])


def route(order, axes, tasks=None):
    """Each item is (action, destination); WORK is explicitly selected."""
    tasks = TASKS if tasks is None else tasks
    pos = [0, 0]
    actions = []
    for name in (*order, "dock"):
        target = DOCK if name == "dock" else tasks[name]
        for axis in axes:
            while pos[axis] != target[axis]:
                pos[axis] += 1 if target[axis] > pos[axis] else -1
                actions.append(("MOVE", tuple(pos)))
        if name != "dock":
            actions.append(("WORK", tuple(pos)))
    return actions


def detect(records, side):
    inside = lambda p: p[0] >= 10-side and p[1] >= 10-side
    candidates = []
    onset = None
    confirmation = None
    final_entry = None
    for i, row in enumerate(records):
        if inside(row["before"]) or not inside(row["position"]):
            continue
        if all(inside(r["position"]) for r in records[i:]):
            final_entry = row["step"]
        progress = 0
        confirmed = None
        for r in records[i:i+8]:
            if (not inside(r["position"]) or r["action"] == "WORK"
                    or distance(r["position"]) > distance(r["before"])
                    or r["outcome"] == "exhausted"):
                break
            progress += distance(r["position"]) < distance(r["before"])
            if progress >= 3:
                confirmed = r["step"]
                break
            if r["outcome"] is not None:
                break
        candidates.append({"step": row["step"], "confirmation": confirmed})
        if onset is None and confirmed is not None:
            onset, confirmation = row["step"], confirmed
    return {"onset": onset, "confirmation": confirmation,
            "final_entry": final_entry, "candidates": candidates}


def simulate(actions, battery, config, tasks=None):
    tasks = TASKS if tasks is None else tasks
    pos = (0, 0)
    work = 0
    completed = set()
    records = []
    terms = dict(production=0., reserve=0., common=0.)
    gamma = config["agent"]["gamma"]
    reward = config["reward"]
    env = config["environment"]
    for index, (action, target) in enumerate(actions):
        if not (0 <= target[0] < 10 and 0 <= target[1] < 10):
            raise ValueError("Out of grid")
        if action == "WORK":
            if target != pos or pos not in tasks.values() or pos in completed:
                raise ValueError("Invalid WORK")
            cost = env["costs"]["work"]
        elif action == "MOVE" and sum(abs(a-b) for a,b in zip(target,pos)) == 1:
            cost = env["costs"]["move_right"]
        else:
            raise ValueError("Invalid movement")
        before = pos
        affordable = battery >= cost
        unit = int(action == "WORK" and affordable)
        if affordable:
            pos = target
            if unit:
                completed.add(pos)
        work += unit
        battery = max(0., battery-cost)
        outcome = ("exhausted" if battery <= 0 else
                   ("returned" if work else "returned_without_work") if pos == DOCK else
                   "time_limit" if index+1 >= env["max_steps"] else None)
        deficit = max(0., min(1., (reward["safe_margin"] - (battery-distance(pos))) / env["capacity"]))
        common = -reward["time_cost"]
        if outcome == "exhausted":
            common -= reward["exhaustion_penalty"]
        elif outcome == "returned":
            common += reward["return_bonus"]
        elif outcome == "returned_without_work":
            common -= reward["empty_return_penalty"]
        factor = gamma**index
        terms["production"] += factor*unit
        terms["reserve"] += factor*deficit
        terms["common"] += factor*common
        records.append(dict(step=index+1, action=action, before=before, position=pos,
                            battery=battery, work=unit, deficit=deficit,
                            common=common, outcome=outcome))
        if outcome:
            break
    scores = {c: w["production"]*terms["production"] - w["reserve"]*terms["reserve"] + terms["common"]
              for c,w in reward["conditions"].items()}
    return dict(work=work, steps=len(records), battery=battery,
                outcome=records[-1]["outcome"], terms=terms, scores=scores,
                events={str(s): detect(records,s) for s in (4,5)}, records=records)


def best(rows, condition):
    score = max(r["scores"][condition] for r in rows)
    tied = [r for r in rows if abs(r["scores"][condition]-score) < 1e-10]
    return dict(score=score, count=len(tied), route_ids=[r["id"] for r in tied],
                work=sorted({r["work"] for r in tied}),
                battery=sorted({r["battery"] for r in tied}),
                outcomes=sorted({r["outcome"] for r in tied}),
                onset={str(s): sorted({r["events"][str(s)]["onset"] for r in tied},
                                     key=lambda x: -1 if x is None else x) for s in (4,5)})


def run(output):
    config = yaml.safe_load(CONFIG.read_text())
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    selected = []
    with (output/"routes.jsonl").open("w") as handle:
        for battery in BATTERIES:
            rows = []
            seen = set()
            for n in range(7):
                for order in permutations(TASKS, n):
                    for axes in ((0,1), (1,0)):
                        actions = route(order,axes)
                        # Deduplicate full intended action sequences, not failed prefixes.
                        signature = tuple(actions)
                        if signature in seen:
                            continue
                        seen.add(signature)
                        result = simulate(actions,battery,config)
                        row = dict(id=f"b{battery}_{''.join(order) or 'empty'}_{axes[0]}",
                                   initial_battery=battery, task_order=order, axes=axes, **result)
                        handle.write(json.dumps(row)+"\n")
                        rows.append(row)
            successful = [r for r in rows if r["outcome"] == "returned"]
            winners = {c:best(rows,c) for c in config["reward"]["conditions"]}
            safe = {c:best(successful,c) for c in winners}
            ids = set().union(*(set(w["route_ids"]) for w in winners.values()),
                              *(set(w["route_ids"]) for w in safe.values()))
            selected.extend(r for r in rows if r["id"] in ids)
            points = sorted({(r["work"],r["battery"]) for r in successful})
            frontier = [p for p in points if not any(q[0]>=p[0] and q[1]>=p[1] and q!=p for q in points)]
            summaries.append(dict(battery=battery, route_count=len(rows),
                                  outcomes=dict(Counter(r["outcome"] for r in rows)),
                                  best_all=winners, best_successful=safe,
                                  successful_work_battery_frontier=frontier))
    summary = dict(kind="finite_route_design_audit_not_learned_policy", version="v0.6.2",
                   scope="ordered task subsets; two global axis-order shortest-path rules; not global optimum",
                   config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                   config=config, tasks=TASKS, dock=DOCK, grid_size=10,
                   batteries=BATTERIES, summaries=summaries, selected_routes=selected)
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines = ["# 10×10 多任务路线验证 — v0.6.2", "",
             "有限路线集合的计算检查；非全局最优、非学习策略证据。完整设计见 experiments/recharge_return/grid_route_audit.md。",
             "", "| 电量 | 条件 | 最佳成功路线任务数 | 到站电量 | 4×4 ONSET集合 | 5×5 ONSET集合 | 折扣回报 |",
             "|---|---|---|---|---|---|---|"]
    for s in summaries:
        for c,w in s["best_successful"].items():
            lines.append(f"| {s['battery']} | {c} | {w['work']} | {w['battery']} | {w['onset']['4']} | {w['onset']['5']} | {w['score']:.6f} |")
    lines += ["", "不同条件的回报权重不同，表中回报不能跨条件衡量策略优劣。",
              "并列最佳全部保留；ONSET集合反映路径歧义，不任意挑一条制造排序。",
              "所有失败和空返路线亦已计分；summary.json 中 best_all 与 best_successful 分列。",
              "电量、区域和任务位置事先固定；没有搜索能产生预期排序的权重或位置。",
              "只枚举每个任务最多工作一次、两种全程轴顺序；未穷举所有原子路径，因此未分离不能证明所有策略相同。",
              "", "## 全部路线终局与产出—电量前沿", ""]
    for s in summaries:
        lines.append(f"- 电量{s['battery']}：{s['route_count']}条；{s['outcomes']}；成功路线前沿{ s['successful_work_battery_frontier']}。")
    (output/"report.md").write_text("\n".join(lines)+"\n")
    print(output/"report.md")
    for s in summaries:
        print(s["battery"],json.dumps(s["best_successful"],ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/recharge_return_v0.6.2_grid_audit"))
    run(parser.parse_args().output)
