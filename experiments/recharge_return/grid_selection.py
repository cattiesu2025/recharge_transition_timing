"""Development pilot with reward-only validation checkpoint selection."""
import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import random
import shutil
import statistics

import numpy as np
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
import torch

from .agent import MetricsCallback
from .grid_env import GridRechargeEnv, GridTrainingEnv
from .grid_run import GridDoubleDQN, SEEDS, TRAIN_STEPS, sha, specification
from scripts.audit_grid_routes import detect
from scripts.reanalyze_final_entry import final_entry
from scripts.sweep_grid_routes import BATTERIES, CONDITIONS, generated_tasks, layouts

ROOT = Path("outputs/recharge_return_v0.7.2_grid_selection")
INTERVAL = 50000


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def validation_layouts():
    result = {f"{kind}_{count}_{seed}": generated_tasks(kind, count, seed)
              for kind in ("aligned", "detour") for count in (4, 6)
              for seed in (9201, 9202)}
    signature = lambda tasks: tuple(sorted(tuple(p) for p in tasks.values()))
    keys = [signature(t) for t in result.values()]
    if len(set(keys)) != len(keys) or set(keys) & {signature(t) for t in layouts().values()}:
        raise ValueError("Validation layouts overlap; revise protocol before training")
    return result


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def loop_moves(records):
    """Chronological loop erasure between successful WORK anchors."""
    stack = [tuple(records[0]["before"])]
    erased = 0
    for r in records:
        p = tuple(r["position"])
        if r["work"]:
            stack = [p]
        elif p != tuple(r["before"]):
            if p in stack:
                i = stack.index(p)
                erased += len(stack) - i
                stack = stack[:i + 1]
            else:
                stack.append(p)
    return erased


def evaluate(model, config, condition, scenes, output, checkpoint_hash, events=False):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with preserve_rng():
        env = GridRechargeEnv(config, condition)
        try:
            with gzip.open(output / "trajectories.jsonl.gz", "wt") as handle:
                for name, tasks in scenes.items():
                    for battery in BATTERIES:
                        obs, _ = env.reset(seed=0, options=dict(tasks=tasks, battery=battery))
                        records = []
                        while not env.done:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, _, _, record = env.step(int(action))
                            records.append(record)
                        row = dict(layout=name, initial_battery=battery, condition=condition,
                                   checkpoint_sha256=checkpoint_hash, work=env.work,
                                   remaining_battery=env.battery, outcome=records[-1]["outcome"],
                                   steps=env.steps, loop_moves=loop_moves(records),
                                   return_sum=sum(r["reward"] for r in records),
                                   discounted_return=sum(config["agent"]["gamma"]**i * r["reward"]
                                                         for i, r in enumerate(records)),
                                   invalid_actions=sum(not r["mask_before"][r["action_id"]] for r in records))
                        if events:
                            row["final_entry"] = final_entry(records)
                            row["sensitivity"] = {f"s{s}_k{k}": final_entry(records, s, k)
                                                  for s in (4, 5) for k in (1, 2, 3)}
                            row["legacy_events"] = {str(s): detect(records, s) for s in (4, 5)}
                        rows.append(row)
                        handle.write(json.dumps(dict(layout=name, battery=battery, records=records)) + "\n")
        finally:
            env.close()
    write_json(output / "episodes.json", rows)
    summary = dict(episodes=len(rows), mean_return=statistics.mean(r["return_sum"] for r in rows),
                   mean_discounted_return=statistics.mean(r["discounted_return"] for r in rows),
                   mean_work=statistics.mean(r["work"] for r in rows),
                   mean_loop_moves=statistics.mean(r["loop_moves"] for r in rows),
                   outcomes=dict(Counter(r["outcome"] for r in rows)))
    write_json(output / "summary.json", summary)
    return summary


def select_best(candidates):
    if not candidates or any(not np.isfinite(c["mean_return"]) for c in candidates):
        raise ValueError("Missing or nonfinite validation score")
    return max(candidates, key=lambda c: (c["mean_return"], -c["steps"]))


class SelectionCallback(BaseCallback):
    """Save after the previous rollout's updates, including the final update."""
    def __init__(self, output, config, condition, interval, budget):
        super().__init__()
        self.output, self.config, self.condition = output, config, condition
        self.interval, self.budget = interval, budget
        self.candidates = []

    def _on_step(self):
        return True

    def _capture(self):
        step = self.model.num_timesteps
        if not step or step % self.interval or any(c["steps"] == step for c in self.candidates):
            return
        checkpoint = self.output / "checkpoints" / f"step_{step:06d}.zip"
        checkpoint.parent.mkdir(exist_ok=True)
        with preserve_rng():
            self.model.save(checkpoint)
            summary = evaluate(self.model, self.config, self.condition, validation_layouts(),
                               self.output / "validation" / f"step_{step:06d}", sha(checkpoint))
        self.candidates.append(dict(steps=step, updates=self.model._n_updates,
                                    checkpoint=str(checkpoint.relative_to(self.output)),
                                    checkpoint_sha256=sha(checkpoint), **summary))
        print(f"checkpoint {step}: validation return={summary['mean_return']:.6f}", flush=True)

    def _on_rollout_start(self):
        self._capture()

    def _on_training_end(self):
        self._capture()
        if [c["steps"] for c in self.candidates] != list(range(self.interval, self.budget + 1, self.interval)):
            raise RuntimeError("Incomplete checkpoint schedule")


def make_model(config, condition, seed, steps):
    a = config["agent"]
    env = GridTrainingEnv(GridRechargeEnv(config, condition), seed)
    return GridDoubleDQN("MlpPolicy", env, learning_rate=a["learning_rate"],
        buffer_size=a["replay_capacity"], learning_starts=a["learning_starts"], batch_size=a["batch_size"],
        gamma=a["gamma"], train_freq=a["train_frequency"], gradient_steps=a["gradient_steps"],
        replay_buffer_kwargs={"handle_timeout_termination": False}, target_update_interval=a["target_update_interval"],
        exploration_fraction=min(1., a["epsilon_decay_steps"] / steps), exploration_initial_eps=a["epsilon_start"],
        exploration_final_eps=a["epsilon_end"], max_grad_norm=a["gradient_clip"],
        policy_kwargs={"net_arch": a["hidden_sizes"]}, seed=seed, device="cpu", verbose=0)


def train_eval(condition, seed, output, steps=TRAIN_STEPS, interval=INTERVAL, smoke=False):
    if condition not in CONDITIONS or seed not in SEEDS or steps <= 0 or interval <= 0:
        raise ValueError("Invalid condition/seed/budget")
    if steps % interval or interval % 4:
        raise ValueError("Budget must divide into intervals aligned with train_freq=4")
    if not smoke and (steps != TRAIN_STEPS or interval != INTERVAL):
        raise ValueError("Pilot requires 600000 steps and 50000 interval")
    if smoke and "smoke" not in str(output):
        raise ValueError("Smoke output must contain smoke")
    validation = validation_layouts()
    spec = specification()
    spec.update(version="v0.7.2", train_steps=steps, checkpoint_interval=interval,
                phase="smoke" if smoke else "development_pilot", validation_layouts=validation,
                selection_metric="mean_undiscounted_episode_return", tie_break="earliest_step")
    for name in ("experiments/recharge_return/grid_selection.py",
                 "experiments/recharge_return/checkpoint_selection.md", "scripts/reanalyze_final_entry.py"):
        spec["source_hashes"][name] = sha(name)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "specification.json", spec)
    torch.set_num_threads(1)
    model = make_model(spec["config"], condition, seed, steps)
    metrics = MetricsCallback(output / "training_metrics.jsonl")
    selection = SelectionCallback(output, spec["config"], condition, interval, steps)
    try:
        model.learn(total_timesteps=steps, callback=[metrics, selection], log_interval=None)
        best = select_best(selection.candidates)
        final = selection.candidates[-1]
        for role, candidate in (("best", best), ("final", final)):
            shutil.copyfile(output / candidate["checkpoint"], output / f"{role}.zip")
        write_json(output / "selection.json", dict(metric=spec["selection_metric"],
            tie_break=spec["tie_break"], candidates=selection.candidates, selected=best,
            specification_sha256=sha(output / "specification.json")))
        metadata = dict(version="v0.7.2", phase=spec["phase"], condition=condition, seed=seed,
                        training_steps=model.num_timesteps, updates=model._n_updates, episodes=metrics.completed,
                        best_steps=best["steps"], best_sha256=sha(output / "best.zip"),
                        final_sha256=sha(output / "final.zip"), sb3_version=stable_baselines3.__version__,
                        torch_version=torch.__version__, specification_sha256=sha(output / "specification.json"),
                        selection_sha256=sha(output / "selection.json"))
    finally:
        model.get_env().close()
        if metrics.handle:
            metrics.handle.close()
    del model
    for role in ("best", "final"):
        reloaded = GridDoubleDQN.load(output / f"{role}.zip", device="cpu")
        evaluate(reloaded, spec["config"], condition, layouts(), output / f"eval_{role}",
                 metadata[f"{role}_sha256"], events=True)
        del reloaded
    metadata["status"] = "complete"
    write_json(output / "metadata.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--interval", type=int, default=INTERVAL)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dest = args.output or ROOT / args.condition / str(args.seed)
    train_eval(args.condition, args.seed, dest, args.steps, args.interval, args.smoke)
