# Battery-aware work-to-return transition

当前公开版本的文件哈希清单在 [v0.3.2 manifest](docs/versions/v0.3.2.json)。本地研究日志、计划与用户约束文件保留在 `docs/`，不纳入仓库。

This project implements a single-sortie, battery-aware work-to-return ONSET experiment using the training and audit conventions of the adjacent Highway and LunarLander projects. It currently provides a **pilot implementation**, not trained formal evidence. The task extends MiniGrid 2.5.0 with custom battery, work, reward, and event rules; see [the protocol](experiments/recharge_return/protocol.md) for precise endpoints. v0.2.4 recharge-and-resume checkpoints and outputs belong to a different task and cannot be evaluated or pooled under v0.3.0.

## Run locally

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
pytest -q
python -m experiments.recharge_return.run probe --output outputs/recharge_return_v0.3.0_pilot/probe.jsonl
python -m experiments.recharge_return.run train --condition RES --seed 4100 --output outputs/recharge_return_v0.3.0_pilot/RES/4100
python -m experiments.recharge_return.run evaluate --condition RES --seed 4100 --manifest experiments/recharge_return/grids/development.json --checkpoint outputs/recharge_return_v0.3.0_pilot/RES/4100/final.zip --output outputs/recharge_return_v0.3.0_pilot/dev_eval/RES/4100
```

Repeat training and development evaluation for BAL and PROD and all declared seeds. The default evaluation grid is held out; use it only after the formal freeze. The aggregate requires all condition × seed × scenario × intervention rows:

```bash
python scripts/aggregate_recharge_return.py \
  --config experiments/recharge_return/configs/pilot.yaml \
  --manifest experiments/recharge_return/grids/development.json \
  --input outputs/recharge_return_v0.3.0_pilot/dev_eval \
  --output outputs/recharge_return_v0.3.0_pilot/dev_summary
```

Freeze a reviewed configuration and manifests before formal training:

```bash
python -m experiments.recharge_return.run freeze \
  --config path/to/formal.yaml --confirm-calibrated \
  --pip-lock path/to/pip-freeze.txt --python-lock path/to/python-version.txt \
  --output outputs/recharge_return_formal/freeze
```

Formal training and evaluation require `--freeze-manifest outputs/recharge_return_formal/freeze/manifest.sha256.json`. The freeze command refuses to overwrite a bundle. Preserve a Python dependency lock alongside the bundle before running on Katana.

The pilot config is only a starting point. The feasibility probe does not establish a learned-policy result. Historical v0.2.4 pilot results remain separate.

## Fixed 600,000-step pilot and trajectory diagnostic

`configs/pilot_600k.yaml` differs from the 300,000-step pilot only in `experiment.train_steps`. Run all three conditions and all declared seeds from scratch, then evaluate each final checkpoint on the development grid. Keep the 300,000-step and 600,000-step outputs separate. The diagnostic checks checkpoint/config provenance and complete paired trajectories, then compares each learned return with the best **successful direct route** among `n` WORK actions followed by straight FORWARD actions. It reports discounted return differences, invalid actions, waits, turns, and the interval from the final WORK to confirmed ONSET. This route benchmark is not a claim of global MDP optimality; pilot results are not formal evidence.

For the locally available 300,000-step seed 4100, run:

```bash
.venv/bin/python -m scripts.diagnose_recharge_pilot \
  --config experiments/recharge_return/configs/pilot.yaml \
  --input outputs/recharge_return_v0.3.0_pilot/dev_eval --seeds 4100 \
  --output outputs/recharge_return_v0.3.0_pilot/diagnostic_4100.json
```

After all 600,000-step jobs finish and their outputs are copied locally, omit `--seeds` to require all three declared seeds:

```bash
.venv/bin/python -m scripts.diagnose_recharge_pilot \
  --config experiments/recharge_return/configs/pilot_600k.yaml \
  --input outputs/recharge_return_v0.3.2_pilot_600k/dev_eval \
  --output outputs/recharge_return_v0.3.2_pilot_600k/diagnostic_all.json
```

## Katana environment setup

From the project root on a Katana login node, submit **`scripts/katana_recharge_setup.pbs`**:

```bash
qsub scripts/katana_recharge_setup.pbs
```

The setup job runs on a compute node. It loads `python/3.11.3` by default, creates `/srv/scratch/$USER/environments/recharge-return-py311`, installs CPU PyTorch and `requirements.txt`, runs `pip check` and the unit tests, then writes dependency records to `outputs/recharge_return_v0.3.0_pilot/environment/`. Check the PBS log and `qstat` before training. Set `RECHARGE_PYTHON_MODULE`, `RECHARGE_PYTHON_VERSION`, or `RECHARGE_VENV_DIR` at submission if your Katana account uses different values. Use the same overrides for every later job.

After setup succeeds, submit one 10,000-step smoke job from the project root. The single `scripts/katana_recharge_pilot_train_eval.pbs` job trains, checks its checkpoint, then evaluates the development grid using that checkpoint. The default `RECHARGE_PILOT_BUDGET=300k` retains the earlier paths. `RECHARGE_PILOT_BUDGET=600k` selects the 600,000-step config and writes full pilot results under `outputs/recharge_return_v0.3.2_pilot_600k/` and smoke results under `outputs/recharge_return_v0.3.2_smoke_600k/`. Check the smoke log, checkpoint, and `dev_eval` files before submitting the full pilot. The job sources `scripts/katana_recharge_env.sh` and fails if the shared venv is absent or has the wrong Python/MiniGrid version. Its requested walltime is 08:00:00 for combined work and still needs verification on Katana.

```bash
qsub -v CONDITION=RES,SEED=4100,RECHARGE_TRAIN_STEPS=10000 scripts/katana_recharge_pilot_train_eval.pbs
qsub -v CONDITION=RES,SEED=4100 scripts/katana_recharge_pilot_train_eval.pbs
for condition in RES BAL PROD; do
  for seed in 4100 4101 4102; do
    qsub -v CONDITION=$condition,SEED=$seed,RECHARGE_PILOT_BUDGET=600k scripts/katana_recharge_pilot_train_eval.pbs
  done
done
```
