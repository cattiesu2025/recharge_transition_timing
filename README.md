# Battery-aware work-to-return transition

当前公开版本的文件哈希清单在 [v0.3.1 manifest](docs/versions/v0.3.1.json)。本地研究日志、计划与用户约束文件保留在 `docs/`，不纳入仓库。

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

## Katana environment setup

From the project root on a Katana login node, submit **`scripts/katana_recharge_setup.pbs`**:

```bash
qsub scripts/katana_recharge_setup.pbs
```

The setup job runs on a compute node. It loads `python/3.11.3` by default, creates `/srv/scratch/$USER/environments/recharge-return-py311`, installs CPU PyTorch and `requirements.txt`, runs `pip check` and the unit tests, then writes dependency records to `outputs/recharge_return_v0.3.0_pilot/environment/`. Check the PBS log and `qstat` before training. Set `RECHARGE_PYTHON_MODULE`, `RECHARGE_PYTHON_VERSION`, or `RECHARGE_VENV_DIR` at submission if your Katana account uses different values. Use the same overrides for every later job.

After setup succeeds, submit one 10,000-step smoke job from the project root. The single `scripts/katana_recharge_pilot_train_eval.pbs` job trains, checks its checkpoint, then evaluates the development grid using that checkpoint. Smoke training and evaluation write under `outputs/recharge_return_v0.3.0_smoke/`; full pilot outputs write under `outputs/recharge_return_v0.3.0_pilot/`. Check the smoke job log, checkpoint, and `dev_eval` files before submitting the full pilot. The job sources `scripts/katana_recharge_env.sh` and fails if the shared venv is absent or has the wrong Python/MiniGrid version. Its requested walltime is 04:00:00 for the combined work and needs verification on Katana.

```bash
qsub -v CONDITION=RES,SEED=4100,RECHARGE_TRAIN_STEPS=10000 scripts/katana_recharge_pilot_train_eval.pbs
qsub -v CONDITION=RES,SEED=4100 scripts/katana_recharge_pilot_train_eval.pbs
```
