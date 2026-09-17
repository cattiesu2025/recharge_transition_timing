# Battery-aware task-to-recharge transition

当前公开版本的文件哈希清单在 [v0.2.4 manifest](docs/versions/v0.2.4.json)。本地研究日志、计划与用户约束文件保留在 `docs/`，不纳入仓库。

This project implements a battery-aware return-to-charge ONSET experiment using the training and audit conventions of the adjacent Highway and LunarLander projects. It currently provides a **pilot implementation**, not trained formal evidence. The task extends MiniGrid 2.5.0 with custom battery, work, charge, reward, and event rules; see [the protocol](experiments/recharge_return/protocol.md) for precise endpoints.

## Run locally

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
pytest -q
python -m experiments.recharge_return.run probe --output outputs/recharge_return_pilot/probe.jsonl
python -m experiments.recharge_return.run train --condition RES --seed 4100 --output outputs/recharge_return_pilot/RES/4100
python -m experiments.recharge_return.run evaluate --condition RES --seed 4100 --manifest experiments/recharge_return/grids/development.json --checkpoint outputs/recharge_return_pilot/RES/4100/final.zip --output outputs/recharge_return_pilot/dev_eval/RES/4100
```

Repeat training and development evaluation for BAL and PROD and all declared seeds. The default evaluation grid is held out; use it only after the formal freeze. The aggregate requires all condition × seed × scenario × intervention rows:

```bash
python scripts/aggregate_recharge_return.py \
  --config experiments/recharge_return/configs/pilot.yaml \
  --manifest experiments/recharge_return/grids/development.json \
  --input outputs/recharge_return_pilot/dev_eval \
  --output outputs/recharge_return_pilot/dev_summary
```

Freeze a reviewed configuration and manifests before formal training:

```bash
python -m experiments.recharge_return.run freeze \
  --config path/to/formal.yaml --confirm-calibrated \
  --pip-lock path/to/pip-freeze.txt --python-lock path/to/python-version.txt \
  --output outputs/recharge_return_formal/freeze
```

Formal training and evaluation require `--freeze-manifest outputs/recharge_return_formal/freeze/manifest.sha256.json`. The freeze command refuses to overwrite a bundle. Preserve a Python dependency lock alongside the bundle before running on Katana.

The pilot config is only a starting point. No learned policy or hypothesis result has been produced by the included feasibility probe.

## Katana environment setup

From the project root on a Katana login node, submit **`scripts/katana_recharge_setup.pbs`**:

```bash
qsub scripts/katana_recharge_setup.pbs
```

The setup job runs on a compute node. It loads `python/3.11.3` by default, creates `/srv/scratch/$USER/environments/recharge-return-py311`, installs CPU PyTorch and `requirements.txt`, runs `pip check` and the unit tests, then writes dependency records to `outputs/recharge_return_pilot/environment/`. Check the PBS log and `qstat` before training. Set `RECHARGE_PYTHON_MODULE`, `RECHARGE_PYTHON_VERSION`, or `RECHARGE_VENV_DIR` at submission if your Katana account uses different values. Use the same overrides for every later job.

After setup succeeds, submit one 10,000-step training smoke job from the project root. It writes to a separate `outputs/recharge_return_smoke/` directory and exercises updates beyond the 5,000-step learning start. Check its log and checkpoint before submitting the full pilot. The train/eval PBS files source `scripts/katana_recharge_env.sh` and fail if the shared venv is absent or has the wrong Python/MiniGrid version.

```bash
qsub -v CONDITION=RES,SEED=4100,RECHARGE_TRAIN_STEPS=10000 scripts/katana_recharge_pilot_train.pbs
qsub -v CONDITION=RES,SEED=4100 scripts/katana_recharge_pilot_train.pbs
```
