# Battery-aware work-to-return transition

当前公开版本的文件哈希清单在 [v0.4.0 manifest](docs/versions/v0.4.0.json)。本地研究日志、计划与用户约束文件保留在 `docs/`，不纳入仓库。

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

After all v0.4.0 600,000-step array jobs finish and their outputs are copied locally, omit `--seeds` to require all three declared seeds:

```bash
.venv/bin/python -m scripts.diagnose_recharge_pilot \
  --config experiments/recharge_return/configs/pilot_600k.yaml \
  --input outputs/recharge_return_v0.4.0_pilot_600k/dev_eval \
  --output outputs/recharge_return_v0.4.0_pilot_600k/diagnostic_all.json
```

## Katana environment setup

From the project root on a Katana login node, submit **`scripts/katana_recharge_setup.pbs`**:

```bash
qsub scripts/katana_recharge_setup.pbs
```

The setup job runs on a compute node. It loads `python/3.11.3` by default, creates `/srv/scratch/$USER/environments/recharge-return-py311`, installs CPU PyTorch and `requirements.txt`, runs `pip check` and the unit tests, then writes dependency records to `outputs/recharge_return_v0.3.0_pilot/environment/`. Check the PBS log and `qstat` before training. Set `RECHARGE_PYTHON_MODULE`, `RECHARGE_PYTHON_VERSION`, or `RECHARGE_VENV_DIR` at submission if your Katana account uses different values. Use the same overrides for every later job.

After setup succeeds, submit the v0.4.0 pilot with one command from the project root. [`katana_recharge_pilot_600k_v0.4.pbs`](scripts/katana_recharge_pilot_600k_v0.4.pbs) fixes the 600,000-step config, all three conditions, and all three declared seeds. Its `#PBS -J 1-9` directive creates nine independently scheduled sub-jobs, each training one final checkpoint and evaluating the development grid. Indices 1–3 are RES/4100–4102, 4–6 BAL/4100–4102, and 7–9 PROD/4100–4102. All results go under `outputs/recharge_return_v0.4.0_pilot_600k/`; the job refuses to overwrite an existing condition/seed output. It requests 08:00:00 per sub-job; actual walltime sufficiency needs verification on Katana. See [Katana's array-job documentation](https://docs.restech.unsw.edu.au/using_katana/running_jobs/) for `#PBS -J` and `PBS_ARRAY_INDEX`.

```bash
qsub scripts/katana_recharge_pilot_600k_v0.4.pbs
```

The earlier `katana_recharge_pilot_train_eval.pbs` remains available for a single condition/seed or a short smoke test. Its default is the 300,000-step pilot, and its explicit `RECHARGE_PILOT_BUDGET=600k` route uses the older v0.3.2 output directory. Do not mix either route's output with the v0.4.0 array directory.
