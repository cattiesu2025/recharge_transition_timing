# Battery-aware work-to-return transition

当前版本的公开文件哈希在 [v0.5.0 manifest](docs/versions/v0.5.0.json)。本地研究日志、计划与用户约束文件留在 `docs/`，不纳入仓库。

This project implements a single-sortie MiniGrid 2.5.0 pilot. The v0.5.0 agent has four actions: LEFT, RIGHT, FORWARD, WORK. WAIT was removed after the v0.4.0 development pilot showed nonproductive waiting. The reward, grid, model, detector, three conditions, three seeds, and 600,000-step budget remain as specified in the [pilot protocol](experiments/recharge_return/protocol.md). This is pilot work, not formal or held-out evidence.

The five-action v0.4.0 checkpoints and trajectories are incompatible with the four-action model. The current loader rejects the old pilot configs; use the v0.4.0 Git revision when reproducing historical results. New training and evaluation use separate `outputs/recharge_return_v0.5.0_pilot_no_wait_600k/` paths.

## Run locally

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q

.venv/bin/python -m experiments.recharge_return.run probe \
  --config experiments/recharge_return/configs/pilot_no_wait_600k.yaml \
  --output outputs/recharge_return_v0.5.0_pilot_no_wait_600k/probe.jsonl

.venv/bin/python -m experiments.recharge_return.run train \
  --config experiments/recharge_return/configs/pilot_no_wait_600k.yaml \
  --condition RES --seed 4100 \
  --output outputs/recharge_return_v0.5.0_pilot_no_wait_600k/RES/4100

.venv/bin/python -m experiments.recharge_return.run evaluate \
  --config experiments/recharge_return/configs/pilot_no_wait_600k.yaml \
  --condition RES --seed 4100 \
  --manifest experiments/recharge_return/grids/development.json \
  --checkpoint outputs/recharge_return_v0.5.0_pilot_no_wait_600k/RES/4100/final.zip \
  --output outputs/recharge_return_v0.5.0_pilot_no_wait_600k/dev_eval/RES/4100
```

Train and evaluate all RES/BAL/PROD × 4100/4101/4102 combinations before running the complete development diagnostic and aggregate:

```bash
.venv/bin/python -m scripts.diagnose_recharge_pilot \
  --config experiments/recharge_return/configs/pilot_no_wait_600k.yaml \
  --input outputs/recharge_return_v0.5.0_pilot_no_wait_600k/dev_eval \
  --output outputs/recharge_return_v0.5.0_pilot_no_wait_600k/diagnostic_all.json

.venv/bin/python scripts/aggregate_recharge_return.py \
  --config experiments/recharge_return/configs/pilot_no_wait_600k.yaml \
  --manifest experiments/recharge_return/grids/development.json \
  --input outputs/recharge_return_v0.5.0_pilot_no_wait_600k/dev_eval \
  --output outputs/recharge_return_v0.5.0_pilot_no_wait_600k/dev_summary
```

The diagnostic checks checkpoint, config, episode, and trajectory provenance. It compares discounted return with the best successful direct work-and-return route and counts invalid actions, turns, stationary actions without work, and stationary actions after the work cap. The route benchmark is not a global MDP optimum. Removing WAIT does not prevent invalid WORK, blocked FORWARD, or repeated turns; check those outcomes before interpreting ONSET timing.

## Run on Katana

If the project venv has not been prepared, submit `qsub scripts/katana_recharge_setup.pbs` from the project root and check its log. The setup script uses `python/3.11.3` and `/srv/scratch/$USER/environments/recharge-return-py311` by default. Use the same Python module and venv for training. From the v0.5.0 checkout, submit all nine fixed-budget jobs with one command:

```bash
qsub scripts/katana_recharge_pilot_no_wait_v0.5.pbs
```

The PBS `#PBS -J 1-9` array maps indices 1–3 to RES/4100–4102, 4–6 to BAL/4100–4102, and 7–9 to PROD/4100–4102. Each sub-job trains its own final checkpoint and evaluates the development grid. It refuses to overwrite an existing condition/seed directory and requests eight hours per sub-job; check actual completion and walltime on Katana. Copy the whole v0.5.0 output tree locally before running the diagnostic. Historical v0.3.x/v0.4.0 PBS scripts remain in Git for audit but use the five-action configs and are rejected by the current loader.

## Formal boundary

Pilot, formal freeze, formal training, and held-out evaluation remain separate. Formal work needs a reviewed `formal` config with 20 declared seeds and an immutable freeze. The freeze command requires `--confirm-calibrated`, Python and dependency locks, and a separate output directory. Formal training/evaluation require its `--freeze-manifest`; do not use development pilot results as confirmatory R1 evidence.
