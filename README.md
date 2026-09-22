# Battery-aware work-to-return transition

The published source hashes for this revision are recorded in the [v0.6.0 manifest](docs/versions/v0.6.0.json). Local research notes remain unpublished.

The current v0.6.0 pilot studies how many work units a policy completes before returning along a one-dimensional line. It uses MiniGrid 2.5.0 as the environment and rendering base, but does not ask the agent to solve route planning. The three atomic actions are MOVE_LEFT, MOVE_RIGHT, and WORK. The policy controls every movement and may reverse, so return ONSET is confirmed from the trajectory rather than supplied by a RETURN macro action.

State-dependent action masks remove movement beyond the line and WORK outside the work point or after quota completion. Masks are applied during warmup, epsilon exploration, greedy and deterministic prediction, and Double DQN bootstrap action selection. They never depend on reward condition or battery reserve. The reward conditions, scenario axes, three seeds, and 600,000-step pilot budget are specified in the [protocol](experiments/recharge_return/protocol.md).

The v0.6.0 action order, observation, model output, evaluation schema, and diagnostics are incompatible with v0.5.0 and earlier files. Historical configs, scripts, checkpoints, and outputs remain for audit and require their matching Git revision. Current outputs use `outputs/recharge_return_v0.6.0_pilot_line_masked_600k/`.

## Run locally

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q

.venv/bin/python -m experiments.recharge_return.run probe \
  --config experiments/recharge_return/configs/pilot_line_masked_600k.yaml \
  --output outputs/recharge_return_v0.6.0_pilot_line_masked_600k/probe.jsonl

.venv/bin/python -m experiments.recharge_return.run train \
  --config experiments/recharge_return/configs/pilot_line_masked_600k.yaml \
  --condition RES --seed 4100 \
  --output outputs/recharge_return_v0.6.0_pilot_line_masked_600k/RES/4100

.venv/bin/python -m experiments.recharge_return.run evaluate \
  --config experiments/recharge_return/configs/pilot_line_masked_600k.yaml \
  --condition RES --seed 4100 \
  --manifest experiments/recharge_return/grids/development.json \
  --checkpoint outputs/recharge_return_v0.6.0_pilot_line_masked_600k/RES/4100/final.zip \
  --output outputs/recharge_return_v0.6.0_pilot_line_masked_600k/dev_eval/RES/4100
```

After all RES/BAL/PROD × 4100/4101/4102 jobs finish, run:

```bash
.venv/bin/python -m scripts.diagnose_recharge_pilot \
  --config experiments/recharge_return/configs/pilot_line_masked_600k.yaml \
  --input outputs/recharge_return_v0.6.0_pilot_line_masked_600k/dev_eval \
  --output outputs/recharge_return_v0.6.0_pilot_line_masked_600k/diagnostic_all.json

.venv/bin/python scripts/aggregate_recharge_return.py \
  --config experiments/recharge_return/configs/pilot_line_masked_600k.yaml \
  --manifest experiments/recharge_return/grids/development.json \
  --input outputs/recharge_return_v0.6.0_pilot_line_masked_600k/dev_eval \
  --output outputs/recharge_return_v0.6.0_pilot_line_masked_600k/dev_summary
```

The diagnostic checks checkpoint, config, episode, and trajectory provenance. It compares each policy with the best direct work-and-return route and the exact finite-state optimum, then reports return regret, work, reversals, mask violations, stationary actions, and ONSET timing. Computational baselines and probes verify the environment and reward; they are not learned-policy evidence.

## Run on Katana

If needed, submit `qsub scripts/katana_recharge_setup.pbs` and inspect its log. The setup uses `python/3.11.3` and `/srv/scratch/$USER/environments/recharge-return-py311` by default. Submit all nine fixed-budget pilot jobs from the v0.6.0 checkout with:

```bash
qsub scripts/katana_recharge_pilot_line_masked_v0.6.pbs
```

The PBS array maps indices 1–3 to RES/4100–4102, 4–6 to BAL/4100–4102, and 7–9 to PROD/4100–4102. Each task trains one final checkpoint and evaluates the development manifest. It refuses to overwrite an existing condition/seed output.

## Evidence boundary

This remains a development pilot. Formal work requires a reviewed formal config, 20 declared seeds, immutable freeze, and a separate output directory. Held-out evaluation occurs only after that freeze. Reference, smoke, development, and pilot results do not provide confirmatory R1 evidence.
