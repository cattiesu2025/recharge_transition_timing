from __future__ import annotations

from copy import deepcopy
import os
import subprocess

import pytest

from experiments.recharge_return.config import canonical_hash, load_config, load_manifest
from experiments.recharge_return.env import Action, RechargeEnv, Scenario, rollout
from experiments.recharge_return.events import DetectorConfig, detect_onset
from experiments.recharge_return.agent import sample_training_scenario
from scripts.aggregate_recharge_return import aggregate, validate_completeness
from scripts.diagnose_recharge_pilot import direct_safe_baseline, trajectory_metrics
from minigrid.minigrid_env import MiniGridEnv


CONFIG = load_config("experiments/recharge_return/configs/pilot.yaml")


def test_600k_pilot_changes_only_training_budget():
    extended = load_config("experiments/recharge_return/configs/pilot_600k.yaml")
    assert extended["experiment"]["train_steps"] == 600000
    extended["experiment"]["train_steps"] = 300000
    assert extended == CONFIG


def test_v040_pbs_array_assigns_each_condition_and_seed_once():
    script = "scripts/katana_recharge_pilot_600k_v0.4.pbs"
    observed = set()
    for index in range(1, 10):
        env = dict(os.environ, PBS_O_WORKDIR=os.getcwd(),
                   PBS_ARRAY_INDEX=str(index), RECHARGE_VALIDATE_MAPPING_ONLY="1")
        result = subprocess.run(["bash", script], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "configs/pilot_600k.yaml" in result.stdout
        condition = ("RES", "BAL", "PROD")[(index - 1) // 3]
        seed = (4100, 4101, 4102)[(index - 1) % 3]
        assert f"condition={condition} seed={seed}" in result.stdout
        observed.add((condition, seed))
    assert len(observed) == 9

    env["PBS_ARRAY_INDEX"] = "10"
    result = subprocess.run(["bash", script], env=env, capture_output=True, text=True)
    assert result.returncode != 0


def test_direct_safe_baseline_excludes_zero_battery_arrival():
    scenario = Scenario("one_safe_work", 4, 8, 3)
    baseline = direct_safe_baseline(CONFIG, "PROD", scenario)
    assert baseline["work"] == 1
    assert baseline["steps"] == 5
    assert baseline["dock_battery"] == 0.5


def test_trajectory_diagnostic_discount_and_idle_after_work():
    scenario = Scenario("diagnostic", 4, 60, 2)
    actions = iter([Action.WORK, Action.WORK, Action.WORK,
                    Action.LEFT, Action.RIGHT, Action.WAIT,
                    Action.FORWARD, Action.FORWARD, Action.FORWARD, Action.FORWARD])
    result = rollout(RechargeEnv(CONFIG, "PROD"), scenario, lambda _: next(actions))
    row = {"outcome": "returned", "completed_work": 2,
           "return": result["return"], "primary_onset_step": 7}
    metrics = trajectory_metrics(result["records"], row, 0.99)
    assert metrics["invalid_actions"] == 1
    assert metrics["turn_actions"] == 2
    assert metrics["wait_actions"] == 1
    assert metrics["post_work_onset_gap"] == 5
    assert metrics["discounted_return"] != pytest.approx(result["return"])


def make_env(battery=16, distance=4, quota=6):
    env = RechargeEnv(CONFIG)
    env.reset(options={"scenario": Scenario("test", distance, battery, quota)})
    return env


def test_minigrid_backend_layout_and_render():
    env = RechargeEnv(CONFIG, render_mode="rgb_array")
    assert isinstance(env, MiniGridEnv)
    observation, _ = env.reset(options={"scenario": Scenario("test", 4, 16, 6)})
    assert observation.shape == (8,)
    assert env.grid.get(1, 4).type == "floor"
    assert env.grid.get(5, 4).type == "floor"
    assert env.grid.get(1, 3).type == "wall"
    assert env.render().shape[-1] == 3
    env.close()


def test_energy_and_observation_include_horizon_and_quota():
    env = make_env()
    before = env._observation()
    assert env.minimum_energy_to_dock() == 4
    env.step(Action.WORK)
    after = env._observation()
    assert after[4] < before[4] and after[5] < before[5]
    assert env.battery == 12.5
    env.step(Action.LEFT)
    assert env.minimum_energy_to_dock() == 4.5
    env.step(Action.RIGHT)
    env.step(Action.FORWARD)
    assert env.minimum_energy_to_dock() == 3


def test_invalid_action_costs_energy_and_no_work_reward():
    env = make_env()
    env.step(Action.FORWARD)
    battery = env.battery
    _, _, _, _, info = env.step(Action.WORK)
    assert not info["valid"] and not info["work_completed"]
    assert env.remaining == 6 and env.battery == battery - 3.5


def test_work_cap_requires_final_return():
    env = make_env(battery=16, quota=1)
    _, _, terminated, _, info = env.step(Action.WORK)
    assert not terminated and info["remaining"] == 0
    for _ in range(4):
        _, _, terminated, _, info = env.step(Action.FORWARD)
    assert terminated and info["outcome"] == "returned" and info["docked"]
    with pytest.raises(RuntimeError):
        env.step(Action.LEFT)


def test_zero_work_return_is_separate_outcome():
    env = make_env(battery=10)
    assert "CHARGE" not in Action.__members__
    for _ in range(4):
        _, reward, terminated, _, info = env.step(Action.FORWARD)
    assert terminated and info["outcome"] == "returned_without_work"
    assert not info["completed"] and info["docked"] and reward < 0


def test_both_grids_allow_one_safe_work_but_not_full_cap_with_safe_margin():
    for name in ("development", "held_out"):
        rows = load_manifest(f"experiments/recharge_return/grids/{name}.json")
        assert len(rows) == 27
        for row in rows:
            env = RechargeEnv(CONFIG)
            env.reset(options={"scenario": row})
            work_energy = row.quota * env._cost(Action.WORK)
            safe = CONFIG["reward"]["safe_margin"]
            assert row.battery - env._cost(Action.WORK) - row.distance >= safe
            assert row.battery - work_energy - row.distance < safe
            assert CONFIG["environment"]["capacity"] - work_energy - row.distance >= safe


def test_exhaustion_is_terminal_not_time_censoring():
    env = make_env(battery=1)
    _, _, terminated, truncated, info = env.step(Action.FORWARD)
    assert terminated and not truncated and info["outcome"] == "exhausted"


def test_final_work_at_zero_energy_is_exhaustion():
    env = make_env(battery=3.5, quota=1)
    _, _, terminated, truncated, info = env.step(Action.WORK)
    assert terminated and not truncated and info["outcome"] == "exhausted"
    assert not info["completed"]


def test_zero_battery_on_arrival_is_exhaustion():
    env = make_env(battery=7.5)
    env.step(Action.WORK)
    for _ in range(4):
        _, _, terminated, _, info = env.step(Action.FORWARD)
    assert terminated and info["outcome"] == "exhausted"
    assert not info["docked"]


def test_confirmed_return_and_failed_candidates():
    env = make_env(battery=16, distance=4)
    actions = iter([Action.FORWARD, Action.LEFT, Action.LEFT, Action.FORWARD,
                    Action.LEFT, Action.LEFT, Action.FORWARD,
                    Action.FORWARD, Action.FORWARD])
    records = []
    env.reset(options={"scenario": Scenario("test", 4, 16, 6)})
    for action in actions:
        before = env._info()
        _, _, term, trunc, after = env.step(action)
        records.append({"step": env.steps, "position_before": before["position"],
                        "position": after["position"], "moved": after["moved"],
                        "work_completed": after["work_completed"],
                        "outcome": after["outcome"], "terminated": term, "truncated": trunc})
    result = detect_onset(records, DetectorConfig())
    assert len(result.candidates) == 2
    assert not result.candidates[0].confirmed
    assert result.observed and result.onset_step == 7 and result.confirmation_step == 9


def test_noop_command_does_not_create_event():
    env = make_env()
    env.step(Action.LEFT)
    records = []
    for _ in range(4):
        before = env._info()
        _, _, terminated, truncated, after = env.step(Action.FORWARD)
        records.append({"step": env.steps, "position_before": before["position"],
                        "position": after["position"], "moved": after["moved"],
                        "work_completed": 0, "outcome": after["outcome"],
                        "terminated": terminated, "truncated": truncated})
    assert not detect_onset(records, DetectorConfig()).observed


def test_training_scenarios_match_across_conditions():
    assert sample_training_scenario(CONFIG, 4100, 7) == sample_training_scenario(CONFIG, 4100, 7)


def test_completeness_retains_missing_and_technical_errors():
    rows = [{"condition": "RES", "seed": 1, "scenario_id": "a",
             "intervention": "original", "outcome": "technical_error", "schema_version": 2}]
    integrity = validate_completeness(rows, [1], ["a"])
    assert not integrity["complete"] and integrity["technical_errors"] == 1
    assert len(integrity["missing"]) == 5


def test_pilot_never_produces_confirmatory_claim():
    config = deepcopy(CONFIG)
    config["experiment"]["seeds"] = [1]
    config["evaluation"]["minimum_joint_events"] = 1
    rows = []
    for condition in ("RES", "BAL", "PROD"):
        for intervention in ("original", "sufficient_battery"):
            rows.append({"schema_version": 2, "config_hash": canonical_hash(config),
                         "condition": condition, "seed": 1, "scenario_id": "a",
                         "intervention": intervention, "outcome": "returned",
                         "terminal_category": "returned", "primary_observed": True,
                         "primary_onset_step": 2 if condition == "PROD" else 1,
                         "task_success": True, "docked": True, "dock_battery": 4,
                         "exhausted": False, "completed_work": 6})
    result = aggregate(config, [Scenario("a", 4, 16, 6)], rows)
    assert result["integrity"]["complete"]
    assert result["confirmatory_prod_minus_res_steps"] is None


def test_old_schema_and_config_cannot_be_aggregated_as_new_evidence():
    config = deepcopy(CONFIG)
    config["experiment"]["seeds"] = [1]
    rows = [{"schema_version": 1, "config_hash": "old", "condition": c, "seed": 1,
             "scenario_id": "a", "intervention": i, "outcome": "returned",
             "terminal_category": "returned", "primary_observed": True,
             "primary_onset_step": 2} for c in ("RES", "BAL", "PROD")
            for i in ("original", "sufficient_battery")]
    integrity = aggregate(config, [Scenario("a", 4, 16, 6)], rows)["integrity"]
    assert not integrity["complete"] and integrity["schema_errors"] == 6
    assert integrity["config_mismatches"] == 6
