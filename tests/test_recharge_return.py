from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.recharge_return.config import load_config, load_manifest
from experiments.recharge_return.env import Action, RechargeEnv, Scenario, rollout
from experiments.recharge_return.events import DetectorConfig, detect_onset
from experiments.recharge_return.agent import sample_training_scenario
from scripts.aggregate_recharge_return import aggregate, validate_completeness
from minigrid.minigrid_env import MiniGridEnv


CONFIG = load_config("experiments/recharge_return/configs/pilot.yaml")


def make_env(battery=16, distance=4, quota=6):
    env = RechargeEnv(CONFIG)
    env.reset(options={"scenario": Scenario("test", distance, battery, quota)})
    return env


def test_minigrid_backend_layout_and_render():
    env = RechargeEnv(CONFIG, render_mode="rgb_array")
    assert isinstance(env, MiniGridEnv)
    observation, _ = env.reset(options={"scenario": Scenario("test", 4, 16, 6)})
    assert observation.shape == (9,)
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


def test_last_step_to_dock_then_charge_and_continue():
    env = make_env(battery=4.25)
    for _ in range(4):
        _, _, terminated, _, info = env.step(Action.FORWARD)
    assert not terminated and info["position"] == [5, 4] and env.battery == .25
    _, _, terminated, _, info = env.step(Action.CHARGE)
    assert not terminated and info["charged"] and env.battery == 6
    assert env.remaining == 6
    env.step(Action.CHARGE)
    # Return to work and do actual work after first charge.
    env.step(Action.LEFT); env.step(Action.LEFT)
    for _ in range(4):
        env.step(Action.FORWARD)
    _, _, _, _, info = env.step(Action.WORK)
    assert info["work_completed"] == 1


def test_charging_cannot_earn_positive_reward():
    env = make_env(battery=10)
    for _ in range(4):
        env.step(Action.FORWARD)
    rewards = [env.step(Action.CHARGE)[1] for _ in range(4)]
    assert all(reward < 0 for reward in rewards)


def test_both_grids_require_energy_to_finish_without_charge_and_high_battery_is_sufficient():
    for name in ("development", "held_out"):
        rows = load_manifest(f"experiments/recharge_return/grids/{name}.json")
        assert len(rows) == 27
        for row in rows:
            env = RechargeEnv(CONFIG)
            env.reset(options={"scenario": row})
            work_energy = row.quota * env._cost(Action.WORK)
            assert work_energy > row.battery
            assert CONFIG["environment"]["capacity"] - work_energy >= row.distance + CONFIG["reward"]["safe_margin"]


def test_exhaustion_is_terminal_not_time_censoring():
    env = make_env(battery=1)
    _, _, terminated, truncated, info = env.step(Action.FORWARD)
    assert terminated and not truncated and info["outcome"] == "exhausted"


def test_final_work_at_zero_energy_is_exhaustion():
    env = make_env(battery=3.5, quota=1)
    _, _, terminated, truncated, info = env.step(Action.WORK)
    assert terminated and not truncated and info["outcome"] == "exhausted"
    assert not info["completed"]


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
             "intervention": "original", "outcome": "technical_error"}]
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
            rows.append({"condition": condition, "seed": 1, "scenario_id": "a",
                         "intervention": intervention, "outcome": "completed",
                         "terminal_category": "completed", "primary_observed": True,
                         "primary_onset_step": 2 if condition == "PROD" else 1,
                         "task_success": True, "exhausted": False, "completed_work": 6})
    result = aggregate(config, [Scenario("a", 4, 16, 6)], rows)
    assert result["integrity"]["complete"]
    assert result["confirmatory_prod_minus_res_steps"] is None
