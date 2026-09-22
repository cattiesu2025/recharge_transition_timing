from __future__ import annotations

from copy import deepcopy
import os
import subprocess

import numpy as np
import pytest
from minigrid.minigrid_env import MiniGridEnv

from experiments.recharge_return.agent import DoubleDQN, sample_training_scenario
from experiments.recharge_return.config import canonical_hash, load_config, load_manifest
from experiments.recharge_return.env import Action, RechargeEnv, Scenario, rollout
from experiments.recharge_return.events import DetectorConfig, detect_onset
from scripts.aggregate_recharge_return import aggregate, validate_completeness
from scripts.diagnose_recharge_pilot import (direct_safe_baseline,
                                               exact_optimal_baseline,
                                               trajectory_metrics)


CONFIG_PATH = "experiments/recharge_return/configs/pilot_line_masked_600k.yaml"
CONFIG = load_config(CONFIG_PATH)


def make_env(battery=16, distance=4, quota=6, condition="BAL"):
    env = RechargeEnv(CONFIG, condition)
    env.reset(options={"scenario": Scenario("test", distance, battery, quota)})
    return env


def test_v060_config_and_legacy_rejection():
    assert CONFIG["environment"]["actions"] == ["MOVE_LEFT", "MOVE_RIGHT", "WORK"]
    assert CONFIG["environment"]["costs"] == {
        "move_left": 1.0, "move_right": 1.0, "work": 3.5}
    with pytest.raises(ValueError, match="three-action"):
        load_config("experiments/recharge_return/configs/pilot_no_wait_600k.yaml")


def test_v060_pbs_array_assigns_each_condition_and_seed_once():
    script = "scripts/katana_recharge_pilot_line_masked_v0.6.pbs"
    observed = set()
    for index in range(1, 10):
        env = dict(os.environ, PBS_O_WORKDIR=os.getcwd(),
                   PBS_ARRAY_INDEX=str(index), RECHARGE_VALIDATE_MAPPING_ONLY="1")
        result = subprocess.run(["bash", script], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert CONFIG_PATH in result.stdout
        condition = ("RES", "BAL", "PROD")[(index - 1) // 3]
        seed = (4100, 4101, 4102)[(index - 1) % 3]
        assert f"condition={condition} seed={seed}" in result.stdout
        observed.add((condition, seed))
    assert len(observed) == 9
    env["PBS_ARRAY_INDEX"] = "10"
    result = subprocess.run(["bash", script], env=env, capture_output=True, text=True)
    assert result.returncode != 0


def test_minigrid_line_layout_render_and_observation():
    env = RechargeEnv(CONFIG, render_mode="rgb_array")
    observation, _ = env.reset(options={"scenario": Scenario("test", 4, 16, 6)})
    assert isinstance(env, MiniGridEnv)
    assert env.width == 11 and env.height == 3
    assert observation.shape == (9,)
    assert env.action_space.n == 3 and Action.WORK == 2
    assert observation[-3:].tolist() == [0.0, 1.0, 1.0]
    assert env.grid.get(1, 1).type == "floor"
    assert env.grid.get(5, 1).type == "floor"
    assert env.grid.get(6, 1).type == "wall"
    assert env.render().shape[-1] == 3
    env.close()
    text_env = make_env(distance=4)
    assert text_env.render() == "A-.-.-.-H"


def test_masks_and_energy_follow_one_dimensional_state():
    env = make_env()
    before = env._observation()
    assert env.position == 0 and env.minimum_energy_to_dock() == 4
    env.step(Action.WORK)
    after_work = env._observation()
    assert after_work[2] < before[2] and after_work[3] < before[3]
    assert env.battery == 12.5
    env.step(Action.MOVE_RIGHT)
    assert env.position == 1 and env.minimum_energy_to_dock() == 3
    assert env.action_mask().tolist() == [True, True, False]
    env.step(Action.MOVE_LEFT)
    assert env.position == 0


def test_masked_actions_are_rejected_without_state_change():
    env = make_env()
    before = env._info()
    with pytest.raises(ValueError, match="masked"):
        env.step(Action.MOVE_LEFT)
    assert env._info() == before
    env.step(Action.MOVE_RIGHT)
    before = env._info()
    with pytest.raises(ValueError, match="masked"):
        env.step(Action.WORK)
    assert env._info() == before


def test_random_action_selection_respects_mask():
    env = make_env()
    observation = env._observation()
    sampled = {int(DoubleDQN._random_valid_actions(observation)[0]) for _ in range(200)}
    assert sampled == {Action.MOVE_RIGHT, Action.WORK}
    env.step(Action.MOVE_RIGHT)
    sampled = {int(DoubleDQN._random_valid_actions(env._observation())[0]) for _ in range(200)}
    assert sampled == {Action.MOVE_LEFT, Action.MOVE_RIGHT}


def test_work_cap_requires_final_return_and_masks_work():
    env = make_env(battery=16, quota=1)
    _, _, terminated, _, info = env.step(Action.WORK)
    assert not terminated and info["remaining"] == 0
    assert env.action_mask().tolist() == [False, True, False]
    for _ in range(4):
        _, _, terminated, _, info = env.step(Action.MOVE_RIGHT)
    assert terminated and info["outcome"] == "returned" and info["docked"]
    with pytest.raises(RuntimeError):
        env.step(Action.MOVE_LEFT)


def test_zero_work_return_is_separate_outcome():
    env = make_env(battery=10)
    for _ in range(4):
        _, reward, terminated, _, info = env.step(Action.MOVE_RIGHT)
    assert terminated and info["outcome"] == "returned_without_work"
    assert not info["completed"] and info["docked"] and reward < 0


def test_both_grids_keep_energy_design():
    for name in ("development", "held_out"):
        rows = load_manifest(f"experiments/recharge_return/grids/{name}.json")
        assert len(rows) == 27
        for row in rows:
            env = RechargeEnv(CONFIG)
            env.reset(options={"scenario": row})
            work_energy = row.quota * env._cost(Action.WORK)
            route_energy = row.distance * env._cost(Action.MOVE_RIGHT)
            safe = CONFIG["reward"]["safe_margin"]
            assert row.battery - env._cost(Action.WORK) - route_energy >= safe
            assert row.battery - work_energy - route_energy < safe
            assert CONFIG["environment"]["capacity"] - work_energy - route_energy >= safe


def test_exhaustion_and_zero_battery_arrival_precedence():
    env = make_env(battery=1)
    _, _, terminated, truncated, info = env.step(Action.MOVE_RIGHT)
    assert terminated and not truncated and info["outcome"] == "exhausted"
    env = make_env(battery=7.5)
    env.step(Action.WORK)
    for _ in range(4):
        _, _, terminated, _, info = env.step(Action.MOVE_RIGHT)
    assert terminated and info["outcome"] == "exhausted" and not info["docked"]


def test_confirmed_onset_rejects_first_exit_followed_by_reversal():
    env = make_env(battery=16, distance=4)
    actions = iter([Action.MOVE_RIGHT, Action.MOVE_LEFT,
                    Action.MOVE_RIGHT, Action.MOVE_RIGHT, Action.MOVE_RIGHT,
                    Action.MOVE_RIGHT])
    result = rollout(env, Scenario("test", 4, 16, 6), lambda _: next(actions))
    detection = detect_onset(result["records"], DetectorConfig())
    assert len(detection.candidates) == 2
    assert not detection.candidates[0].confirmed
    assert detection.candidates[0].rejection_reasons == ["reverse_move"]
    assert detection.observed and detection.onset_step == 3 and detection.confirmation_step == 5


def test_direct_and_exact_baselines_cover_current_decision_problem():
    scenario = Scenario("baseline", 4, 16, 6)
    direct = direct_safe_baseline(CONFIG, "PROD", scenario)
    optimum = exact_optimal_baseline(CONFIG, "PROD", scenario)
    assert direct["work"] >= 1 and direct["steps"] == direct["work"] + scenario.distance
    assert optimum["discounted_return"] >= direct["discounted_return"] - 1e-9
    assert optimum["outcome"] in {"returned", "exhausted"}


def test_trajectory_diagnostic_counts_reversal_and_mask_compliance():
    scenario = Scenario("diagnostic", 4, 60, 2)
    actions = iter([Action.WORK, Action.WORK,
                    Action.MOVE_RIGHT, Action.MOVE_LEFT,
                    Action.MOVE_RIGHT, Action.MOVE_RIGHT,
                    Action.MOVE_RIGHT, Action.MOVE_RIGHT])
    result = rollout(RechargeEnv(CONFIG, "PROD"), scenario, lambda _: next(actions))
    detection = detect_onset(result["records"], DetectorConfig()).to_dict()
    row = {"outcome": "returned", "completed_work": 2,
           "return": result["return"], "primary_onset_step": detection["onset_step"]}
    metrics = trajectory_metrics(result["records"], row, 0.99)
    assert metrics["masked_action_violations"] == 0
    assert metrics["left_moves"] == 1 and metrics["right_moves"] == 5
    assert metrics["stationary_nonwork_actions"] == 0
    assert metrics["discounted_return"] != pytest.approx(result["return"])


def test_training_scenarios_match_across_conditions():
    assert sample_training_scenario(CONFIG, 4100, 7) == sample_training_scenario(CONFIG, 4100, 7)


def test_completeness_retains_missing_and_technical_errors():
    rows = [{"condition": "RES", "seed": 1, "scenario_id": "a",
             "intervention": "original", "outcome": "technical_error", "schema_version": 4}]
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
            rows.append({"schema_version": 4, "config_hash": canonical_hash(config),
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
    rows = [{"schema_version": 3, "config_hash": "old", "condition": c, "seed": 1,
             "scenario_id": "a", "intervention": i, "outcome": "returned",
             "terminal_category": "returned", "primary_observed": True,
             "primary_onset_step": 2} for c in ("RES", "BAL", "PROD")
            for i in ("original", "sufficient_battery")]
    integrity = aggregate(config, [Scenario("a", 4, 16, 6)], rows)["integrity"]
    assert not integrity["complete"] and integrity["schema_errors"] == 6
    assert integrity["config_mismatches"] == 6
