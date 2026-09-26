import pytest
import yaml

from scripts.audit_grid_routes import CONFIG, TASKS, detect, route, simulate


def test_energy_and_discounted_decomposition():
    config = yaml.safe_load(CONFIG.read_text())
    r = simulate(route(("A", "D"), (0, 1)), 60, config)
    assert r["outcome"] == "returned"
    assert r["work"] == 2
    assert r["battery"] == 60 - (r["steps"]-2) - 7
    for condition, weights in config["reward"]["conditions"].items():
        independent = sum(config["agent"]["gamma"]**i * (
            weights["production"]*step["work"] - weights["reserve"]*step["deficit"]
            + step["common"]) for i, step in enumerate(r["records"]))
        assert r["scores"][condition] == pytest.approx(independent)


def test_zero_battery_at_dock_is_exhaustion_and_preserves_onset():
    config = yaml.safe_load(CONFIG.read_text())
    r = simulate(route((), (0, 1)), 18, config)
    assert r["records"][-1]["position"] == (9, 9)
    assert r["outcome"] == "exhausted"
    assert r["events"]["4"]["onset"] == 15
    assert r["events"]["4"]["confirmation"] == 17
    assert simulate(route((), (0, 1)), 15, config)["events"]["4"]["onset"] is None


def test_region_exit_rejects_then_new_entry_can_confirm():
    positions = [(5, 6), (6, 6), (5, 6), (6, 6), (7, 6), (8, 6)]
    records = [dict(step=i+1, before=a, position=b, action="MOVE", outcome=None)
               for i,(a,b) in enumerate(zip(positions,positions[1:]))]
    r = detect(records,4)
    assert r["candidates"][0]["confirmation"] is None
    assert r["onset"] == 3
    assert r["confirmation"] == 5


def test_task_zones_and_unaffordable_work():
    assert all(x < 5 or y < 5 for x,y in TASKS.values())
    config = yaml.safe_load(CONFIG.read_text())
    r = simulate(route(("A",), (0, 1)), 5, config)
    assert r["outcome"] == "exhausted"
    assert r["work"] == 0
    assert len(r["records"]) == 4
