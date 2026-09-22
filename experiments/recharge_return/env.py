"""Fully observed one-dimensional work-to-home MiniGrid pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Callable

import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor, Wall
from minigrid.minigrid_env import MiniGridEnv


class Action(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    WORK = 2


ACTION_COUNT = len(Action)
MASK_SLICE = slice(-ACTION_COUNT, None)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    distance: int
    battery: float
    quota: int

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Scenario":
        return cls(str(row["scenario_id"]), int(row["distance"]),
                   float(row["battery"]), int(row["quota"]))


class RechargeEnv(MiniGridEnv):
    """A straight line with work at x=0 and the terminal dock at x=distance."""

    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 10}

    def __init__(self, config: dict[str, Any], condition: str = "BAL",
                 render_mode: str | None = None) -> None:
        if render_mode not in (None, "ansi", "rgb_array", "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        super().__init__(mission_space=MissionSpace(
            mission_func=lambda: "Work and return home safely"
        ), width=11, height=3, max_steps=int(config["environment"]["max_steps"]),
            see_through_walls=True, render_mode=render_mode if render_mode != "ansi" else None)
        self._requested_render_mode = render_mode
        self.config = config
        self.params = config["environment"]
        self.reward_params = config["reward"]
        if condition not in self.reward_params["conditions"]:
            raise ValueError(f"Unknown condition: {condition}")
        self.condition = condition
        self.action_space = spaces.Discrete(ACTION_COUNT)
        # Six state features followed by three state-dependent valid-action bits.
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6 + ACTION_COUNT,), dtype=np.float32)
        self.scenario: Scenario | None = None
        self._position = 0
        self.battery = 0.0
        self.remaining = 0
        self.steps = 0
        self.total_work = 0
        self.done = False

    @property
    def position(self) -> int:
        return int(self._position)

    @position.setter
    def position(self, value: int) -> None:
        self._position = int(value)
        # MiniGrid coordinates are used only to render the one-dimensional state.
        self.agent_pos = (self._position + 1, 1)
        self.agent_dir = 0

    @property
    def dock(self) -> int:
        assert self.scenario is not None
        return self.scenario.distance

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for grid_x in range(1, width - 1):
            if self.scenario is not None and grid_x <= self.scenario.distance + 1:
                color = "green" if grid_x == 1 else "yellow" if grid_x == self.scenario.distance + 1 else "blue"
                self.grid.set(grid_x, 1, Floor(color))
            else:
                self.grid.set(grid_x, 1, Wall())
        self.position = 0

    def _cost(self, action: Action) -> float:
        return float(self.params["costs"][action.name.lower()])

    def action_mask(self) -> np.ndarray:
        """Mask only actions that are physically or semantically undefined."""
        assert self.scenario is not None
        return np.asarray([
            self.position > 0,
            self.position < self.dock,
            self.position == 0 and self.remaining > 0,
        ], dtype=bool)

    def minimum_energy_to_dock(self, position: int | None = None) -> float:
        assert self.scenario is not None
        x = self.position if position is None else int(position)
        if not 0 <= x <= self.dock:
            raise ValueError("Position is not on the task line")
        return (self.dock - x) * self._cost(Action.MOVE_RIGHT)

    @property
    def margin(self) -> float:
        return self.battery - self.minimum_energy_to_dock()

    def _observation(self) -> np.ndarray:
        assert self.scenario is not None
        capacity = float(self.params["capacity"])
        max_distance = float(max(self.params["training_distribution"]["distances"]))
        state = [
            self.position / max_distance,
            self.battery / capacity,
            self.remaining / int(self.params["max_quota"]),
            (int(self.params["max_steps"]) - self.steps) / int(self.params["max_steps"]),
            self.scenario.distance / max_distance,
            max(0.0, self.margin) / capacity,
        ]
        return np.asarray([*state, *self.action_mask().astype(float)], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        row = (options or {}).get("scenario")
        if row is None:
            row = self.params["default_scenario"]
        scenario = row if isinstance(row, Scenario) else Scenario.from_dict(row)
        if not 4 <= scenario.distance <= 9 or not 1 <= scenario.quota <= int(self.params["max_quota"]):
            raise ValueError("Scenario outside supported geometry or quota")
        if not 0 < scenario.battery <= float(self.params["capacity"]):
            raise ValueError("Scenario battery outside capacity")
        self.scenario = scenario
        super().reset(seed=seed)
        self.position = 0
        self.battery = scenario.battery
        self.remaining = scenario.quota
        self.steps = self.total_work = 0
        self.done = False
        return self._observation(), self._info()

    def _info(self) -> dict[str, Any]:
        return {"position": self.position, "battery": self.battery,
                "remaining": self.remaining, "margin": self.margin,
                "minimum_energy_to_dock": self.minimum_energy_to_dock(),
                "action_mask": self.action_mask().tolist(),
                "steps": self.steps, "total_work": self.total_work,
                "scenario": asdict(self.scenario) if self.scenario else None}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode has ended; reset before stepping")
        action = Action(int(action))
        if not self.action_mask()[action]:
            raise ValueError(f"Action {action.name} is masked in the current state")
        before = self.position
        battery_before = self.battery
        affordable = battery_before >= self._cost(action)
        work_completed = 0
        if affordable and action == Action.MOVE_LEFT:
            self.position -= 1
        elif affordable and action == Action.MOVE_RIGHT:
            self.position += 1
        elif affordable and action == Action.WORK:
            self.remaining -= 1
            self.total_work += 1
            work_completed = 1
        self.battery = max(0.0, self.battery - self._cost(action))
        self.steps += 1
        self.step_count = self.steps
        exhausted = self.battery <= 0
        docked = self.position == self.dock and not exhausted
        completed = docked and self.total_work > 0
        empty_return = docked and self.total_work == 0
        timed_out = self.steps >= int(self.params["max_steps"])
        terminated = docked or exhausted
        truncated = timed_out and not terminated
        self.done = terminated or truncated
        deficit = np.clip((float(self.reward_params["safe_margin"]) - self.margin)
                          / float(self.params["capacity"]), 0.0, 1.0)
        weights = self.reward_params["conditions"][self.condition]
        reward = (float(weights["production"]) * work_completed
                  - float(weights["reserve"]) * float(deficit)
                  - float(self.reward_params["time_cost"]))
        if completed:
            reward += float(self.reward_params["return_bonus"])
        if empty_return:
            reward -= float(self.reward_params["empty_return_penalty"])
        if exhausted:
            reward -= float(self.reward_params["exhaustion_penalty"])
        info = self._info()
        info.update({"action": action.name, "valid": True, "affordable": affordable,
                     "moved": self.position != before, "work_completed": work_completed,
                     "completed": completed, "docked": docked, "exhausted": exhausted,
                     "outcome": "returned" if completed else "returned_without_work" if empty_return else
                                "exhausted" if exhausted else "time_limit" if truncated else None})
        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> str | np.ndarray | None:
        if self._requested_render_mode in ("rgb_array", "human"):
            return super().render()
        cells = ["W", *(["."] * (self.dock - 1)), "H"]
        cells[self.position] = "A"
        return "-".join(cells)


def rollout(env: RechargeEnv, scenario: Scenario,
            policy: Callable[[np.ndarray], int]) -> dict[str, Any]:
    obs, _ = env.reset(options={"scenario": scenario})
    records: list[dict[str, Any]] = []
    total_reward = 0.0
    while True:
        before = env._info()
        action = int(policy(obs))
        obs, reward, terminated, truncated, after = env.step(action)
        total_reward += reward
        records.append({"step": env.steps, "action": action,
                        "position_before": before["position"], "position": after["position"],
                        "battery_before": before["battery"], "battery": after["battery"],
                        "remaining_before": before["remaining"], "remaining": after["remaining"],
                        "margin_before": before["margin"], "margin": after["margin"],
                        "minimum_energy_to_dock": after["minimum_energy_to_dock"],
                        "minimum_energy_to_dock_before": before["minimum_energy_to_dock"],
                        "action_mask_before": before["action_mask"], "action_mask": after["action_mask"],
                        "work_completed": after["work_completed"], "docked": after["docked"],
                        "moved": after["moved"], "valid": after["valid"],
                        "affordable": after["affordable"],
                        "terminated": terminated, "truncated": truncated,
                        "outcome": after["outcome"], "reward": reward})
        if terminated or truncated:
            return {"records": records, "outcome": after["outcome"],
                    "return": total_reward, "final": after}
