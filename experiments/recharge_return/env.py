"""Fully observed single-sortie MiniGrid work-to-home corridor.

The work tile is (1, 4); home is (distance + 1, 4). Cells
between them belong only to the return branch. Obstacles occupy every
other cell of the 17 x 9 MiniGrid grid.
"""

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
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    WAIT = 3
    WORK = 4


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


DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


class RechargeEnv(MiniGridEnv):
    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 10}

    def __init__(self, config: dict[str, Any], condition: str = "BAL",
                 render_mode: str | None = None) -> None:
        if render_mode not in (None, "ansi", "rgb_array", "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        super().__init__(mission_space=MissionSpace(
            mission_func=lambda: "Work and return home safely"
        ), width=17, height=9, max_steps=int(config["environment"]["max_steps"]),
            see_through_walls=True, render_mode=render_mode if render_mode != "ansi" else None)
        self._requested_render_mode = render_mode
        self.config = config
        self.params = config["environment"]
        self.reward_params = config["reward"]
        if condition not in self.reward_params["conditions"]:
            raise ValueError(f"Unknown condition: {condition}")
        self.condition = condition
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32)
        self.scenario: Scenario | None = None
        self.position = (1, 4)
        self.direction = 0
        self.battery = 0.0
        self.remaining = 0
        self.steps = 0
        self.total_work = 0
        self.done = False

    @property
    def position(self) -> tuple[int, int]:
        return tuple(self.agent_pos)

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        self.agent_pos = tuple(value)

    @property
    def direction(self) -> int:
        return int(self.agent_dir)

    @direction.setter
    def direction(self, value: int) -> None:
        self.agent_dir = int(value)

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                self.grid.set(x, y, Wall())
        for x in range(1, self.dock[0] + 1):
            color = "green" if x == 1 else "yellow" if x == self.dock[0] else "blue"
            self.grid.set(x, 4, Floor(color))
        self.position = (1, 4)
        self.direction = 0

    @property
    def dock(self) -> tuple[int, int]:
        assert self.scenario is not None
        return (self.scenario.distance + 1, 4)

    def _cost(self, action: Action) -> float:
        costs = self.params["costs"]
        return float(costs[action.name.lower()])

    def minimum_energy_to_dock(self, position: tuple[int, int] | None = None,
                               direction: int | None = None) -> float:
        """Exact minimum route energy, including the cheapest required turn."""
        assert self.scenario is not None
        x, y = self.position if position is None else position
        heading = self.direction if direction is None else direction
        if y != 4 or not 1 <= x <= self.dock[0]:
            raise ValueError("Position is not on the task corridor")
        moves = self.dock[0] - x
        if moves == 0:
            return 0.0
        turns = min((0 - heading) % 4, (heading - 0) % 4)
        return moves * self._cost(Action.FORWARD) + turns * min(
            self._cost(Action.LEFT), self._cost(Action.RIGHT))

    @property
    def margin(self) -> float:
        return self.battery - self.minimum_energy_to_dock()

    def _observation(self) -> np.ndarray:
        assert self.scenario is not None
        capacity = float(self.params["capacity"])
        return np.asarray([
            self.position[0] / 16, self.position[1] / 8,
            self.direction / 3, self.battery / capacity,
            self.remaining / int(self.params["max_quota"]),
            (int(self.params["max_steps"]) - self.steps) / int(self.params["max_steps"]),
            self.scenario.distance / 15, self.margin / capacity if self.margin >= 0 else 0,
        ], dtype=np.float32)

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
        self.position = (1, 4)
        self.direction = 0
        self.battery = scenario.battery
        self.remaining = scenario.quota
        self.steps = self.total_work = 0
        self.done = False
        return self._observation(), self._info()

    def _info(self) -> dict[str, Any]:
        return {"position": list(self.position), "direction": self.direction,
                "battery": self.battery, "remaining": self.remaining,
                "margin": self.margin, "minimum_energy_to_dock": self.minimum_energy_to_dock(),
                "steps": self.steps, "total_work": self.total_work,
                "scenario": asdict(self.scenario) if self.scenario else None}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode has ended; reset before stepping")
        action = Action(int(action))
        before = self.position
        battery_before = self.battery
        affordable = battery_before >= self._cost(action)
        valid = affordable
        work_completed = 0
        if not affordable:
            pass
        elif action == Action.LEFT:
            self.direction = (self.direction - 1) % 4
        elif action == Action.RIGHT:
            self.direction = (self.direction + 1) % 4
        elif action == Action.FORWARD:
            dx, dy = DIRS[self.direction]
            target = (before[0] + dx, before[1] + dy)
            if target[1] == 4 and 1 <= target[0] <= self.dock[0]:
                self.position = target
            else:
                valid = False
        elif action == Action.WORK:
            if self.position == (1, 4) and self.remaining > 0:
                self.remaining -= 1
                self.total_work += 1
                work_completed = 1
            else:
                valid = False
        # Arrival with zero battery is exhaustion, including on the dock.
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
        info.update({"action": action.name, "valid": valid, "moved": self.position != before,
                     "work_completed": work_completed,
                     "completed": completed, "docked": docked, "exhausted": exhausted,
                     "outcome": "returned" if completed else "returned_without_work" if empty_return else "exhausted" if exhausted
                     else "time_limit" if truncated else None})
        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> str | np.ndarray | None:
        if self._requested_render_mode in ("rgb_array", "human"):
            return super().render()
        cells = ["#" * 17 for _ in range(9)]
        row = list(cells[4])
        for x in range(1, self.dock[0] + 1):
            row[x] = "."
        row[1] = "W"
        row[self.dock[0]] = "H"
        row[self.position[0]] = "A"
        cells[4] = "".join(row)
        return "\n".join(cells)


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
                        "position_before": before["position"],
                        "position": after["position"], "direction_before": before["direction"],
                        "direction": after["direction"], "battery_before": before["battery"],
                        "battery": after["battery"], "remaining_before": before["remaining"],
                        "remaining": after["remaining"], "margin_before": before["margin"],
                        "margin": after["margin"],
                        "minimum_energy_to_dock": after["minimum_energy_to_dock"],
                        "minimum_energy_to_dock_before": before["minimum_energy_to_dock"],
                        "work_completed": after["work_completed"], "docked": after["docked"],
                        "moved": after["moved"], "valid": after["valid"],
                        "terminated": terminated, "truncated": truncated,
                        "outcome": after["outcome"], "reward": reward})
        if terminated or truncated:
            return {"records": records, "outcome": after["outcome"],
                    "return": total_reward, "final": after}
