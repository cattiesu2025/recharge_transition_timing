"""Physical work-zone exit onset and auxiliary return event detectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .env import Action


@dataclass(frozen=True)
class DetectorConfig:
    window_steps: int = 8
    progress_moves: int = 3

    def __post_init__(self) -> None:
        if self.window_steps < 1 or self.progress_moves < 1:
            raise ValueError("Detector window and progress must be positive")
        if self.progress_moves > self.window_steps:
            raise ValueError("Progress requirement exceeds window")


@dataclass
class Candidate:
    step: int
    confirmed: bool
    confirmation_step: int | None = None
    progress_moves: int = 0
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class DetectionResult:
    detector: str
    observed: bool
    onset_step: int | None
    confirmation_step: int | None
    candidates: list[Candidate]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _toward_dock(record: dict[str, Any]) -> bool:
    before = record["position_before"]
    after = record["position"]
    return bool(record.get("moved") and after[0] > before[0] and after[1] == before[1] == 4)


def detect_onset(records: Iterable[dict[str, Any]], config: DetectorConfig) -> DetectionResult:
    steps = list(records)
    candidates: list[Candidate] = []
    for index, record in enumerate(steps):
        if record["position_before"] != [1, 4] or record["position"] != [2, 4] or not _toward_dock(record):
            continue
        progress = 0
        confirmation: int | None = None
        reasons: list[str] = []
        for item in steps[index:index + config.window_steps]:
            if item.get("work_completed") or item.get("action") == Action.WORK or item["position"] == [1, 4]:
                reasons.append("returned_to_work_zone")
                break
            if item.get("outcome") == "exhausted":
                reasons.append("exhausted_before_confirmation")
                break
            if item.get("terminated") or item.get("truncated"):
                reasons.append("episode_ended_before_confirmation")
                break
            if _toward_dock(item):
                progress += 1
                if progress >= config.progress_moves:
                    confirmation = int(item["step"])
                    break
            elif item.get("moved") and item["position"][0] < item["position_before"][0]:
                reasons.append("reverse_move")
                break
        if confirmation is None and not reasons:
            reasons.append("insufficient_progress")
        candidates.append(Candidate(int(record["step"]), confirmation is not None,
                                    confirmation, progress, reasons))
    first = next((candidate for candidate in candidates if candidate.confirmed), None)
    return DetectionResult("onset", first is not None,
                           first.step if first else None,
                           first.confirmation_step if first else None, candidates)


def auxiliary_events(records: Iterable[dict[str, Any]], dock_x: int) -> dict[str, int | None]:
    rows = list(records)
    def first(predicate):
        return next((int(row["step"]) for row in rows if predicate(row)), None)
    return {"first_toward_dock_step": first(_toward_dock),
            "first_branch_exit_step": first(lambda r: r["position_before"] == [1, 4] and r["position"] == [2, 4]),
            "first_dock_step": first(lambda r: r["position"] == [dock_x, 4])}
