#!/usr/bin/env python3
"""Render a seminar GIF explaining the v0.6.0 work-to-return ONSET task.

The animation is deliberately schematic. It illustrates the protocol with one
hand-authored trajectory and must not be interpreted as learned-policy output.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1600
HEIGHT = 900
FPS = 10
BACKGROUND = "#F7F8FA"
INK = "#1F2933"
MUTED = "#647481"
LANE = "#34404A"
LANE_EDGE = "#26313A"
GRID = "#71808B"
BLUE = "#55B6E8"
BLUE_DARK = "#1679AD"
BLUE_PALE = "#DDF3FC"
ORANGE = "#F2A84A"
ORANGE_PALE = "#FCEBD4"
GREEN = "#52B788"
GREEN_PALE = "#DCF4E8"
PINK_PALE = "#FBE6E3"
WHITE = "#FFFFFF"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    bold_candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in bold_candidates if bold else candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_18 = _font(18)
FONT_20 = _font(20)
FONT_22 = _font(22)
FONT_24 = _font(24)
FONT_26_B = _font(26, bold=True)
FONT_30_B = _font(30, bold=True)
FONT_36_B = _font(36, bold=True)
FONT_46_B = _font(46, bold=True)


def _rounded_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    text_fill: str,
    pad_x: int = 16,
    pad_y: int = 8,
    radius: int = 18,
) -> tuple[int, int, int, int]:
    left, top = xy
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + 2 * pad_x
    height = box[3] - box[1] + 2 * pad_y
    bounds = (left, top, left + width, top + height)
    draw.rounded_rectangle(bounds, radius=radius, fill=fill)
    draw.text((left + pad_x, top + pad_y - 2), text, font=font, fill=text_fill)
    return bounds


def _ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _phase_progress(t: float, start: float, end: float) -> float:
    if t <= start:
        return 0.0
    if t >= end:
        return 1.0
    return _ease((t - start) / (end - start))


def _agent_x(t: float, xs: list[int]) -> float:
    moves = [(2.65, 3.25), (3.55, 4.15), (4.45, 5.05), (5.45, 6.05), (6.35, 6.95)]
    x = float(xs[0])
    for index, (start, end) in enumerate(moves):
        progress = _phase_progress(t, start, end)
        if progress > 0:
            x = xs[index] + (xs[index + 1] - xs[index]) * progress
        if progress < 1:
            break
    return x


def _completed_work(t: float) -> int:
    if t >= 2.25:
        return 2
    if t >= 1.55:
        return 1
    return 0


def _battery(t: float, agent_x: float, xs: list[int]) -> float:
    work_cost = 3.5 * _completed_work(t)
    move_cost = max(0.0, (agent_x - xs[0]) / (xs[1] - xs[0]))
    return max(0.0, 19.0 - work_cost - move_cost)


def _draw_battery(draw: ImageDraw.ImageDraw, battery: float) -> None:
    draw.text((1210, 117), "Battery", font=FONT_20, fill=MUTED)
    bx0, by0, bx1, by1 = 1322, 116, 1480, 150
    draw.rounded_rectangle((bx0, by0, bx1, by1), radius=9, outline="#AFBBC4", width=3, fill=WHITE)
    draw.rounded_rectangle((bx1, by0 + 9, bx1 + 10, by1 - 9), radius=3, fill="#AFBBC4")
    ratio = battery / 19.0
    color = GREEN if ratio >= 0.45 else ORANGE
    if ratio > 0:
        draw.rounded_rectangle((bx0 + 5, by0 + 5, bx0 + 5 + int((bx1 - bx0 - 10) * ratio), by1 - 5), radius=5, fill=color)
    draw.text((1498, 116), f"{battery:0.1f}", font=FONT_22, fill=INK, anchor="la")


def _draw_station(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.rounded_rectangle((x - 44, y - 39, x + 44, y + 39), radius=12, fill=WHITE, outline="#D7DEE3", width=3)
    draw.rounded_rectangle((x - 24, y - 22, x + 24, y + 25), radius=7, fill="#E8EDF1")
    draw.line((x - 12, y + 5, x + 12, y + 5), fill="#9AA8B3", width=4)
    draw.line((x, y - 7, x, y + 17), fill="#9AA8B3", width=4)
    draw.text((x, y + 58), "HOME DOCK", font=FONT_18, fill=WHITE, anchor="ma")


def _draw_work_zone(draw: ImageDraw.ImageDraw, x: int, y: int, pulse: float) -> None:
    radius = 52 + int(8 * pulse)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#2D566C", outline="#6BC9F4", width=3)
    draw.rounded_rectangle((x - 28, y - 28, x + 28, y + 28), radius=12, fill=ORANGE)
    draw.line((x - 13, y, x + 13, y), fill=WHITE, width=5)
    draw.line((x, y - 13, x, y + 13), fill=WHITE, width=5)
    draw.text((x, y + 70), "WORK ZONE", font=FONT_18, fill=WHITE, anchor="ma")


def _draw_robot(draw: ImageDraw.ImageDraw, x: float, y: int, work_pulse: float) -> None:
    if work_pulse > 0:
        radius = 42 + int(12 * work_pulse)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="#A7E0F7", width=5)
    draw.rounded_rectangle((x - 39, y - 31, x + 39, y + 31), radius=17, fill=BLUE, outline="#C7EDFC", width=3)
    draw.rounded_rectangle((x - 20, y - 13, x + 18, y + 10), radius=7, fill=WHITE)
    draw.ellipse((x - 11, y - 5, x - 5, y + 1), fill=BLUE_DARK)
    draw.ellipse((x + 4, y - 5, x + 10, y + 1), fill=BLUE_DARK)
    draw.line((x - 15, y + 20, x - 15, y + 29), fill=BLUE_DARK, width=5)
    draw.line((x + 15, y + 20, x + 15, y + 29), fill=BLUE_DARK, width=5)


def _draw_timeline(draw: ImageDraw.ImageDraw, t: float) -> None:
    y = 692
    draw.text((120, 642), "ACTION TIMELINE", font=FONT_18, fill=MUTED)
    actions = [
        ("WORK", 1.05, 1.55),
        ("WORK", 1.75, 2.25),
        ("RIGHT ①", 2.65, 3.25),
        ("RIGHT ②", 3.55, 4.15),
        ("RIGHT ③", 4.45, 5.05),
        ("RIGHT", 5.45, 6.05),
        ("RIGHT", 6.35, 6.95),
    ]
    start_x = 120
    gap = 18
    widths = [130, 130, 164, 164, 164, 130, 130]
    current_x = start_x
    onset_left = 0
    confirm_right = 0
    for index, ((label, start, end), width) in enumerate(zip(actions, widths)):
        active = start <= t <= end
        complete = t > end
        if index < 2:
            fill = ORANGE if active else ORANGE_PALE
            text_fill = WHITE if active else "#9B601B"
        else:
            fill = BLUE_DARK if active else (BLUE_PALE if not complete else "#BFE7F8")
            text_fill = WHITE if active else BLUE_DARK
        draw.rounded_rectangle((current_x, y, current_x + width, y + 54), radius=17, fill=fill)
        draw.text((current_x + width / 2, y + 27), label, font=FONT_20, fill=text_fill, anchor="mm")
        if index == 2:
            onset_left = current_x
        if index == 4:
            confirm_right = current_x + width
        current_x += width + gap
    if t >= 2.65:
        bracket_y = y + 79
        draw.line((onset_left + 2, bracket_y, confirm_right - 2, bracket_y), fill=BLUE_DARK, width=4)
        draw.line((onset_left + 2, bracket_y - 8, onset_left + 2, bracket_y + 8), fill=BLUE_DARK, width=4)
        draw.line((confirm_right - 2, bracket_y - 8, confirm_right - 2, bracket_y + 8), fill=BLUE_DARK, width=4)
        draw.text(((onset_left + confirm_right) / 2, bracket_y + 17), "3 progress moves within W = 8 actions", font=FONT_18, fill=BLUE_DARK, anchor="ma")


def _caption(t: float) -> tuple[str, str, str]:
    if t < 1.0:
        return ("SHARED START STATE", "The policy chooses how much work to complete.", BLUE_PALE)
    if t < 2.55:
        return ("WORKING", "WORK earns production reward and costs 3.5 battery.", ORANGE_PALE)
    if t < 3.35:
        return ("CANDIDATE ONSET", "First actual departure: x = 0 → 1.", PINK_PALE)
    if t < 5.05:
        return ("PERSISTENCE CHECK", "Confirm only after 3 rightward moves within 8 actions.", BLUE_PALE)
    if t < 6.95:
        return ("ONSET CONFIRMED", "The event time stays at the first departure step.", GREEN_PALE)
    return ("RETURNED", "Dock reached → episode ends; there is no charging action.", GREEN_PALE)


def render_frame(t: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((84, 55), "WORK → RETURN", font=FONT_46_B, fill=INK)
    draw.text((84, 111), "When does the agent stop working and begin a sustained return?", font=FONT_24, fill=MUTED)
    _rounded_text(draw, (84, 163), "RES", FONT_20, BLUE_PALE, BLUE_DARK, 18, 7)
    _rounded_text(draw, (174, 163), "BAL", FONT_20, "#EEE8FA", "#7656A8", 18, 7)
    _rounded_text(draw, (264, 163), "PROD", FONT_20, ORANGE_PALE, "#9B601B", 18, 7)
    draw.text((370, 174), "same task · only reward weights differ", font=FONT_20, fill=MUTED)

    lane = (78, 265, 1522, 566)
    draw.rounded_rectangle(lane, radius=24, fill=LANE, outline=LANE_EDGE, width=4)
    lane_y = 416
    draw.line((153, lane_y, 1447, lane_y), fill=GRID, width=5)
    xs = [184, 431, 678, 925, 1172, 1419]
    for index, x in enumerate(xs):
        draw.ellipse((x - 8, lane_y - 8, x + 8, lane_y + 8), fill="#95A3AD")
        draw.text((x, 530), f"x={index}", font=FONT_18, fill="#C6D0D6", anchor="ma")

    pulse = 0.0
    if 1.05 <= t <= 1.55:
        pulse = math.sin(math.pi * (t - 1.05) / 0.5)
    elif 1.75 <= t <= 2.25:
        pulse = math.sin(math.pi * (t - 1.75) / 0.5)
    _draw_work_zone(draw, xs[0], lane_y, pulse)
    _draw_station(draw, xs[-1], lane_y)

    agent_x = _agent_x(t, xs)
    _draw_robot(draw, agent_x, lane_y - 96, pulse)
    battery = _battery(t, agent_x, xs)
    _draw_battery(draw, battery)
    work_done = _completed_work(t)
    draw.text((862, 117), "Work", font=FONT_20, fill=MUTED)
    draw.text((930, 111), f"{work_done} / 8", font=FONT_30_B, fill=INK)
    draw.text((1031, 118), "completed", font=FONT_20, fill=MUTED)

    progress = 0
    if t >= 3.25:
        progress = 1
    if t >= 4.15:
        progress = 2
    if t >= 5.05:
        progress = 3
    if t >= 2.65:
        badge_text = f"ONSET persistence  {progress}/3"
        badge_fill = GREEN_PALE if progress == 3 else BLUE_PALE
        badge_color = "#2F7D5B" if progress == 3 else BLUE_DARK
        _rounded_text(draw, (1020, 188), badge_text, FONT_20, badge_fill, badge_color, 18, 8)

    title, subtitle, fill = _caption(t)
    draw.rounded_rectangle((78, 585, 1522, 630), radius=16, fill=fill)
    draw.text((99, 607), title, font=FONT_22, fill=INK, anchor="lm")
    draw.text((398, 607), subtitle, font=FONT_20, fill=INK, anchor="lm")
    _draw_timeline(draw, t)

    draw.line((78, 836, 1522, 836), fill="#D9E0E5", width=2)
    draw.text((78, 857), "Illustrative protocol trajectory — not a learned-policy result", font=FONT_18, fill=MUTED)
    draw.text((1522, 857), "v0.6.0 masked-action pilot", font=FONT_18, fill=MUTED, anchor="ra")
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/figures/recharge_return_onset_seminar.gif"))
    parser.add_argument("--poster", type=Path, default=Path("docs/figures/recharge_return_onset_seminar_poster.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.poster.parent.mkdir(parents=True, exist_ok=True)

    duration_seconds = 8.4
    frames = [render_frame(index / FPS) for index in range(int(duration_seconds * FPS))]
    frames[-1].save(args.poster)
    palette_frames = [frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for frame in frames]
    palette_frames[0].save(
        args.output,
        save_all=True,
        append_images=palette_frames[1:],
        duration=round(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {args.output} ({len(frames)} frames, {WIDTH}x{HEIGHT}, {FPS} fps)")
    print(f"wrote {args.poster}")


if __name__ == "__main__":
    main()
