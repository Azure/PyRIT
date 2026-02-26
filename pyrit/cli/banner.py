# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Animated ASCII banner for PyRIT CLI.

Displays an animated raccoon mascot revealing the PYRIT logo on shell startup.
Inspired by the GitHub Copilot CLI animated banner approach:
  - Frame-based animation with ANSI cursor repositioning
  - Semantic color roles with light/dark theme support
  - Graceful degradation to static banner when animation isn't supported

The animation plays for ~2.5 seconds and settles into the familiar static banner.
Press Ctrl+C during animation to skip to the static banner immediately.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ColorRole(Enum):
    """Semantic color roles for banner elements."""

    BORDER = "border"
    PYRIT_TEXT = "pyrit_text"
    SUBTITLE = "subtitle"
    RACCOON_BODY = "raccoon_body"
    RACCOON_MASK = "raccoon_mask"
    RACCOON_EYES = "raccoon_eyes"
    RACCOON_TAIL = "raccoon_tail"
    SPARKLE = "sparkle"
    COMMANDS = "commands"
    RESET = "reset"


# ANSI 4-bit color codes (work on virtually all terminals)
ANSI_COLORS = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}

# Theme mappings: role -> ANSI color name
DARK_THEME: dict[ColorRole, str] = {
    ColorRole.BORDER: "cyan",
    ColorRole.PYRIT_TEXT: "bright_red",
    ColorRole.SUBTITLE: "bright_white",
    ColorRole.RACCOON_BODY: "bright_white",
    ColorRole.RACCOON_MASK: "bright_black",
    ColorRole.RACCOON_EYES: "bright_green",
    ColorRole.RACCOON_TAIL: "white",
    ColorRole.SPARKLE: "bright_yellow",
    ColorRole.COMMANDS: "white",
    ColorRole.RESET: "reset",
}

LIGHT_THEME: dict[ColorRole, str] = {
    ColorRole.BORDER: "blue",
    ColorRole.PYRIT_TEXT: "red",
    ColorRole.SUBTITLE: "black",
    ColorRole.RACCOON_BODY: "bright_black",
    ColorRole.RACCOON_MASK: "black",
    ColorRole.RACCOON_EYES: "green",
    ColorRole.RACCOON_TAIL: "bright_black",
    ColorRole.SPARKLE: "yellow",
    ColorRole.COMMANDS: "bright_black",
    ColorRole.RESET: "reset",
}


def _get_color(role: ColorRole, theme: dict[ColorRole, str]) -> str:
    """Resolve a color role to an ANSI escape sequence."""
    color_name = theme.get(role, "reset")
    return ANSI_COLORS.get(color_name, ANSI_COLORS["reset"])


def _detect_theme() -> dict[ColorRole, str]:
    """Detect whether terminal is light or dark themed. Defaults to dark."""
    # COLORFGBG is set by some terminals (e.g. xterm): "fg;bg"
    colorfgbg = os.environ.get("COLORFGBG", "")
    if colorfgbg:
        parts = colorfgbg.split(";")
        if len(parts) >= 2:
            try:
                bg = int(parts[-1])
                # bg >= 8 generally means light background
                if bg >= 8:
                    return LIGHT_THEME
            except ValueError:
                pass
    return DARK_THEME


@dataclass
class AnimationFrame:
    """A single frame of the banner animation."""

    lines: list[str]
    color_map: dict[int, ColorRole] = field(default_factory=dict)
    duration: float = 0.15  # seconds to display this frame


def can_animate() -> bool:
    """Check whether the terminal supports animation."""
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("PYRIT_NO_ANIMATION"):
        return False
    # CI environments
    if os.environ.get("CI"):
        return False
    return True


# ── Raccoon ASCII art ──────────────────────────────────────────────────────────
# The raccoon is ~12 chars wide × 7 lines tall, designed to fit inside the banner box.

RACCOON_FRAMES = [
    # Frame 0: raccoon looking right (walking pose 1)
    [
        r"   /\_/\   ",
        r"  ( o.o )  ",
        r"   > ^ <   ",
        r"  /|   |\ ~",
        r" (_|   |_) ",
    ],
    # Frame 1: raccoon looking right (walking pose 2 - tail up)
    [
        r"   /\_/\  ~",
        r"  ( o.o )  ",
        r"   > ^ <   ",
        r"  /|   |\  ",
        r" (_|   |_) ",
    ],
    # Frame 2: raccoon winking
    [
        r"   /\_/\   ",
        r"  ( -.o )  ",
        r"   > ^ <   ",
        r"  /|   |\ ~",
        r" (_|   |_) ",
    ],
    # Frame 3: raccoon celebrating (arms up)
    [
        r"   /\_/\   ",
        r"  ( ^.^ ) *",
        r"   > ^ <   ",
        r"  \|   |/  ",
        r"  (_   _)  ",
    ],
]


# ── PYRIT block letters (same style as existing banner) ────────────────────────

PYRIT_LETTERS = [
    "██████╗ ██╗   ██╗██████╗ ██╗████████╗",
    "██╔══██╗╚██╗ ██╔╝██╔══██╗██║╚══██╔══╝",
    "██████╔╝ ╚████╔╝ ██████╔╝██║   ██║   ",
    "██╔═══╝   ╚██╔╝  ██╔══██╗██║   ██║   ",
    "██║        ██║   ██║  ██║██║   ██║   ",
    "╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝   ╚═╝   ",
]

# How many characters to reveal per frame (left to right)
PYRIT_WIDTH = 37  # approximate visible width of PYRIT_LETTERS


# ── Static banner (final frame / fallback) ─────────────────────────────────────

STATIC_BANNER_LINES = [
    "╔══════════════════════════════════════════════════════════════════════════════════════════════╗",
    "║                                                                                              ║",
    "║           /\\_/\\          ██████╗ ██╗   ██╗██████╗ ██╗████████╗                              ║",
    "║          ( o.o )         ██╔══██╗╚██╗ ██╔╝██╔══██╗██║╚══██╔══╝                              ║",
    "║           > ^ <          ██████╔╝ ╚████╔╝ ██████╔╝██║   ██║                                 ║",
    "║          /|   |\\ ~       ██╔═══╝   ╚██╔╝  ██╔══██╗██║   ██║                                 ║",
    "║         (_|   |_)        ██║        ██║   ██║  ██║██║   ██║                                 ║",
    "║                          ╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝   ╚═╝                                 ║",
    "║                                                                                              ║",
    "║                          Python Risk Identification Tool                                     ║",
    "║                                Interactive Shell                                             ║",
    "║                                                                                              ║",
    "╠══════════════════════════════════════════════════════════════════════════════════════════════╣",
    "║                                                                                              ║",
    "║  Commands:                                                                                   ║",
    "║    • list-scenarios        - See all available scenarios                                     ║",
    "║    • list-initializers     - See all available initializers                                  ║",
    "║    • run <scenario> [opts] - Execute a security scenario                                     ║",
    "║    • scenario-history      - View your session history                                       ║",
    "║    • print-scenario [N]    - Display detailed results                                        ║",
    "║    • help [command]        - Get help on any command                                         ║",
    "║    • exit                  - Quit the shell                                                  ║",
    "║                                                                                              ║",
    "║  Quick Start:                                                                                ║",
    "║    pyrit> list-scenarios                                                                     ║",
    "║    pyrit> run foundry --initializers openai_objective_target load_default_datasets           ║",
    "║                                                                                              ║",
    "╚══════════════════════════════════════════════════════════════════════════════════════════════╝",
]

# Color role for each line index in the static banner
STATIC_COLOR_MAP: dict[int, ColorRole] = {
    0: ColorRole.BORDER,
    1: ColorRole.BORDER,
    2: ColorRole.RACCOON_BODY,  # raccoon + PYRIT line
    3: ColorRole.RACCOON_BODY,
    4: ColorRole.RACCOON_BODY,
    5: ColorRole.RACCOON_BODY,
    6: ColorRole.RACCOON_BODY,
    7: ColorRole.RACCOON_BODY,
    8: ColorRole.BORDER,
    9: ColorRole.SUBTITLE,
    10: ColorRole.SUBTITLE,
    11: ColorRole.BORDER,
    12: ColorRole.BORDER,
    13: ColorRole.BORDER,
    14: ColorRole.COMMANDS,
    15: ColorRole.COMMANDS,
    16: ColorRole.COMMANDS,
    17: ColorRole.COMMANDS,
    18: ColorRole.COMMANDS,
    19: ColorRole.COMMANDS,
    20: ColorRole.COMMANDS,
    21: ColorRole.COMMANDS,
    22: ColorRole.BORDER,
    23: ColorRole.COMMANDS,
    24: ColorRole.COMMANDS,
    25: ColorRole.COMMANDS,
    26: ColorRole.BORDER,
    27: ColorRole.BORDER,
}


def _build_animation_frames() -> list[AnimationFrame]:
    """Build the sequence of animation frames."""
    frames: list[AnimationFrame] = []
    box_w = 94  # inner width of the box (matches static banner)
    top = "╔" + "═" * box_w + "╗"
    bot = "╚" + "═" * box_w + "╝"
    mid = "╠" + "═" * box_w + "╣"
    empty = "║" + " " * box_w + "║"

    # ── Phase 1: Raccoon enters from right (4 frames) ──────────────────────
    # Raccoon slides in from position 85 → 65 → 45 → 12 (its final x position)
    raccoon_positions = [78, 58, 38, 10]
    for i, x_pos in enumerate(raccoon_positions):
        lines = [top, empty]
        raccoon = RACCOON_FRAMES[i % 2]  # alternate walking poses
        for r_line in raccoon:
            padded = " " * x_pos + r_line
            # Trim/pad to fit box
            content = padded[:box_w].ljust(box_w)
            lines.append("║" + content + "║")
        # Fill remaining rows of logo area with empty (6 rows for raccoon+PYRIT, then subtitle area)
        for _ in range(4):
            lines.append(empty)
        lines.append(empty)  # subtitle placeholder
        lines.append(empty)
        lines.append(mid)
        # commands section (hidden during entry)
        for _ in range(13):
            lines.append(empty)
        lines.append(bot)

        color_map = {j: ColorRole.BORDER for j in range(len(lines))}
        for j in range(2, 2 + len(raccoon)):
            color_map[j] = ColorRole.RACCOON_BODY
        frames.append(AnimationFrame(lines=lines, color_map=color_map, duration=0.18))

    # ── Phase 2: PYRIT text reveals left-to-right (4 frames) ──────────────
    reveal_steps = [9, 18, 27, PYRIT_WIDTH]
    raccoon = RACCOON_FRAMES[0]  # standing pose
    raccoon_x = 10

    for step_i, chars_visible in enumerate(reveal_steps):
        lines = [top, empty]
        for row_i in range(6):
            r_part = ""
            if row_i < len(raccoon):
                r_part = raccoon[row_i]
            raccoon_padded = " " * raccoon_x + r_part
            raccoon_section = raccoon_padded[:24].ljust(24)

            # Reveal PYRIT letters progressively
            if row_i < len(PYRIT_LETTERS):
                full_letter_line = PYRIT_LETTERS[row_i]
                visible = full_letter_line[:chars_visible]
                letter_section = visible.ljust(len(full_letter_line))
            else:
                letter_section = ""

            content = (raccoon_section + "  " + letter_section)[:box_w].ljust(box_w)
            lines.append("║" + content + "║")

        lines.append(empty)
        # Subtitle appears on last reveal step
        if step_i == len(reveal_steps) - 1:
            sub = "Python Risk Identification Tool"
            sub_line = " " * 26 + sub
            lines.append("║" + sub_line[:box_w].ljust(box_w) + "║")
            sub2 = "Interactive Shell"
            sub2_line = " " * 32 + sub2
            lines.append("║" + sub2_line[:box_w].ljust(box_w) + "║")
        else:
            lines.append(empty)
            lines.append(empty)

        lines.append(empty)
        lines.append(mid)
        for _ in range(14):
            lines.append(empty)
        lines.append(bot)

        color_map = {0: ColorRole.BORDER, 1: ColorRole.BORDER}
        for j in range(2, 8):
            color_map[j] = ColorRole.PYRIT_TEXT
        color_map[8] = ColorRole.BORDER
        color_map[9] = ColorRole.SUBTITLE
        color_map[10] = ColorRole.SUBTITLE
        frames.append(AnimationFrame(lines=lines, color_map=color_map, duration=0.15))

    # ── Phase 3: Raccoon wink + sparkle (2 frames) ────────────────────────
    for sparkle_frame in [2, 3]:  # wink, celebrate
        raccoon = RACCOON_FRAMES[sparkle_frame]
        lines = [top, empty]
        for row_i in range(6):
            r_part = ""
            if row_i < len(raccoon):
                r_part = raccoon[row_i]
            raccoon_padded = " " * raccoon_x + r_part
            raccoon_section = raccoon_padded[:24].ljust(24)

            if row_i < len(PYRIT_LETTERS):
                letter_section = PYRIT_LETTERS[row_i]
            else:
                letter_section = ""

            content = (raccoon_section + "  " + letter_section)[:box_w].ljust(box_w)
            lines.append("║" + content + "║")

        lines.append(empty)
        sub = "Python Risk Identification Tool"
        sub_line = " " * 26 + sub
        lines.append("║" + sub_line[:box_w].ljust(box_w) + "║")
        sub2 = "Interactive Shell"
        sub2_line = " " * 32 + sub2
        lines.append("║" + sub2_line[:box_w].ljust(box_w) + "║")
        lines.append(empty)
        lines.append(mid)
        for _ in range(14):
            lines.append(empty)
        lines.append(bot)

        color_map = {}
        for j in range(len(lines)):
            if 2 <= j <= 7:
                color_map[j] = ColorRole.SPARKLE if sparkle_frame == 3 else ColorRole.RACCOON_BODY
            elif j in (9, 10):
                color_map[j] = ColorRole.SUBTITLE
            else:
                color_map[j] = ColorRole.BORDER
        frames.append(AnimationFrame(lines=lines, color_map=color_map, duration=0.25))

    # ── Phase 4: Commands section reveals (2 frames) ──────────────────────
    command_lines = STATIC_BANNER_LINES[14:27]  # commands portion

    for cmd_step in [0, 1]:
        lines = list(STATIC_BANNER_LINES[:14])  # header through divider
        if cmd_step == 0:
            # Show first half of commands
            lines.extend(command_lines[:7])
            for _ in range(6):
                lines.append(empty)
            lines.append(bot)
        else:
            # Show all commands
            lines.extend(command_lines)
            lines.append(bot)

        color_map = dict(STATIC_COLOR_MAP)
        frames.append(AnimationFrame(lines=lines, color_map=color_map, duration=0.15))

    return frames


def _render_frame(frame: AnimationFrame, theme: dict[ColorRole, str]) -> str:
    """Render a single frame with colors applied."""
    reset = _get_color(ColorRole.RESET, theme)
    rendered_lines: list[str] = []
    for i, line in enumerate(frame.lines):
        role = frame.color_map.get(i, ColorRole.BORDER)
        color = _get_color(role, theme)
        rendered_lines.append(f"{color}{line}{reset}")
    return "\n".join(rendered_lines)


def _render_static_banner(theme: dict[ColorRole, str]) -> str:
    """Render the static banner with colors."""
    reset = _get_color(ColorRole.RESET, theme)
    rendered_lines: list[str] = []
    for i, line in enumerate(STATIC_BANNER_LINES):
        role = STATIC_COLOR_MAP.get(i, ColorRole.BORDER)
        color = _get_color(role, theme)
        rendered_lines.append(f"{color}{line}{reset}")
    return "\n".join(rendered_lines)


def get_static_banner() -> str:
    """Get the static (non-animated) banner string, with colors if supported."""
    if sys.stdout.isatty() and not os.environ.get("NO_COLOR"):
        theme = _detect_theme()
        return _render_static_banner(theme)
    return "\n".join(STATIC_BANNER_LINES)


def play_animation(no_animation: bool = False) -> str:
    """
    Play the animated banner or return the static banner.

    Args:
        no_animation: If True, skip animation and return static banner.

    Returns:
        The final static banner string (to be used as the shell intro).
    """
    if no_animation or not can_animate():
        return get_static_banner()

    theme = _detect_theme()
    frames = _build_animation_frames()
    frame_height = max(len(f.lines) for f in frames)

    try:
        # Hide cursor during animation
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

        for frame_idx, frame in enumerate(frames):
            rendered = _render_frame(frame, theme)

            if frame_idx == 0:
                # First frame: just print
                sys.stdout.write(rendered)
                sys.stdout.flush()
            else:
                # Move cursor up to overwrite previous frame
                sys.stdout.write(f"\033[{frame_height}A")
                sys.stdout.write("\r")
                sys.stdout.write(rendered)
                sys.stdout.flush()

            time.sleep(frame.duration)

        # Final frame: overwrite with the static banner (colored)
        sys.stdout.write(f"\033[{frame_height}A")
        sys.stdout.write("\r")
        static = _render_static_banner(theme)
        sys.stdout.write(static)
        sys.stdout.write("\n")
        sys.stdout.flush()

    except KeyboardInterrupt:
        # User pressed Ctrl+C — show static banner immediately
        sys.stdout.write("\r\033[J")  # clear from cursor to end of screen
        static = _render_static_banner(theme)
        sys.stdout.write(static)
        sys.stdout.write("\n")
        sys.stdout.flush()

    finally:
        # Show cursor again
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

    # Return empty string since we already printed the banner
    return ""
