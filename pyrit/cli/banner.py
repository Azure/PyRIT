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


# ── Raccoon braille art ────────────────────────────────────────────────────────
# High-detail raccoon face rendered in Unicode braille characters.
# The raccoon's bandit mask and features are visible as lighter dot patterns
# against the solid ⣿ background.

BRAILLE_RACCOON = [
    "⠀⠀⠀⠀⠀⠀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⡀⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⣼⢻⠈⢑⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⢎⠁⠉⣻⡀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⡇⠀⠁⢙⣿⣮⢲⠀⠀⠀⠀⠀⠀⠀⢠⣾⣟⠀⠸⢫⡇⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⣧⢀⠀⠘⣷⣿⠆⠀⠐⠘⠿⠓⠀⠀⢾⣧⠃⠀⠐⣼⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⠘⣇⢰⣶⠛⣁⣐⣷⣦⠐⢘⣼⣷⣂⡀⠛⢽⣆⣸⠁⠀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠀⣚⣾⡿⢡⣴⣿⣿⣿⣿⠇⠸⣿⣿⣿⣿⣶⡄⠾⣷⣟⡀⠀⠀⠀⠀",
    "⠀⠀⠀⠀⠘⣻⠇⣲⡿⠟⠋⢉⠉⢿⠰⠆⡿⠋⠉⠙⠿⣿⣆⡻⣿⣓⠀⠀⠀⠀",
    "⠀⠀⠀⣰⢿⣷⠞⢩⠀⠀⠀⠈⢀⣀⠀⡀⣠⡀⠈⠀⠀⣨⠛⢷⣿⣭⠃⠀⠀⠀",
    "⠀⠀⠀⣶⠟⠁⠶⠀⠀⠀⠀⣠⣾⡟⠘⠃⢻⣿⣌⠀⠀⠀⠀⠀⠀⠻⣷⠀⠀⠀",
    "⠀⠀⠘⠿⣔⠺⠀⠀⠀⠀⢰⣿⣿⡀⠘⠀⢀⣿⣿⡆⡂⠀⡈⠡⠜⣙⣿⠇⠀⠀",
    "⠀⠀⠀⠐⠻⢿⣶⣅⢀⠐⠀⠙⣒⡃⡀⠄⢘⠉⠋⠁⠆⢀⢼⣿⣿⡟⠋⠁⠀⠀",
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠭⠛⠿⠿⠛⠧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
]

# ── PYRIT block letters (same style as existing banner) ────────────────────────

PYRIT_LETTERS = [
    "██████╗          ██████╗ ██╗████████╗",
    "██╔══██╗██╗   ██╗██╔══██╗██║╚══██╔══╝",
    "██████╔╝╚██╗ ██╔╝██████╔╝██║   ██║   ",
    "██╔═══╝  ╚████╔╝ ██╔══██╗██║   ██║   ",
    "██║       ╚██╔╝  ██║  ██║██║   ██║   ",
    "╚═╝        ██║   ╚═╝  ╚═╝╚═╝   ╚═╝   ",
    "           ╚═╝                       ",
]

# How many characters to reveal per frame (left to right)
PYRIT_WIDTH = 37  # approximate visible width of PYRIT_LETTERS

# ── Banner layout constants ────────────────────────────────────────────────────

BOX_W = 94  # inner width between ║ chars
RACCOON_COL = 32  # width reserved for raccoon column in header (30 + 2 padding)
HEADER_ROWS = 12  # match braille raccoon height
PYRIT_START_ROW = 2  # PYRIT text starts at this row within the header


def _box_line(content: str) -> str:
    """Wrap content in box border chars, padded to BOX_W."""
    return "║" + content.ljust(BOX_W) + "║"


def _empty_line() -> str:
    return _box_line("")


# ── Static banner (final frame / fallback) ─────────────────────────────────────


def _build_static_banner() -> tuple[list[str], dict[int, ColorRole]]:
    """Build the static banner lines and color map programmatically."""
    raccoon = BRAILLE_RACCOON
    lines: list[str] = []
    color_map: dict[int, ColorRole] = {}

    def add(line: str, role: ColorRole) -> None:
        color_map[len(lines)] = role
        lines.append(line)

    # Top border + empty
    add("╔" + "═" * BOX_W + "╗", ColorRole.BORDER)
    add(_empty_line(), ColorRole.BORDER)

    # Header: braille raccoon + PYRIT text side by side
    # PYRIT text at rows PYRIT_START_ROW..+5, subtitles 2 rows after PYRIT
    subtitle_row_1 = PYRIT_START_ROW + len(PYRIT_LETTERS) + 1
    subtitle_row_2 = subtitle_row_1 + 1
    for i in range(HEADER_ROWS):
        r_part = (" " + raccoon[i] + " ").ljust(RACCOON_COL)
        pyrit_idx = i - PYRIT_START_ROW
        if 0 <= pyrit_idx < len(PYRIT_LETTERS):
            p_part = PYRIT_LETTERS[pyrit_idx]
        elif i == subtitle_row_1:
            p_part = "Python Risk Identification Tool"
        elif i == subtitle_row_2:
            p_part = "      Interactive Shell"
        else:
            p_part = ""
        role = ColorRole.RACCOON_BODY
        add(_box_line(r_part + p_part), role)

    add(_empty_line(), ColorRole.BORDER)

    # Mid divider (with tail attachment point)
    tail_col = 82
    tail = ["║", "│", "║", "│", "║", "│", "╲", " ~"]
    divider_content = "═" * tail_col + "╤" + "═" * (BOX_W - tail_col - 1)
    add("╠" + divider_content + "╣", ColorRole.BORDER)

    # Commands section with striped tail hanging from divider
    commands = [
        "Commands:",
        "  • list-scenarios        - See all available scenarios",
        "  • list-initializers     - See all available initializers",
        "  • run <scenario> [opts] - Execute a security scenario",
        "  • scenario-history      - View your session history",
        "  • print-scenario [N]    - Display detailed results",
        "  • help [command]        - Get help on any command",
        "  • exit                  - Quit the shell",
    ]
    cmd_section: list[tuple[str, ColorRole]] = [
        ("", ColorRole.BORDER),  # empty line after divider
    ]
    for cmd in commands:
        cmd_section.append(("  " + cmd, ColorRole.COMMANDS))
    cmd_section.append(("", ColorRole.BORDER))  # empty line after commands

    for i, (content, cmd_role) in enumerate(cmd_section):
        if i < len(tail):
            content = content.ljust(tail_col) + tail[i]
        add(_box_line(content), cmd_role)

    add(_empty_line(), ColorRole.BORDER)

    # Quick start
    quick_start = [
        "Quick Start:",
        "  pyrit> list-scenarios",
        "  pyrit> run foundry --initializers openai_objective_target load_default_datasets",
    ]
    for qs in quick_start:
        add(_box_line("  " + qs), ColorRole.COMMANDS)

    add(_empty_line(), ColorRole.BORDER)

    # Bottom border
    add("╚" + "═" * BOX_W + "╝", ColorRole.BORDER)

    return lines, color_map


STATIC_BANNER_LINES, STATIC_COLOR_MAP = _build_static_banner()


def _build_animation_frames() -> list[AnimationFrame]:
    """Build the sequence of animation frames."""
    frames: list[AnimationFrame] = []
    target_height = len(STATIC_BANNER_LINES)
    top = "╔" + "═" * BOX_W + "╗"
    bot = "╚" + "═" * BOX_W + "╝"
    mid = "╠" + "═" * BOX_W + "╣"
    empty = _empty_line()

    def _pad_to_height(lines: list[str], color_map: dict[int, ColorRole]) -> None:
        """Pad frame lines to match static banner height."""
        while len(lines) < target_height - 1:  # -1 for bottom border
            color_map[len(lines)] = ColorRole.BORDER
            lines.append(empty)
        color_map[len(lines)] = ColorRole.BORDER
        lines.append(bot)

    # ── Phase 1: Raccoon enters from right (4 frames) ──────────────────────
    raccoon = BRAILLE_RACCOON
    raccoon_w = max(len(line) for line in raccoon)
    raccoon_positions = [BOX_W - raccoon_w, (BOX_W - raccoon_w) * 2 // 3, (BOX_W - raccoon_w) // 3, 1]
    for i, x_pos in enumerate(raccoon_positions):
        lines = [top, empty]
        color_map: dict[int, ColorRole] = {0: ColorRole.BORDER, 1: ColorRole.BORDER}
        for r_line in raccoon:
            padded = " " * x_pos + r_line
            content = padded[:BOX_W].ljust(BOX_W)
            color_map[len(lines)] = ColorRole.RACCOON_BODY
            lines.append("║" + content + "║")
        # Empty line + divider
        color_map[len(lines)] = ColorRole.BORDER
        lines.append(empty)
        color_map[len(lines)] = ColorRole.BORDER
        lines.append(mid)
        _pad_to_height(lines, color_map)
        frames.append(AnimationFrame(lines=lines, color_map=color_map, duration=0.18))

    # ── Phase 2: PYRIT text reveals left-to-right (4 frames) ──────────────
    reveal_steps = [9, 18, 27, PYRIT_WIDTH]
    subtitle_row_1 = PYRIT_START_ROW + len(PYRIT_LETTERS) + 1
    subtitle_row_2 = subtitle_row_1 + 1

    for step_i, chars_visible in enumerate(reveal_steps):
        lines = [top, empty]
        color_map = {0: ColorRole.BORDER, 1: ColorRole.BORDER}

        for row_i in range(HEADER_ROWS):
            r_part = (" " + raccoon[row_i] + " ").ljust(RACCOON_COL)
            pyrit_idx = row_i - PYRIT_START_ROW
            if 0 <= pyrit_idx < len(PYRIT_LETTERS):
                full_letter = PYRIT_LETTERS[pyrit_idx]
                visible = full_letter[:chars_visible]
                p_part = visible.ljust(len(full_letter))
            elif row_i == subtitle_row_1 and step_i == len(reveal_steps) - 1:
                p_part = "Python Risk Identification Tool"
            elif row_i == subtitle_row_2 and step_i == len(reveal_steps) - 1:
                p_part = "      Interactive Shell"
            else:
                p_part = ""
            color_map[len(lines)] = ColorRole.RACCOON_BODY
            lines.append(_box_line(r_part + p_part))

        if step_i == len(reveal_steps) - 1:
            # Fix subtitle colors on final reveal
            for line_idx in range(len(lines)):
                if line_idx >= 2:
                    row_in_header = line_idx - 2
                    if row_in_header in (subtitle_row_1, subtitle_row_2):
                        color_map[line_idx] = ColorRole.SUBTITLE

        color_map[len(lines)] = ColorRole.BORDER
        lines.append(empty)
        color_map[len(lines)] = ColorRole.BORDER
        lines.append(mid)
        _pad_to_height(lines, color_map)
        frames.append(AnimationFrame(lines=lines, color_map=color_map, duration=0.15))

    # ── Phase 3: Sparkle celebration (2 frames) ───────────────────────────
    for sparkle_idx in range(2):
        lines = [top, empty]
        color_map = {0: ColorRole.BORDER, 1: ColorRole.BORDER}
        base_role = ColorRole.SPARKLE if sparkle_idx == 1 else ColorRole.RACCOON_BODY

        for row_i in range(HEADER_ROWS):
            r_part = (" " + raccoon[row_i] + " ").ljust(RACCOON_COL)
            pyrit_idx = row_i - PYRIT_START_ROW
            if 0 <= pyrit_idx < len(PYRIT_LETTERS):
                p_part = PYRIT_LETTERS[pyrit_idx]
            elif row_i == subtitle_row_1:
                p_part = "Python Risk Identification Tool"
            elif row_i == subtitle_row_2:
                p_part = "      Interactive Shell"
            else:
                p_part = ""
            role = ColorRole.SUBTITLE if row_i in (subtitle_row_1, subtitle_row_2) else base_role
            color_map[len(lines)] = role
            lines.append(_box_line(r_part + p_part))

        color_map[len(lines)] = ColorRole.BORDER
        lines.append(empty)
        color_map[len(lines)] = ColorRole.BORDER
        lines.append(mid)
        _pad_to_height(lines, color_map)
        frames.append(AnimationFrame(lines=lines, color_map=color_map, duration=0.25))

    # ── Phase 4: Commands section reveals (2 frames) ──────────────────────
    # Use the actual static banner lines, revealing commands section
    header_end = next(
        i for i, line in enumerate(STATIC_BANNER_LINES) if "╠" in line
    ) + 1  # line after mid divider
    cmd_start = header_end
    cmd_lines = STATIC_BANNER_LINES[cmd_start:]

    for cmd_step in [0, 1]:
        lines = list(STATIC_BANNER_LINES[:cmd_start])
        color_map = {i: STATIC_COLOR_MAP.get(i, ColorRole.BORDER) for i in range(len(lines))}

        if cmd_step == 0:
            half = len(cmd_lines) // 2
            for cl in cmd_lines[:half]:
                color_map[len(lines)] = ColorRole.COMMANDS
                lines.append(cl)
            _pad_to_height(lines, color_map)
        else:
            for j, cl in enumerate(cmd_lines):
                color_map[len(lines)] = STATIC_COLOR_MAP.get(cmd_start + j, ColorRole.COMMANDS)
                lines.append(cl)

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

        # Reserve vertical space so the terminal doesn't scroll during animation.
        # Print blank lines to push content up, then move cursor back to the top.
        sys.stdout.write("\n" * (frame_height - 1))
        sys.stdout.write(f"\033[{frame_height - 1}A")
        sys.stdout.write("\r")
        sys.stdout.flush()

        for frame_idx, frame in enumerate(frames):
            rendered = _render_frame(frame, theme)

            if frame_idx > 0:
                # Move cursor back to the top of the reserved space
                sys.stdout.write(f"\033[{frame_height - 1}A\r")

            sys.stdout.write(rendered)
            sys.stdout.flush()
            time.sleep(frame.duration)

        # Final frame: overwrite with the static banner (colored)
        sys.stdout.write(f"\033[{frame_height - 1}A\r")
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
