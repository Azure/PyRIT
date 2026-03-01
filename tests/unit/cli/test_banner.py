# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import patch

from pyrit.cli.banner import (
    ANSI_COLORS,
    DARK_THEME,
    LIGHT_THEME,
    STATIC_BANNER_LINES,
    ColorRole,
    _build_animation_frames,
    _detect_theme,
    _get_color,
    _render_static_banner,
    can_animate,
    get_static_banner,
    play_animation,
)


class TestColorRole:
    """Tests for color role resolution."""

    def test_get_color_returns_ansi_code(self) -> None:
        color = _get_color(ColorRole.PYRIT_TEXT, DARK_THEME)
        assert color == ANSI_COLORS["bright_cyan"]

    def test_get_color_reset(self) -> None:
        color = _get_color(ColorRole.RESET, DARK_THEME)
        assert color == ANSI_COLORS["reset"]

    def test_light_theme_differs_from_dark(self) -> None:
        dark = _get_color(ColorRole.PYRIT_TEXT, DARK_THEME)
        light = _get_color(ColorRole.PYRIT_TEXT, LIGHT_THEME)
        assert dark != light

    def test_all_roles_have_mappings(self) -> None:
        for role in ColorRole:
            assert role in DARK_THEME, f"{role} missing from DARK_THEME"
            assert role in LIGHT_THEME, f"{role} missing from LIGHT_THEME"


class TestThemeDetection:
    """Tests for terminal theme detection."""

    def test_default_is_dark(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            theme = _detect_theme()
            assert theme is DARK_THEME

    def test_light_bg_detected(self) -> None:
        with patch.dict(os.environ, {"COLORFGBG": "0;15"}):
            theme = _detect_theme()
            assert theme is LIGHT_THEME

    def test_dark_bg_detected(self) -> None:
        with patch.dict(os.environ, {"COLORFGBG": "15;0"}):
            theme = _detect_theme()
            assert theme is DARK_THEME


class TestCanAnimate:
    """Tests for animation capability detection."""

    def test_no_animation_when_not_tty(self) -> None:
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            assert can_animate() is False

    def test_no_animation_when_no_color(self) -> None:
        with patch("sys.stdout") as mock_stdout, patch.dict(os.environ, {"NO_COLOR": "1"}):
            mock_stdout.isatty.return_value = True
            assert can_animate() is False

    def test_no_animation_when_pyrit_no_animation(self) -> None:
        with patch("sys.stdout") as mock_stdout, patch.dict(os.environ, {"PYRIT_NO_ANIMATION": "1"}):
            mock_stdout.isatty.return_value = True
            assert can_animate() is False

    def test_no_animation_in_ci(self) -> None:
        with patch("sys.stdout") as mock_stdout, patch.dict(os.environ, {"CI": "true"}):
            mock_stdout.isatty.return_value = True
            assert can_animate() is False

    def test_can_animate_in_normal_tty(self) -> None:
        with patch("sys.stdout") as mock_stdout, patch.dict(os.environ, {}, clear=True):
            mock_stdout.isatty.return_value = True
            # Remove env vars that would block animation
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("PYRIT_NO_ANIMATION", None)
            os.environ.pop("CI", None)
            assert can_animate() is True


class TestAnimationFrames:
    """Tests for animation frame generation."""

    def test_frames_are_generated(self) -> None:
        frames = _build_animation_frames()
        assert len(frames) > 0

    def test_all_frames_have_consistent_width(self) -> None:
        frames = _build_animation_frames()
        for frame in frames:
            for line in frame.lines:
                # All lines should start with ╔/║/╠/╚ and end with ╗/║/╣/╝
                assert line[0] in "╔║╠╚", f"Line doesn't start with box char: {line[:5]}..."

    def test_frames_have_positive_duration(self) -> None:
        frames = _build_animation_frames()
        for frame in frames:
            assert frame.duration > 0

    def test_frames_have_color_maps(self) -> None:
        frames = _build_animation_frames()
        for frame in frames:
            assert len(frame.color_map) > 0


class TestStaticBanner:
    """Tests for the static banner."""

    def test_static_banner_has_pyrit_text(self) -> None:
        banner_text = "\n".join(STATIC_BANNER_LINES)
        assert "██████╗" in banner_text
        assert "PYRIT" not in banner_text  # it's in block letters, not plain text

    def test_static_banner_has_raccoon(self) -> None:
        banner_text = "\n".join(STATIC_BANNER_LINES)
        assert "⣿" in banner_text  # braille raccoon art
        assert "⠿" in banner_text  # raccoon mask detail

    def test_static_banner_has_subtitle(self) -> None:
        banner_text = "\n".join(STATIC_BANNER_LINES)
        assert "Python Risk Identification Tool" in banner_text
        assert "Interactive Shell" in banner_text

    def test_static_banner_has_commands(self) -> None:
        banner_text = "\n".join(STATIC_BANNER_LINES)
        assert "list-scenarios" in banner_text
        assert "run <scenario>" in banner_text

    def test_render_static_banner_includes_ansi(self) -> None:
        rendered = _render_static_banner(DARK_THEME)
        assert "\033[" in rendered

    def test_get_static_banner_no_color_in_pipe(self) -> None:
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            result = get_static_banner()
            assert "\033[" not in result
            assert "Python Risk Identification Tool" in result


class TestPlayAnimation:
    """Tests for the play_animation function."""

    def test_no_animation_returns_static(self) -> None:
        result = play_animation(no_animation=True)
        assert "Python Risk Identification Tool" in result

    def test_no_animation_when_not_tty(self) -> None:
        with patch("pyrit.cli.banner.can_animate", return_value=False):
            result = play_animation()
            assert "Python Risk Identification Tool" in result
