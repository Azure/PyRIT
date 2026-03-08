# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyrit.analytics.result_analysis import AnalysisResult, AttackStats, analyze_results
from pyrit.models import AttackOutcome, AttackResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_attack(
    *,
    conversation_id: str,
    attack_type: str = "CrescendoAttack",
    outcome: AttackOutcome = AttackOutcome.SUCCESS,
    harm_category: str = "violence",
) -> AttackResult:
    mock_piece = MagicMock()
    mock_piece.targeted_harm_categories = [harm_category]
    mock_piece.converter_identifiers = []

    attack = MagicMock(spec=AttackResult)
    attack.conversation_id = conversation_id
    attack.objective = "test"
    attack.outcome = outcome
    attack.last_response = mock_piece
    attack.attack_identifier = MagicMock()
    attack.attack_identifier.class_name = attack_type
    attack.converter_identifiers = []
    return attack


def _make_result(*, group_by: list[str] | None = None) -> AnalysisResult:
    attacks = [
        _make_attack(conversation_id="c1", attack_type="CrescendoAttack", outcome=AttackOutcome.SUCCESS),
        _make_attack(conversation_id="c2", attack_type="CrescendoAttack", outcome=AttackOutcome.FAILURE),
        _make_attack(
            conversation_id="c3",
            attack_type="RedTeamingAttack",
            outcome=AttackOutcome.SUCCESS,
            harm_category="hate_speech",
        ),
        _make_attack(
            conversation_id="c4",
            attack_type="RedTeamingAttack",
            outcome=AttackOutcome.UNDETERMINED,
            harm_category="hate_speech",
        ),
    ]
    return analyze_results(attacks, group_by=group_by or ["attack_type", "harm_category"])


# ---------------------------------------------------------------------------
# Tests: save_html
# ---------------------------------------------------------------------------


class TestSaveHtml:
    """Tests for save_html function."""

    def test_save_html_creates_file(self, tmp_path: Path) -> None:
        """save_html writes an HTML file at the given path."""
        plotly = pytest.importorskip("plotly")  # noqa: F841
        from pyrit.analytics.visualization import save_html

        result = _make_result()
        out = tmp_path / "report.html"
        returned = save_html(result, out)

        assert returned == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_html_returns_path(self, tmp_path: Path) -> None:
        """save_html returns a Path object."""
        pytest.importorskip("plotly")
        from pyrit.analytics.visualization import save_html

        result = _make_result()
        returned = save_html(result, tmp_path / "r.html")
        assert isinstance(returned, Path)

    def test_save_html_accepts_string_path(self, tmp_path: Path) -> None:
        """save_html accepts a plain string as the path argument."""
        pytest.importorskip("plotly")
        from pyrit.analytics.visualization import save_html

        result = _make_result()
        str_path = str(tmp_path / "report.html")
        returned = save_html(result, str_path)
        assert returned == Path(str_path)
        assert Path(str_path).exists()

    def test_save_html_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_html creates missing parent directories."""
        pytest.importorskip("plotly")
        from pyrit.analytics.visualization import save_html

        result = _make_result()
        nested = tmp_path / "a" / "b" / "report.html"
        save_html(result, nested)
        assert nested.exists()

    def test_save_html_contains_plotly_cdn(self, tmp_path: Path) -> None:
        """The saved HTML includes the Plotly CDN script tag."""
        pytest.importorskip("plotly")
        from pyrit.analytics.visualization import save_html

        result = _make_result()
        out = tmp_path / "report.html"
        save_html(result, out)
        content = out.read_text(encoding="utf-8")
        assert "plotly" in content.lower()

    def test_save_html_contains_title(self, tmp_path: Path) -> None:
        """Custom title appears in the saved HTML."""
        pytest.importorskip("plotly")
        from pyrit.analytics.visualization import save_html

        result = _make_result()
        out = tmp_path / "report.html"
        save_html(result, out, title="My Custom Report")
        content = out.read_text(encoding="utf-8")
        assert "My Custom Report" in content

    def test_save_html_contains_overall_stats(self, tmp_path: Path) -> None:
        """The KPI card includes overall ASR and counts."""
        pytest.importorskip("plotly")
        from pyrit.analytics.visualization import save_html

        result = _make_result()
        out = tmp_path / "report.html"
        save_html(result, out)
        content = out.read_text(encoding="utf-8")
        assert "Overall ASR" in content
        assert "Total Attacks" in content

    def test_save_html_no_plotly_raises(self, tmp_path: Path) -> None:
        """save_html raises ImportError when plotly is not installed."""
        with patch.dict("sys.modules", {"plotly": None, "plotly.graph_objects": None, "plotly.io": None}):
            import importlib

            import pyrit.analytics.visualization as vis_mod

            importlib.reload(vis_mod)

            result = _make_result()
            with pytest.raises(ImportError, match="plotly is required"):
                vis_mod.save_html(result, tmp_path / "r.html")


# ---------------------------------------------------------------------------
# Tests: sparsity guard
# ---------------------------------------------------------------------------


class TestSparsityGuard:
    """Tests for the sparsity filtering in save_html."""

    def test_sparse_heatmap_skipped(self, tmp_path: Path) -> None:
        """A composite dim with >50% empty cells is not rendered as a heatmap."""
        pytest.importorskip("plotly")
        from pyrit.analytics.visualization import _is_sparse

        sparse_data = {
            ("a", "x"): AttackStats(success_rate=0.5, total_decided=2, successes=1, failures=1, undetermined=0),
            ("b", "x"): AttackStats(success_rate=None, total_decided=0, successes=0, failures=0, undetermined=1),
            ("c", "x"): AttackStats(success_rate=None, total_decided=0, successes=0, failures=0, undetermined=1),
        }
        assert _is_sparse(sparse_data, threshold=0.5) is True

    def test_dense_heatmap_not_skipped(self) -> None:
        """A composite dim with ≤50% empty cells passes the sparsity check."""
        from pyrit.analytics.visualization import _is_sparse

        dense_data = {
            ("a", "x"): AttackStats(success_rate=0.8, total_decided=5, successes=4, failures=1, undetermined=0),
            ("b", "x"): AttackStats(success_rate=0.4, total_decided=5, successes=2, failures=3, undetermined=0),
            ("c", "x"): AttackStats(success_rate=None, total_decided=0, successes=0, failures=0, undetermined=1),
        }
        assert _is_sparse(dense_data, threshold=0.5) is False

    def test_empty_dim_data_is_sparse(self) -> None:
        """Empty dim_data is considered sparse."""
        from pyrit.analytics.visualization import _is_sparse

        assert _is_sparse({}) is True


# ---------------------------------------------------------------------------
# Tests: heatmap builders
# ---------------------------------------------------------------------------


class TestBuildZMatrix:
    """Tests for _build_z_matrix."""

    def test_z_matrix_shape(self) -> None:
        """z matrix dimensions match row_keys × col_keys."""
        from pyrit.analytics.visualization import _build_z_matrix

        row_keys = ["r1", "r2"]
        col_keys = ["c1", "c2", "c3"]
        lookup = {
            ("r1", "c1"): AttackStats(success_rate=0.8, total_decided=5, successes=4, failures=1, undetermined=0),
        }
        z, text = _build_z_matrix(row_keys=row_keys, col_keys=col_keys, lookup=lookup)
        assert len(z) == 2
        assert len(z[0]) == 3

    def test_missing_cell_is_none(self) -> None:
        """Cells absent from lookup get None in the z matrix."""
        from pyrit.analytics.visualization import _build_z_matrix

        z, _ = _build_z_matrix(row_keys=["r1"], col_keys=["c1"], lookup={})
        assert z[0][0] is None

    def test_present_cell_has_rate(self) -> None:
        """Cells present in lookup get the correct success_rate."""
        from pyrit.analytics.visualization import _build_z_matrix

        lookup = {
            ("r1", "c1"): AttackStats(success_rate=0.75, total_decided=4, successes=3, failures=1, undetermined=0),
        }
        z, _ = _build_z_matrix(row_keys=["r1"], col_keys=["c1"], lookup=lookup)
        assert z[0][0] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestAsrCssClass:
    """Tests for _asr_css_class."""

    def test_high_rate_is_green(self) -> None:
        from pyrit.analytics.visualization import _asr_css_class

        assert _asr_css_class(0.9) == "green"

    def test_mid_rate_is_yellow(self) -> None:
        from pyrit.analytics.visualization import _asr_css_class

        assert _asr_css_class(0.45) == "yellow"

    def test_low_rate_is_red(self) -> None:
        from pyrit.analytics.visualization import _asr_css_class

        assert _asr_css_class(0.1) == "red"

    def test_none_rate_is_empty(self) -> None:
        from pyrit.analytics.visualization import _asr_css_class

        assert _asr_css_class(None) == ""
