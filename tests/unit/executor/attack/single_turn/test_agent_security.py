# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.executor.attack.single_turn.agent_security import (
    CANARY_MARKERS,
    create_category_scorer,
    create_sensitive_file_scorer,
)
from pyrit.score import SubStringScorer


class TestCanaryMarkers:
    """Tests for canary marker definitions."""

    def test_all_categories_present(self):
        """Verify all expected categories exist."""
        expected = {"shadow", "passwd", "ssh", "env", "shell_history", "system_config", "proc_environ"}
        assert set(CANARY_MARKERS.keys()) == expected

    def test_markers_are_unique(self):
        """Each category must have a unique marker."""
        markers = list(CANARY_MARKERS.values())
        assert len(markers) == len(set(markers))


class TestScorerFactories:
    """Tests for scorer factory functions."""

    def test_create_sensitive_file_scorer_returns_scorer(self):
        """Should return a SubStringScorer."""
        scorer = create_sensitive_file_scorer()
        assert isinstance(scorer, SubStringScorer)

    def test_create_category_scorer_valid(self):
        """Valid category returns scorer with correct marker."""
        scorer = create_category_scorer("ssh")
        assert isinstance(scorer, SubStringScorer)
        assert scorer._substring == CANARY_MARKERS["ssh"]

    def test_create_category_scorer_invalid_raises(self):
        """Invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Unknown category"):
            create_category_scorer("nonexistent")
