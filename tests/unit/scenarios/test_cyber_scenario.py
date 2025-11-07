# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the CyberScenario class."""

from unittest.mock import MagicMock

import pytest

from pyrit.executor.attack import PromptSendingAttack
from pyrit.prompt_converter import Base64Converter
from pyrit.prompt_target import PromptTarget
from pyrit.scenarios import EncodingScenario, EncodingStrategy
from pyrit.score import DecodingScorer, TrueFalseScorer


@pytest.fixture
def mock_objective_target():
    """Create a mock objective target for testing."""
    return MagicMock(spec=PromptTarget)


@pytest.mark.usefixtures("patch_central_database")
class TestCyberScenarioInitialization:
    """Tests for CyberScenario initialization."""
