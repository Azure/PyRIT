# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the CyberScenario class."""

from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from pyrit.executor.attack.core.attack_config import AttackScoringConfig
from pyrit.prompt_target import PromptTarget, PromptChatTarget
from pyrit.scenarios import CyberScenario, CyberStrategy
from pyrit.score import SelfAskTrueFalseScorer


@pytest.fixture
def mock_runtime_env():
    with patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    ):
        yield


@pytest.fixture
def mock_objective_target():
    """Create a mock objective target for testing."""
    mock = MagicMock(spec=PromptTarget)
    mock.get_identifier.return_value = {"__type__": "MockObjectiveTarget", "__module__": "test"}
    return mock


@pytest.fixture
def mock_objective_scorer():
    """Create a mock objective scorer for testing."""
    mock = MagicMock(spec=SelfAskTrueFalseScorer)
    mock.get_identifier.return_value = {"__type__": "MockObjectiveScorer", "__module__": "test"}
    return mock


@pytest.fixture
def mock_adversarial_target():
    """Create a mock adversarial target for testing."""
    mock = MagicMock(spec=PromptChatTarget)
    mock.get_identifier.return_value = {"__type__": "MockAdversarialTarget", "__module__": "test"}
    return mock


@pytest.fixture
def sample_objectives() -> List[str]:
    """Create sample objectives for testing."""
    return ["test prompt 1", "test prompt 2"]


FIXTURES = ["patch_central_database", "mock_runtime_env"]


@pytest.mark.usefixtures(*FIXTURES)
class TestCyberScenarioInitialization:
    """Tests for CyberScenario initialization."""

    def test_init_with_custom_objectives(self, mock_objective_target, mock_objective_scorer, sample_objectives) -> None:
        """Test initialization with custom objectives."""

        scenario = CyberScenario(
            objectives=sample_objectives,
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario._objectives == sample_objectives
        assert scenario._objective_target == mock_objective_target
        assert scenario.name == "Cyber Scenario"
        assert scenario.version == 1

    # TODO: Patch malware.prompt filereader for scenario._objectives comparison
    def test_init_with_default_objectives(self, mock_objective_target, mock_objective_scorer) -> None:
        """Test initialization with default objectives."""

        scenario = CyberScenario(objective_target=mock_objective_target, objective_scorer=mock_objective_scorer)

        assert scenario._objectives == ...
        assert scenario._objective_target == mock_objective_target
        assert scenario.name == "Cyber Scenario"
        assert scenario.version == 1

    def test_init_with_custom_scorer(self, mock_objective_target) -> None:
        """Test initialization with custom scorer."""

        scorer = MagicMock(SelfAskTrueFalseScorer)
        scenario = CyberScenario(objective_target=mock_objective_target, objective_scorer=scorer)
        assert isinstance(scenario._scorer_config, AttackScoringConfig)
        assert scenario._objective_target == mock_objective_target
        assert scenario.name == "Cyber Scenario"
        assert scenario.version == 1

    def test_init_with_default_scorer(self, mock_objective_target) -> None:
        """Test initialization with default scorer."""
        

    def test_init_with_adversarial_chat(self) -> None: ...

    def test_init_with_max_concurrency(self) -> None: ...

    def test_init_with_memory_labels(self) -> None: ...


@pytest.mark.usefixtures(*FIXTURES)
class TestCyberScenarioAttackGeneration:
    """Tests for CyberScenario attack generation."""

    def test_attack_generation_for_all(self) -> None: ...

    def test_attack_generation_for_singleturn(self) -> None: ...

    def test_attack_generation_for_multiturn(self) -> None: ...

    def test_attack_generation_well_formed(self) -> None: ...


@pytest.mark.usefixtures(*FIXTURES)
class TestCyberScenarioExecution:
    """Tests for CyberScenario execution."""

    def test_end_to_end_execution_all(self) -> None: ...

    def test_end_to_end_execution_singleturn(self) -> None: ...

    def test_end_to_end_execution_multiturn(self) -> None: ...

    def test_get_atomic_attacks_async_returns_attacks(self) -> None: ...

    def test_get_prompt_attacks_creates_attack_runs(self) -> None: ...

    def test_attack_runs_include_objectives(self) -> None: ...


@pytest.mark.usefixtures(*FIXTURES)
class TestCyberScenarioProperties:
    """
    Tests for CyberScenario properties and attributes.
    """

    def test_scenario_version_is_set(self, mock_objective_target, mock_objective_scorer, sample_objectives) -> None:
        """Test that scenario version is properly set."""
        scenario = CyberScenario(
            objective_target=mock_objective_target,
            scenario_strategies=[CyberStrategy.SingleTurn],
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario.version == 1

    def test_no_target_duplication(self) -> None:
        """Test that all three targets (adversarial, object, scorer) are distinct."""
