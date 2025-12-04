# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the ContentHarmsScenario class."""

from unittest.mock import MagicMock, patch

import pytest

from pyrit.models.seed_group import SeedGroup
from pyrit.models.seed_objective import SeedObjective
from pyrit.models.seed_prompt import SeedPrompt
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.scenario import (
    ContentHarmsScenario,
    ContentHarmsStrategy,
)
from pyrit.score import TrueFalseScorer


@pytest.fixture
def mock_objective_target():
    """Create a mock objective target for testing."""
    mock = MagicMock(spec=PromptTarget)
    mock.get_identifier.return_value = {"__type__": "MockObjectiveTarget", "__module__": "test"}
    return mock


@pytest.fixture
def mock_adversarial_target():
    """Create a mock adversarial target for testing."""
    mock = MagicMock(spec=PromptChatTarget)
    mock.get_identifier.return_value = {"__type__": "MockAdversarialTarget", "__module__": "test"}
    return mock


@pytest.fixture
def mock_objective_scorer():
    """Create a mock objective scorer for testing."""
    mock = MagicMock(spec=TrueFalseScorer)
    mock.get_identifier.return_value = {"__type__": "MockObjectiveScorer", "__module__": "test"}
    return mock


@pytest.fixture
def sample_objectives():
    """Create sample objectives for testing."""
    return ["objective1", "objective2", "objective3"]


@pytest.fixture
def mock_seed_groups():
    """Create mock seed groups for testing."""

    def create_seed_groups_for_strategy(strategy_name: str):
        """Helper to create seed groups for a given strategy."""
        return [
            SeedGroup(
                seeds=[
                    SeedObjective(value=f"{strategy_name} objective 1"),
                    SeedPrompt(value=f"{strategy_name} prompt 1"),
                ]
            ),
            SeedGroup(
                seeds=[
                    SeedObjective(value=f"{strategy_name} objective 2"),
                    SeedPrompt(value=f"{strategy_name} prompt 2"),
                ]
            ),
        ]

    return create_seed_groups_for_strategy


class TestContentHarmsStrategy:
    """Tests for the ContentHarmsStrategy enum."""

    def test_all_harm_categories_exist(self):
        """Test that all expected harm categories exist as strategies."""
        expected_categories = ["hate", "fairness", "violence", "sexual", "harassment", "misinformation", "leakage"]
        strategy_values = [s.value for s in ContentHarmsStrategy if s != ContentHarmsStrategy.ALL]

        for category in expected_categories:
            assert category in strategy_values, f"Expected harm category '{category}' not found in strategies"

    def test_strategy_tags_are_sets(self):
        """Test that all strategy tags are set objects."""
        for strategy in ContentHarmsStrategy:
            assert isinstance(strategy.tags, set), f"Tags for {strategy.name} are not a set"

    def test_enum_members_count(self):
        """Test that we have the expected number of strategy members."""
        # ALL + 7 harm categories = 8 total
        assert len(list(ContentHarmsStrategy)) == 8

    def test_all_strategies_can_be_accessed_by_name(self):
        """Test that all strategies can be accessed by their name."""
        assert ContentHarmsStrategy.ALL == ContentHarmsStrategy["ALL"]
        assert ContentHarmsStrategy.Hate == ContentHarmsStrategy["Hate"]
        assert ContentHarmsStrategy.Fairness == ContentHarmsStrategy["Fairness"]
        assert ContentHarmsStrategy.Violence == ContentHarmsStrategy["Violence"]
        assert ContentHarmsStrategy.Sexual == ContentHarmsStrategy["Sexual"]
        assert ContentHarmsStrategy.Harassment == ContentHarmsStrategy["Harassment"]
        assert ContentHarmsStrategy.Misinformation == ContentHarmsStrategy["Misinformation"]
        assert ContentHarmsStrategy.Leakage == ContentHarmsStrategy["Leakage"]

    def test_all_strategies_can_be_accessed_by_value(self):
        """Test that all strategies can be accessed by their value."""
        assert ContentHarmsStrategy("all") == ContentHarmsStrategy.ALL
        assert ContentHarmsStrategy("hate") == ContentHarmsStrategy.Hate
        assert ContentHarmsStrategy("fairness") == ContentHarmsStrategy.Fairness
        assert ContentHarmsStrategy("violence") == ContentHarmsStrategy.Violence
        assert ContentHarmsStrategy("sexual") == ContentHarmsStrategy.Sexual
        assert ContentHarmsStrategy("harassment") == ContentHarmsStrategy.Harassment
        assert ContentHarmsStrategy("misinformation") == ContentHarmsStrategy.Misinformation
        assert ContentHarmsStrategy("leakage") == ContentHarmsStrategy.Leakage

    def test_strategies_are_unique(self):
        """Test that all strategy values are unique."""
        values = [s.value for s in ContentHarmsStrategy]
        assert len(values) == len(set(values)), "Strategy values are not unique"

    def test_strategy_iteration(self):
        """Test that we can iterate over all strategies."""
        strategies = list(ContentHarmsStrategy)
        assert len(strategies) == 8
        assert ContentHarmsStrategy.ALL in strategies
        assert ContentHarmsStrategy.Hate in strategies

    def test_strategy_comparison(self):
        """Test that strategy comparison works correctly."""
        assert ContentHarmsStrategy.Hate == ContentHarmsStrategy.Hate
        assert ContentHarmsStrategy.Hate != ContentHarmsStrategy.Violence
        assert ContentHarmsStrategy.ALL != ContentHarmsStrategy.Hate

    def test_strategy_hash(self):
        """Test that strategies can be hashed and used in sets/dicts."""
        strategy_set = {ContentHarmsStrategy.Hate, ContentHarmsStrategy.Violence}
        assert len(strategy_set) == 2
        assert ContentHarmsStrategy.Hate in strategy_set

        strategy_dict = {ContentHarmsStrategy.Hate: "hate_value"}
        assert strategy_dict[ContentHarmsStrategy.Hate] == "hate_value"

    def test_strategy_string_representation(self):
        """Test string representation of strategies."""
        assert "Hate" in str(ContentHarmsStrategy.Hate)
        assert "ALL" in str(ContentHarmsStrategy.ALL)

    def test_invalid_strategy_value_raises_error(self):
        """Test that accessing invalid strategy value raises ValueError."""
        with pytest.raises(ValueError):
            ContentHarmsStrategy("invalid_strategy")

    def test_invalid_strategy_name_raises_error(self):
        """Test that accessing invalid strategy name raises KeyError."""
        with pytest.raises(KeyError):
            ContentHarmsStrategy["InvalidStrategy"]

    def test_get_aggregate_tags_includes_harm_categories(self):
        """Test that get_aggregate_tags includes 'all' tag."""
        aggregate_tags = ContentHarmsStrategy.get_aggregate_tags()

        # The simple implementation only returns the 'all' tag
        assert "all" in aggregate_tags
        assert isinstance(aggregate_tags, set)

    def test_get_aggregate_tags_returns_set(self):
        """Test that get_aggregate_tags returns a set."""
        aggregate_tags = ContentHarmsStrategy.get_aggregate_tags()
        assert isinstance(aggregate_tags, set)

    def test_get_aggregate_strategies(self):
        """Test that ALL aggregate expands to all individual harm strategies."""
        # The ALL strategy should include all individual harm categories
        all_strategies = list(ContentHarmsStrategy)
        assert len(all_strategies) == 8  # ALL + 7 harm categories


@pytest.mark.usefixtures("patch_central_database")
class TestContentHarmsScenarioBasic:
    """Basic tests for ContentHarmsScenario initialization and properties."""

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_objectives_by_harm")
    async def test_initialization_with_minimal_parameters(
        self,
        mock_get_objectives,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test initialization with only required parameters."""
        mock_get_scorer.return_value = mock_objective_scorer
        # Return seed groups for all harm strategies that might be used
        mock_get_objectives.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
            "violence": mock_seed_groups("violence"),
            "sexual": mock_seed_groups("sexual"),
            "harassment": mock_seed_groups("harassment"),
            "misinformation": mock_seed_groups("misinformation"),
            "leakage": mock_seed_groups("leakage"),
        }

        scenario = ContentHarmsScenario(adversarial_chat=mock_adversarial_target)

        # Constructor should set adversarial chat and basic metadata
        assert scenario._adversarial_chat == mock_adversarial_target
        assert scenario.name == "Content Harms Scenario"
        assert scenario.version == 1

        # Initialization populates objective target and scenario composites
        await scenario.initialize_async(objective_target=mock_objective_target)

        assert scenario._objective_target == mock_objective_target

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_objectives_by_harm")
    async def test_initialization_with_custom_strategies(
        self,
        mock_get_objectives,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test initialization with custom harm strategies."""
        mock_get_scorer.return_value = mock_objective_scorer
        mock_get_objectives.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
        }

        strategies = [ContentHarmsStrategy.Hate, ContentHarmsStrategy.Fairness]

        scenario = ContentHarmsScenario(adversarial_chat=mock_adversarial_target)

        await scenario.initialize_async(objective_target=mock_objective_target, scenario_strategies=strategies)

        # Prepared composites should match provided strategies
        assert len(scenario._scenario_composites) == 2

    def test_initialization_with_custom_scorer(
        self, mock_objective_target, mock_adversarial_target, mock_objective_scorer
    ):
        """Test initialization with custom objective scorer."""
        scenario = ContentHarmsScenario(
            adversarial_chat=mock_adversarial_target,
            objective_scorer=mock_objective_scorer,
        )

        # The scorer is stored in _scorer_config.objective_scorer
        assert scenario._scorer_config.objective_scorer == mock_objective_scorer

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_objectives_by_harm")
    async def test_initialization_with_custom_max_concurrency(
        self,
        mock_get_objectives,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test initialization with custom max concurrency."""
        mock_get_scorer.return_value = mock_objective_scorer
        mock_get_objectives.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
            "violence": mock_seed_groups("violence"),
            "sexual": mock_seed_groups("sexual"),
            "harassment": mock_seed_groups("harassment"),
            "misinformation": mock_seed_groups("misinformation"),
            "leakage": mock_seed_groups("leakage"),
        }

        scenario = ContentHarmsScenario(adversarial_chat=mock_adversarial_target)

        await scenario.initialize_async(objective_target=mock_objective_target, max_concurrency=10)

        assert scenario._max_concurrency == 10

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_objectives_by_harm")
    async def test_initialization_with_custom_dataset_path(
        self,
        mock_get_objectives,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test initialization with custom seed dataset prefix."""
        mock_get_scorer.return_value = mock_objective_scorer
        mock_get_objectives.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
            "violence": mock_seed_groups("violence"),
            "sexual": mock_seed_groups("sexual"),
            "harassment": mock_seed_groups("harassment"),
            "misinformation": mock_seed_groups("misinformation"),
            "leakage": mock_seed_groups("leakage"),
        }

        scenario = ContentHarmsScenario(adversarial_chat=mock_adversarial_target)

        await scenario.initialize_async(objective_target=mock_objective_target)

        # Just verify it initializes without error
        assert scenario is not None

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_objectives_by_harm")
    async def test_initialization_defaults_to_all_strategy(
        self,
        mock_get_objectives,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test that initialization defaults to ALL strategy when none provided."""
        mock_get_scorer.return_value = mock_objective_scorer
        mock_get_objectives.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
            "violence": mock_seed_groups("violence"),
            "sexual": mock_seed_groups("sexual"),
            "harassment": mock_seed_groups("harassment"),
            "misinformation": mock_seed_groups("misinformation"),
            "leakage": mock_seed_groups("leakage"),
        }

        scenario = ContentHarmsScenario(adversarial_chat=mock_adversarial_target)

        await scenario.initialize_async(objective_target=mock_objective_target)

        # Should have strategies from the ALL aggregate
        assert len(scenario._scenario_composites) > 0

    def test_get_default_strategy_returns_all(self):
        """Test that get_default_strategy returns ALL strategy."""
        assert ContentHarmsScenario.get_default_strategy() == ContentHarmsStrategy.ALL

    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.endpoint",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test_key",
        },
    )
    def test_get_default_adversarial_target(self, mock_objective_target):
        """Test default adversarial target creation."""
        scenario = ContentHarmsScenario()

        assert scenario._adversarial_chat is not None

    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.endpoint",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test_key",
        },
    )
    def test_get_default_scorer(self, mock_objective_target):
        """Test default scorer creation."""
        scenario = ContentHarmsScenario()

        assert scenario._objective_scorer is not None

    def test_scenario_version(self):
        """Test that scenario has correct version."""
        assert ContentHarmsScenario.version == 1

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_objectives_by_harm")
    async def test_initialization_with_max_retries(
        self,
        mock_get_objectives,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test initialization with max_retries parameter."""
        mock_get_scorer.return_value = mock_objective_scorer
        mock_get_objectives.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
            "violence": mock_seed_groups("violence"),
            "sexual": mock_seed_groups("sexual"),
            "harassment": mock_seed_groups("harassment"),
            "misinformation": mock_seed_groups("misinformation"),
            "leakage": mock_seed_groups("leakage"),
        }

        scenario = ContentHarmsScenario(adversarial_chat=mock_adversarial_target)

        await scenario.initialize_async(objective_target=mock_objective_target, max_retries=3)

        assert scenario._max_retries == 3

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_objectives_by_harm")
    async def test_memory_labels_are_stored(
        self,
        mock_get_objectives,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test that memory labels are properly stored."""
        mock_get_scorer.return_value = mock_objective_scorer
        mock_get_objectives.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
            "violence": mock_seed_groups("violence"),
            "sexual": mock_seed_groups("sexual"),
            "harassment": mock_seed_groups("harassment"),
            "misinformation": mock_seed_groups("misinformation"),
            "leakage": mock_seed_groups("leakage"),
        }

        memory_labels = {"test_run": "123", "category": "harm"}

        scenario = ContentHarmsScenario(adversarial_chat=mock_adversarial_target)

        await scenario.initialize_async(objective_target=mock_objective_target, memory_labels=memory_labels)

        assert scenario._memory_labels == memory_labels

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_objectives_by_harm")
    async def test_initialization_with_all_parameters(
        self,
        mock_get_objectives,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test initialization with all possible parameters."""
        mock_get_objectives.return_value = {
            "hate": mock_seed_groups("hate"),
            "violence": mock_seed_groups("violence"),
        }

        memory_labels = {"test": "value"}
        strategies = [ContentHarmsStrategy.Hate, ContentHarmsStrategy.Violence]

        scenario = ContentHarmsScenario(
            adversarial_chat=mock_adversarial_target,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=strategies,
            memory_labels=memory_labels,
            max_concurrency=5,
            max_retries=2,
        )

        assert scenario._objective_target == mock_objective_target
        assert scenario._adversarial_chat == mock_adversarial_target
        assert scenario._scorer_config.objective_scorer == mock_objective_scorer
        assert scenario._memory_labels == memory_labels
        assert scenario._max_concurrency == 5
        assert scenario._max_retries == 2

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harms_scenario.ContentHarmsScenario._get_objectives_by_harm")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.endpoint",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test_key",
        },
    )
    async def test_initialization_with_objectives_by_harm(
        self, mock_get_objectives, mock_objective_target, mock_adversarial_target, mock_seed_groups
    ):
        """Test initialization with custom objectives_by_harm parameter."""
        # Setup custom objectives by harm
        custom_objectives = {
            "hate": mock_seed_groups("hate"),
            "violence": mock_seed_groups("violence"),
        }

        mock_get_objectives.return_value = custom_objectives

        scenario = ContentHarmsScenario(
            adversarial_chat=mock_adversarial_target,
            objectives_by_harm=custom_objectives,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=[ContentHarmsStrategy.Hate, ContentHarmsStrategy.Violence],
        )

        # Verify the objectives_by_harm is stored
        assert scenario._objectives_by_harm == custom_objectives
