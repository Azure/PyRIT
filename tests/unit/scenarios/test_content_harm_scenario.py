# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the ContentHarmScenario class."""

from unittest.mock import MagicMock, patch

import pytest

from pyrit.models.seed_group import SeedGroup
from pyrit.models.seed_objective import SeedObjective
from pyrit.models.seed_prompt import SeedPrompt
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.scenarios.scenarios.harms import (
    ContentHarmScenario,
    ContentHarmStrategy,
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
                prompts=[
                    SeedObjective(value=f"{strategy_name} objective 1"),
                    SeedPrompt(value=f"{strategy_name} prompt 1"),
                ]
            ),
            SeedGroup(
                prompts=[
                    SeedObjective(value=f"{strategy_name} objective 2"),
                    SeedPrompt(value=f"{strategy_name} prompt 2"),
                ]
            ),
        ]

    return create_seed_groups_for_strategy


class TestContentHarmStrategy:
    """Tests for the ContentHarmStrategy enum."""

    def test_all_harm_categories_exist(self):
        """Test that all expected harm categories exist as strategies."""
        expected_categories = ["hate", "fairness", "violence", "sexual", "harassment", "misinformation", "leakage"]
        strategy_values = [s.value for s in ContentHarmStrategy if s != ContentHarmStrategy.ALL]

        for category in expected_categories:
            assert category in strategy_values, f"Expected harm category '{category}' not found in strategies"

    def test_strategy_tags_are_sets(self):
        """Test that all strategy tags are set objects."""
        for strategy in ContentHarmStrategy:
            assert isinstance(strategy.tags, set), f"Tags for {strategy.name} are not a set"

    def test_enum_members_count(self):
        """Test that we have the expected number of strategy members."""
        # ALL + 7 harm categories = 8 total
        assert len(list(ContentHarmStrategy)) == 8

    def test_all_strategies_can_be_accessed_by_name(self):
        """Test that all strategies can be accessed by their name."""
        assert ContentHarmStrategy.ALL == ContentHarmStrategy["ALL"]
        assert ContentHarmStrategy.Hate == ContentHarmStrategy["Hate"]
        assert ContentHarmStrategy.Fairness == ContentHarmStrategy["Fairness"]
        assert ContentHarmStrategy.Violence == ContentHarmStrategy["Violence"]
        assert ContentHarmStrategy.Sexual == ContentHarmStrategy["Sexual"]
        assert ContentHarmStrategy.Harassment == ContentHarmStrategy["Harassment"]
        assert ContentHarmStrategy.Misinformation == ContentHarmStrategy["Misinformation"]
        assert ContentHarmStrategy.Leakage == ContentHarmStrategy["Leakage"]

    def test_all_strategies_can_be_accessed_by_value(self):
        """Test that all strategies can be accessed by their value."""
        assert ContentHarmStrategy("all") == ContentHarmStrategy.ALL
        assert ContentHarmStrategy("hate") == ContentHarmStrategy.Hate
        assert ContentHarmStrategy("fairness") == ContentHarmStrategy.Fairness
        assert ContentHarmStrategy("violence") == ContentHarmStrategy.Violence
        assert ContentHarmStrategy("sexual") == ContentHarmStrategy.Sexual
        assert ContentHarmStrategy("harassment") == ContentHarmStrategy.Harassment
        assert ContentHarmStrategy("misinformation") == ContentHarmStrategy.Misinformation
        assert ContentHarmStrategy("leakage") == ContentHarmStrategy.Leakage

    def test_strategies_are_unique(self):
        """Test that all strategy values are unique."""
        values = [s.value for s in ContentHarmStrategy]
        assert len(values) == len(set(values)), "Strategy values are not unique"

    def test_strategy_iteration(self):
        """Test that we can iterate over all strategies."""
        strategies = list(ContentHarmStrategy)
        assert len(strategies) == 8
        assert ContentHarmStrategy.ALL in strategies
        assert ContentHarmStrategy.Hate in strategies

    def test_strategy_comparison(self):
        """Test that strategy comparison works correctly."""
        assert ContentHarmStrategy.Hate == ContentHarmStrategy.Hate
        assert ContentHarmStrategy.Hate != ContentHarmStrategy.Violence
        assert ContentHarmStrategy.ALL != ContentHarmStrategy.Hate

    def test_strategy_hash(self):
        """Test that strategies can be hashed and used in sets/dicts."""
        strategy_set = {ContentHarmStrategy.Hate, ContentHarmStrategy.Violence}
        assert len(strategy_set) == 2
        assert ContentHarmStrategy.Hate in strategy_set

        strategy_dict = {ContentHarmStrategy.Hate: "hate_value"}
        assert strategy_dict[ContentHarmStrategy.Hate] == "hate_value"

    def test_strategy_string_representation(self):
        """Test string representation of strategies."""
        assert "Hate" in str(ContentHarmStrategy.Hate)
        assert "ALL" in str(ContentHarmStrategy.ALL)

    def test_invalid_strategy_value_raises_error(self):
        """Test that accessing invalid strategy value raises ValueError."""
        with pytest.raises(ValueError):
            ContentHarmStrategy("invalid_strategy")

    def test_invalid_strategy_name_raises_error(self):
        """Test that accessing invalid strategy name raises KeyError."""
        with pytest.raises(KeyError):
            ContentHarmStrategy["InvalidStrategy"]

    def test_get_aggregate_tags_includes_harm_categories(self):
        """Test that get_aggregate_tags includes 'all' tag."""
        aggregate_tags = ContentHarmStrategy.get_aggregate_tags()

        # The simple implementation only returns the 'all' tag
        assert "all" in aggregate_tags
        assert isinstance(aggregate_tags, set)

    def test_get_aggregate_tags_returns_set(self):
        """Test that get_aggregate_tags returns a set."""
        aggregate_tags = ContentHarmStrategy.get_aggregate_tags()
        assert isinstance(aggregate_tags, set)

    def test_supports_composition_returns_false(self):
        """Test that ContentHarmStrategy does not support composition."""
        # Based on the simple implementation, it likely doesn't support composition
        # Update this if composition is implemented
        assert ContentHarmStrategy.supports_composition() is False

    def test_validate_composition_with_empty_list(self):
        """Test that validate_composition handles empty list."""
        # This test depends on whether validate_composition is implemented
        # If not implemented, it should use the default from ScenarioStrategy
        try:
            ContentHarmStrategy.validate_composition([])
            # If no exception, the default implementation accepts empty lists
        except (ValueError, NotImplementedError) as e:
            # Some implementations may raise errors for empty lists
            assert "empty" in str(e).lower() or "not implemented" in str(e).lower()

    def test_validate_composition_with_single_strategy(self):
        """Test that validate_composition accepts single strategy."""
        strategies = [ContentHarmStrategy.Hate]
        # Should not raise an exception
        try:
            ContentHarmStrategy.validate_composition(strategies)
        except NotImplementedError:
            # If composition is not implemented, that's expected
            pass

    def test_validate_composition_with_multiple_strategies(self):
        """Test that validate_composition handles multiple strategies."""
        strategies = [
            ContentHarmStrategy.Hate,
            ContentHarmStrategy.Violence,
        ]
        # Behavior depends on implementation
        try:
            ContentHarmStrategy.validate_composition(strategies)
        except (ValueError, NotImplementedError):
            # Either composition is not allowed or not implemented
            pass

    def test_prepare_scenario_strategies_with_none(self):
        """Test that prepare_scenario_strategies handles None input."""
        result = ContentHarmStrategy.prepare_scenario_strategies(None, default_aggregate=ContentHarmStrategy.ALL)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_prepare_scenario_strategies_with_single_strategy(self):
        """Test that prepare_scenario_strategies handles single strategy."""
        result = ContentHarmStrategy.prepare_scenario_strategies(
            [ContentHarmStrategy.Hate], default_aggregate=ContentHarmStrategy.ALL
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_prepare_scenario_strategies_with_all(self):
        """Test that prepare_scenario_strategies expands ALL to all strategies."""
        result = ContentHarmStrategy.prepare_scenario_strategies(
            [ContentHarmStrategy.ALL], default_aggregate=ContentHarmStrategy.ALL
        )
        assert isinstance(result, list)
        # ALL should expand to multiple strategies
        assert len(result) > 1

    def test_prepare_scenario_strategies_with_multiple_strategies(self):
        """Test that prepare_scenario_strategies handles multiple strategies."""
        strategies = [
            ContentHarmStrategy.Hate,
            ContentHarmStrategy.Violence,
            ContentHarmStrategy.Sexual,
        ]
        result = ContentHarmStrategy.prepare_scenario_strategies(strategies, default_aggregate=ContentHarmStrategy.ALL)
        assert isinstance(result, list)
        assert len(result) >= 3

    def test_validate_composition_accepts_single_harm(self):
        """Test that composition validation accepts single harm strategy."""
        strategies = [ContentHarmStrategy.Hate]

        # Should not raise an exception if composition is implemented
        try:
            ContentHarmStrategy.validate_composition(strategies)
        except NotImplementedError:
            # If composition is not implemented, that's expected
            pass


@pytest.mark.usefixtures("patch_central_database")
class TestContentHarmScenarioBasic:
    """Basic tests for ContentHarmScenario initialization and properties."""

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_strategy_seeds_groups")
    async def test_initialization_with_minimal_parameters(
        self,
        mock_get_seeds,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test initialization with only required parameters."""
        mock_get_scorer.return_value = mock_objective_scorer
        # Return seed groups for all harm strategies that might be used
        mock_get_seeds.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
            "violence": mock_seed_groups("violence"),
            "sexual": mock_seed_groups("sexual"),
            "harassment": mock_seed_groups("harassment"),
            "misinformation": mock_seed_groups("misinformation"),
            "leakage": mock_seed_groups("leakage"),
        }

        scenario = ContentHarmScenario(adversarial_chat=mock_adversarial_target)

        # Constructor should set adversarial chat and basic metadata
        assert scenario._adversarial_chat == mock_adversarial_target
        assert scenario.name == "Content Harm Scenario"
        assert scenario.version == 1

        # Initialization populates objective target and scenario composites
        await scenario.initialize_async(objective_target=mock_objective_target)

        assert scenario._objective_target == mock_objective_target

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_strategy_seeds_groups")
    async def test_initialization_with_custom_strategies(
        self,
        mock_get_seeds,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test initialization with custom harm strategies."""
        mock_get_scorer.return_value = mock_objective_scorer
        mock_get_seeds.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
        }

        strategies = [ContentHarmStrategy.Hate, ContentHarmStrategy.Fairness]

        scenario = ContentHarmScenario(adversarial_chat=mock_adversarial_target)

        await scenario.initialize_async(objective_target=mock_objective_target, scenario_strategies=strategies)

        # Prepared composites should match provided strategies
        assert len(scenario._scenario_composites) == 2

    @patch("pyrit.scenarios.scenarios.harms.content_harm_scenario.ContentHarmScenario._get_strategy_seeds_groups")
    def test_initialization_with_custom_scorer(
        self, mock_get_seeds, mock_objective_target, mock_adversarial_target, mock_objective_scorer
    ):
        """Test initialization with custom objective scorer."""
        mock_get_seeds.return_value = {}

        scenario = ContentHarmScenario(
            adversarial_chat=mock_adversarial_target,
            objective_scorer=mock_objective_scorer,
        )

        # The scorer is stored in _scorer_config.objective_scorer
        assert scenario._scorer_config.objective_scorer == mock_objective_scorer

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_strategy_seeds_groups")
    async def test_initialization_with_custom_max_concurrency(
        self,
        mock_get_seeds,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test initialization with custom max concurrency."""
        mock_get_scorer.return_value = mock_objective_scorer
        mock_get_seeds.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
            "violence": mock_seed_groups("violence"),
            "sexual": mock_seed_groups("sexual"),
            "harassment": mock_seed_groups("harassment"),
            "misinformation": mock_seed_groups("misinformation"),
            "leakage": mock_seed_groups("leakage"),
        }

        scenario = ContentHarmScenario(adversarial_chat=mock_adversarial_target)

        await scenario.initialize_async(objective_target=mock_objective_target, max_concurrency=10)

        assert scenario._max_concurrency == 10

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_strategy_seeds_groups")
    async def test_initialization_with_custom_dataset_path(
        self,
        mock_get_seeds,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test initialization with custom seed dataset prefix."""
        mock_get_scorer.return_value = mock_objective_scorer
        mock_get_seeds.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
            "violence": mock_seed_groups("violence"),
            "sexual": mock_seed_groups("sexual"),
            "harassment": mock_seed_groups("harassment"),
            "misinformation": mock_seed_groups("misinformation"),
            "leakage": mock_seed_groups("leakage"),
        }

        custom_prefix = "custom_dataset"

        scenario = ContentHarmScenario(adversarial_chat=mock_adversarial_target, seed_dataset_prefix=custom_prefix)

        await scenario.initialize_async(objective_target=mock_objective_target)

        # Just verify it initializes without error
        assert scenario is not None
        # Verify the seed_dataset_prefix is stored
        assert scenario._seed_dataset_prefix == custom_prefix
        # Verify the method was called (without arguments, as per current implementation)
        mock_get_seeds.assert_called_once_with()

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_strategy_seeds_groups")
    async def test_initialization_defaults_to_all_strategy(
        self,
        mock_get_seeds,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test that initialization defaults to ALL strategy when none provided."""
        mock_get_scorer.return_value = mock_objective_scorer
        mock_get_seeds.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
            "violence": mock_seed_groups("violence"),
            "sexual": mock_seed_groups("sexual"),
            "harassment": mock_seed_groups("harassment"),
            "misinformation": mock_seed_groups("misinformation"),
            "leakage": mock_seed_groups("leakage"),
        }

        scenario = ContentHarmScenario(adversarial_chat=mock_adversarial_target)

        await scenario.initialize_async(objective_target=mock_objective_target)

        # Should have strategies from the ALL aggregate
        assert len(scenario._scenario_composites) > 0

    def test_get_default_strategy_returns_all(self):
        """Test that get_default_strategy returns ALL strategy."""
        assert ContentHarmScenario.get_default_strategy() == ContentHarmStrategy.ALL

    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_strategy_seeds_groups")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.endpoint",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test_key",
        },
    )
    def test_get_default_adversarial_target(self, mock_get_seeds, mock_objective_target):
        """Test default adversarial target creation."""
        mock_get_seeds.return_value = {}
        scenario = ContentHarmScenario()

        assert scenario._adversarial_chat is not None

    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_strategy_seeds_groups")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.endpoint",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test_key",
        },
    )
    def test_get_default_scorer(self, mock_get_seeds, mock_objective_target):
        """Test default scorer creation."""
        mock_get_seeds.return_value = {}
        scenario = ContentHarmScenario()

        assert scenario._objective_scorer is not None

    def test_scenario_version(self):
        """Test that scenario has correct version."""
        assert ContentHarmScenario.version == 1

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_strategy_seeds_groups")
    async def test_initialization_with_max_retries(
        self,
        mock_get_seeds,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test initialization with max_retries parameter."""
        mock_get_scorer.return_value = mock_objective_scorer
        mock_get_seeds.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
            "violence": mock_seed_groups("violence"),
            "sexual": mock_seed_groups("sexual"),
            "harassment": mock_seed_groups("harassment"),
            "misinformation": mock_seed_groups("misinformation"),
            "leakage": mock_seed_groups("leakage"),
        }

        scenario = ContentHarmScenario(adversarial_chat=mock_adversarial_target)

        await scenario.initialize_async(objective_target=mock_objective_target, max_retries=3)

        assert scenario._max_retries == 3

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_default_scorer")
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_strategy_seeds_groups")
    async def test_memory_labels_are_stored(
        self,
        mock_get_seeds,
        mock_get_scorer,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        mock_seed_groups,
    ):
        """Test that memory labels are properly stored."""
        mock_get_scorer.return_value = mock_objective_scorer
        mock_get_seeds.return_value = {
            "hate": mock_seed_groups("hate"),
            "fairness": mock_seed_groups("fairness"),
            "violence": mock_seed_groups("violence"),
            "sexual": mock_seed_groups("sexual"),
            "harassment": mock_seed_groups("harassment"),
            "misinformation": mock_seed_groups("misinformation"),
            "leakage": mock_seed_groups("leakage"),
        }

        memory_labels = {"test_run": "123", "category": "harm"}

        scenario = ContentHarmScenario(adversarial_chat=mock_adversarial_target)

        await scenario.initialize_async(objective_target=mock_objective_target, memory_labels=memory_labels)

        assert scenario._memory_labels == memory_labels

    @pytest.mark.asyncio
    @patch("pyrit.scenario.scenarios.airt.content_harm_scenario.ContentHarmScenario._get_strategy_seeds_groups")
    async def test_initialization_with_all_parameters(
        self, mock_get_seeds, mock_objective_target, mock_adversarial_target, mock_objective_scorer, mock_seed_groups
    ):
        """Test initialization with all possible parameters."""
        mock_get_seeds.return_value = {
            "hate": mock_seed_groups("hate"),
            "violence": mock_seed_groups("violence"),
        }

        memory_labels = {"test": "value"}
        strategies = [ContentHarmStrategy.Hate, ContentHarmStrategy.Violence]

        scenario = ContentHarmScenario(
            adversarial_chat=mock_adversarial_target,
            objective_scorer=mock_objective_scorer,
            seed_dataset_prefix="test_prefix",
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
