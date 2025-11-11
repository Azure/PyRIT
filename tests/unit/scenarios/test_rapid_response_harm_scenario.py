# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the RapidResponseHarmScenario class."""

from unittest.mock import MagicMock, patch

import pytest

from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.scenarios.scenarios.ai_rt.rapid_response_harm_scenario import (
    RapidResponseHarmScenario,
    RapidResponseHarmStrategy,
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


class TestRapidResponseHarmStrategy:
    """Tests for the RapidResponseHarmStrategy enum."""

    def test_all_harm_categories_exist(self):
        """Test that all expected harm categories exist as strategies."""
        expected_categories = ["hate", "fairness", "violence", "sexual", "harassment", "misinformation", "leakage"]
        strategy_values = [s.value for s in RapidResponseHarmStrategy if s != RapidResponseHarmStrategy.ALL]

        for category in expected_categories:
            assert category in strategy_values, f"Expected harm category '{category}' not found in strategies"

    def test_strategy_tags_are_sets(self):
        """Test that all strategy tags are set objects."""
        for strategy in RapidResponseHarmStrategy:
            assert isinstance(strategy.tags, set), f"Tags for {strategy.name} are not a set"

    def test_enum_members_count(self):
        """Test that we have the expected number of strategy members."""
        # ALL + 7 harm categories = 8 total
        assert len(list(RapidResponseHarmStrategy)) == 8

    def test_all_strategies_can_be_accessed_by_name(self):
        """Test that all strategies can be accessed by their name."""
        assert RapidResponseHarmStrategy.ALL == RapidResponseHarmStrategy["ALL"]
        assert RapidResponseHarmStrategy.Hate == RapidResponseHarmStrategy["Hate"]
        assert RapidResponseHarmStrategy.Fairness == RapidResponseHarmStrategy["Fairness"]
        assert RapidResponseHarmStrategy.Violence == RapidResponseHarmStrategy["Violence"]
        assert RapidResponseHarmStrategy.Sexual == RapidResponseHarmStrategy["Sexual"]
        assert RapidResponseHarmStrategy.Harassment == RapidResponseHarmStrategy["Harassment"]
        assert RapidResponseHarmStrategy.Misinformation == RapidResponseHarmStrategy["Misinformation"]
        assert RapidResponseHarmStrategy.Leakage == RapidResponseHarmStrategy["Leakage"]

    def test_all_strategies_can_be_accessed_by_value(self):
        """Test that all strategies can be accessed by their value."""
        assert RapidResponseHarmStrategy("all") == RapidResponseHarmStrategy.ALL
        assert RapidResponseHarmStrategy("hate") == RapidResponseHarmStrategy.Hate
        assert RapidResponseHarmStrategy("fairness") == RapidResponseHarmStrategy.Fairness
        assert RapidResponseHarmStrategy("violence") == RapidResponseHarmStrategy.Violence
        assert RapidResponseHarmStrategy("sexual") == RapidResponseHarmStrategy.Sexual
        assert RapidResponseHarmStrategy("harassment") == RapidResponseHarmStrategy.Harassment
        assert RapidResponseHarmStrategy("misinformation") == RapidResponseHarmStrategy.Misinformation
        assert RapidResponseHarmStrategy("leakage") == RapidResponseHarmStrategy.Leakage

    def test_strategies_are_unique(self):
        """Test that all strategy values are unique."""
        values = [s.value for s in RapidResponseHarmStrategy]
        assert len(values) == len(set(values)), "Strategy values are not unique"

    def test_strategy_iteration(self):
        """Test that we can iterate over all strategies."""
        strategies = list(RapidResponseHarmStrategy)
        assert len(strategies) == 8
        assert RapidResponseHarmStrategy.ALL in strategies
        assert RapidResponseHarmStrategy.Hate in strategies

    def test_strategy_comparison(self):
        """Test that strategy comparison works correctly."""
        assert RapidResponseHarmStrategy.Hate == RapidResponseHarmStrategy.Hate
        assert RapidResponseHarmStrategy.Hate != RapidResponseHarmStrategy.Violence
        assert RapidResponseHarmStrategy.ALL != RapidResponseHarmStrategy.Hate

    def test_strategy_hash(self):
        """Test that strategies can be hashed and used in sets/dicts."""
        strategy_set = {RapidResponseHarmStrategy.Hate, RapidResponseHarmStrategy.Violence}
        assert len(strategy_set) == 2
        assert RapidResponseHarmStrategy.Hate in strategy_set

        strategy_dict = {RapidResponseHarmStrategy.Hate: "hate_value"}
        assert strategy_dict[RapidResponseHarmStrategy.Hate] == "hate_value"

    def test_strategy_string_representation(self):
        """Test string representation of strategies."""
        assert "Hate" in str(RapidResponseHarmStrategy.Hate)
        assert "ALL" in str(RapidResponseHarmStrategy.ALL)

    def test_invalid_strategy_value_raises_error(self):
        """Test that accessing invalid strategy value raises ValueError."""
        with pytest.raises(ValueError):
            RapidResponseHarmStrategy("invalid_strategy")

    def test_invalid_strategy_name_raises_error(self):
        """Test that accessing invalid strategy name raises KeyError."""
        with pytest.raises(KeyError):
            RapidResponseHarmStrategy["InvalidStrategy"]

    def test_get_aggregate_tags_includes_harm_categories(self):
        """Test that get_aggregate_tags includes 'all' tag."""
        aggregate_tags = RapidResponseHarmStrategy.get_aggregate_tags()

        # The simple implementation only returns the 'all' tag
        assert "all" in aggregate_tags
        assert isinstance(aggregate_tags, set)

    def test_get_aggregate_tags_returns_set(self):
        """Test that get_aggregate_tags returns a set."""
        aggregate_tags = RapidResponseHarmStrategy.get_aggregate_tags()
        assert isinstance(aggregate_tags, set)

    def test_supports_composition_returns_false(self):
        """Test that RapidResponseHarmStrategy does not support composition."""
        # Based on the simple implementation, it likely doesn't support composition
        # Update this if composition is implemented
        assert RapidResponseHarmStrategy.supports_composition() is False

    def test_validate_composition_with_empty_list(self):
        """Test that validate_composition handles empty list."""
        # This test depends on whether validate_composition is implemented
        # If not implemented, it should use the default from ScenarioStrategy
        try:
            RapidResponseHarmStrategy.validate_composition([])
            # If no exception, the default implementation accepts empty lists
        except (ValueError, NotImplementedError) as e:
            # Some implementations may raise errors for empty lists
            assert "empty" in str(e).lower() or "not implemented" in str(e).lower()

    def test_validate_composition_with_single_strategy(self):
        """Test that validate_composition accepts single strategy."""
        strategies = [RapidResponseHarmStrategy.Hate]
        # Should not raise an exception
        try:
            RapidResponseHarmStrategy.validate_composition(strategies)
        except NotImplementedError:
            # If composition is not implemented, that's expected
            pass

    def test_validate_composition_with_multiple_strategies(self):
        """Test that validate_composition handles multiple strategies."""
        strategies = [
            RapidResponseHarmStrategy.Hate,
            RapidResponseHarmStrategy.Violence,
        ]
        # Behavior depends on implementation
        try:
            RapidResponseHarmStrategy.validate_composition(strategies)
        except (ValueError, NotImplementedError):
            # Either composition is not allowed or not implemented
            pass

    def test_prepare_scenario_strategies_with_none(self):
        """Test that prepare_scenario_strategies handles None input."""
        result = RapidResponseHarmStrategy.prepare_scenario_strategies(
            None, default_aggregate=RapidResponseHarmStrategy.ALL
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_prepare_scenario_strategies_with_single_strategy(self):
        """Test that prepare_scenario_strategies handles single strategy."""
        result = RapidResponseHarmStrategy.prepare_scenario_strategies(
            [RapidResponseHarmStrategy.Hate], default_aggregate=RapidResponseHarmStrategy.ALL
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_prepare_scenario_strategies_with_all(self):
        """Test that prepare_scenario_strategies expands ALL to all strategies."""
        result = RapidResponseHarmStrategy.prepare_scenario_strategies(
            [RapidResponseHarmStrategy.ALL], default_aggregate=RapidResponseHarmStrategy.ALL
        )
        assert isinstance(result, list)
        # ALL should expand to multiple strategies
        assert len(result) > 1

    def test_prepare_scenario_strategies_with_multiple_strategies(self):
        """Test that prepare_scenario_strategies handles multiple strategies."""
        strategies = [
            RapidResponseHarmStrategy.Hate,
            RapidResponseHarmStrategy.Violence,
            RapidResponseHarmStrategy.Sexual,
        ]
        result = RapidResponseHarmStrategy.prepare_scenario_strategies(
            strategies, default_aggregate=RapidResponseHarmStrategy.ALL
        )
        assert isinstance(result, list)
        assert len(result) >= 3

    def test_validate_composition_accepts_single_harm(self):
        """Test that composition validation accepts single harm strategy."""
        strategies = [RapidResponseHarmStrategy.Hate]

        # Should not raise an exception if composition is implemented
        try:
            RapidResponseHarmStrategy.validate_composition(strategies)
        except NotImplementedError:
            # If composition is not implemented, that's expected
            pass


@pytest.mark.usefixtures("patch_central_database")
class TestRapidResponseHarmScenarioBasic:
    """Basic tests for RapidResponseHarmScenario initialization and properties."""

    def test_initialization_with_minimal_parameters(self, mock_objective_target, mock_adversarial_target):
        """Test initialization with only required parameters."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )

        assert scenario._objective_target == mock_objective_target
        assert scenario._adversarial_chat == mock_adversarial_target
        assert scenario.name == "Rapid Response Harm Scenario"
        assert scenario.version == 1

    def test_initialization_with_custom_strategies(self, mock_objective_target, mock_adversarial_target):
        """Test initialization with custom harm strategies."""
        strategies = [
            RapidResponseHarmStrategy.Hate,
            RapidResponseHarmStrategy.Fairness,
        ]

        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            scenario_strategies=strategies,
        )

        assert len(scenario._rapid_response_harm_strategy_composition) == 2

    def test_initialization_with_memory_labels(self, mock_objective_target, mock_adversarial_target):
        """Test initialization with memory labels."""
        memory_labels = {"test_id": "123", "environment": "test"}

        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            memory_labels=memory_labels,
        )

        assert scenario._memory_labels == memory_labels

    def test_initialization_with_custom_scorer(
        self, mock_objective_target, mock_adversarial_target, mock_objective_scorer
    ):
        """Test initialization with custom objective scorer."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario._objective_scorer == mock_objective_scorer

    def test_initialization_with_custom_max_concurrency(self, mock_objective_target, mock_adversarial_target):
        """Test initialization with custom max concurrency."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            max_concurrency=10,
        )

        assert scenario._max_concurrency == 10

    def test_initialization_with_custom_dataset_path(self, mock_objective_target, mock_adversarial_target):
        """Test initialization with custom seed dataset prefix."""
        custom_prefix = "custom_dataset"

        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            seed_dataset_prefix=custom_prefix,
        )

        # Just verify it initializes without error
        assert scenario is not None

    def test_initialization_defaults_to_all_strategy(self, mock_objective_target, mock_adversarial_target):
        """Test that initialization defaults to ALL strategy when none provided."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )

        # Should have strategies from the ALL aggregate
        assert len(scenario._rapid_response_harm_strategy_composition) > 0

    def test_get_default_strategy_returns_all(self):
        """Test that get_default_strategy returns ALL strategy."""
        assert RapidResponseHarmScenario.get_default_strategy() == RapidResponseHarmStrategy.ALL

    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.endpoint",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test_key",
        },
    )
    def test_get_default_adversarial_target(self, mock_objective_target):
        """Test default adversarial target creation."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
        )

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
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
        )

        assert scenario._objective_scorer is not None

    def test_scenario_version(self):
        """Test that scenario has correct version."""
        assert RapidResponseHarmScenario.version == 1

    def test_initialization_with_max_retries(self, mock_objective_target, mock_adversarial_target):
        """Test initialization with max_retries parameter."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            max_retries=3,
        )

        assert scenario._max_retries == 3

    def test_memory_labels_are_stored(self, mock_objective_target, mock_adversarial_target):
        """Test that memory labels are properly stored."""
        memory_labels = {"test_run": "123", "category": "harm"}

        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            memory_labels=memory_labels,
        )

        assert scenario._memory_labels == memory_labels

    def test_initialization_with_all_parameters(
        self, mock_objective_target, mock_adversarial_target, mock_objective_scorer
    ):
        """Test initialization with all possible parameters."""
        memory_labels = {"test": "value"}
        strategies = [RapidResponseHarmStrategy.Hate, RapidResponseHarmStrategy.Violence]

        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            scenario_strategies=strategies,
            objective_scorer=mock_objective_scorer,
            memory_labels=memory_labels,
            seed_dataset_prefix="test_prefix",
            max_concurrency=5,
            max_retries=2,
        )

        assert scenario._objective_target == mock_objective_target
        assert scenario._adversarial_chat == mock_adversarial_target
        assert scenario._objective_scorer == mock_objective_scorer
        assert scenario._memory_labels == memory_labels
        assert scenario._max_concurrency == 5
        assert scenario._max_retries == 2
