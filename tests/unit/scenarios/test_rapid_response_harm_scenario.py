# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the RapidResponseHarmScenario class."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from pyrit.executor.attack import CrescendoAttack, MultiPromptSendingAttack, PromptSendingAttack
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.scenarios import AtomicAttack, ScenarioCompositeStrategy
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

    def test_strategy_has_correct_tags(self):
        """Test that strategies have correct tags."""
        # Test aggregate tags
        assert "all" in RapidResponseHarmStrategy.ALL.tags
        assert "hate" in RapidResponseHarmStrategy.HATE.tags
        assert "fairness" in RapidResponseHarmStrategy.FAIRNESS.tags
        assert "violence" in RapidResponseHarmStrategy.VIOLENCE.tags

        # Test harm-specific strategies
        assert "hate" in RapidResponseHarmStrategy.HateFictionalStory.tags
        assert "harm" in RapidResponseHarmStrategy.HateFictionalStory.tags
        assert "fairness" in RapidResponseHarmStrategy.FairnessEthnicityInference.tags
        assert "harm" in RapidResponseHarmStrategy.FairnessEthnicityInference.tags

        # Test attack strategies
        assert "attack" in RapidResponseHarmStrategy.MultiTurn.tags
        assert "attack" in RapidResponseHarmStrategy.Crescendo.tags

    def test_get_aggregate_tags_includes_harm_categories(self):
        """Test that get_aggregate_tags includes all harm categories."""
        aggregate_tags = RapidResponseHarmStrategy.get_aggregate_tags()
        
        expected_tags = {
            "all",
            "hate",
            "fairness",
            "violence",
            "sexual",
            "harassment",
            "misinformation",
            "leakage",
        }
        
        assert expected_tags.issubset(aggregate_tags)

    def test_supports_composition_returns_true(self):
        """Test that RapidResponseHarmStrategy supports composition."""
        assert RapidResponseHarmStrategy.supports_composition() is True

    def test_validate_composition_accepts_single_harm_single_attack(self):
        """Test that composition validation accepts one harm and one attack strategy."""
        strategies = [
            RapidResponseHarmStrategy.HateFictionalStory,
            RapidResponseHarmStrategy.MultiTurn,
        ]
        
        # Should not raise an exception
        RapidResponseHarmStrategy.validate_composition(strategies)

    def test_validate_composition_accepts_multiple_harms_without_attacks(self):
        """Test that composition validation accepts multiple harm strategies without attacks."""
        strategies = [
            RapidResponseHarmStrategy.HateFictionalStory,
            RapidResponseHarmStrategy.FairnessEthnicityInference,
        ]
        
        # Should not raise an exception
        RapidResponseHarmStrategy.validate_composition(strategies)

    def test_validate_composition_rejects_multiple_attacks(self):
        """Test that composition validation rejects multiple attack strategies."""
        strategies = [
            RapidResponseHarmStrategy.MultiTurn,
            RapidResponseHarmStrategy.Crescendo,
        ]
        
        with pytest.raises(ValueError, match="Cannot compose multiple attack strategies"):
            RapidResponseHarmStrategy.validate_composition(strategies)

    def test_validate_composition_rejects_empty_list(self):
        """Test that composition validation rejects empty strategy list."""
        with pytest.raises(ValueError, match="Cannot validate empty strategy list"):
            RapidResponseHarmStrategy.validate_composition([])

    def test_validate_composition_accepts_non_rapid_response_strategies(self):
        """Test that composition validation handles mixed strategy types."""
        # Mock a different strategy type
        mock_strategy = MagicMock()
        mock_strategy.tags = {"other"}
        
        strategies = [
            RapidResponseHarmStrategy.HateFictionalStory,
            mock_strategy,
        ]
        
        # Should not raise an exception (ignores non-RapidResponseHarmStrategy)
        RapidResponseHarmStrategy.validate_composition(strategies)


@pytest.mark.usefixtures("patch_central_database")
class TestRapidResponseHarmScenarioInitialization:
    """Tests for RapidResponseHarmScenario initialization."""

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
            RapidResponseHarmStrategy.HateFictionalStory,
            RapidResponseHarmStrategy.FairnessEthnicityInference,
        ]
        
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            scenario_strategies=strategies,
        )
        
        assert len(scenario._rapid_response_harm_strategy_compositiion) == 2

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
        """Test initialization with custom objective dataset path."""
        custom_path = "custom_dataset_path_"
        
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            objective_dataset_path=custom_path,
        )
        
        assert scenario.objective_dataset_path == custom_path

    def test_initialization_defaults_to_all_strategy(self, mock_objective_target, mock_adversarial_target):
        """Test that initialization defaults to ALL strategy when none provided."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )
        
        # Should have strategies from the ALL aggregate
        assert len(scenario._rapid_response_harm_strategy_compositiion) > 0

    def test_get_strategy_class_returns_correct_class(self):
        """Test that get_strategy_class returns RapidResponseHarmStrategy."""
        assert RapidResponseHarmScenario.get_strategy_class() == RapidResponseHarmStrategy

    def test_get_default_strategy_returns_all(self):
        """Test that get_default_strategy returns ALL strategy."""
        assert RapidResponseHarmScenario.get_default_strategy() == RapidResponseHarmStrategy.ALL

    @patch.dict("os.environ", {
        "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.endpoint",
        "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test_key"
    })
    def test_get_default_adversarial_target(self, mock_objective_target):
        """Test default adversarial target creation."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
        )
        
        assert scenario._adversarial_chat is not None

    @patch.dict("os.environ", {
        "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.endpoint",
        "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test_key"
    })
    def test_get_default_scorer(self, mock_objective_target):
        """Test default scorer creation."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
        )
        
        assert scenario._objective_scorer is not None


@pytest.mark.usefixtures("patch_central_database")
class TestRapidResponseHarmScenarioAttackCreation:
    """Tests for attack creation in RapidResponseHarmScenario."""

    def test_get_attack_creates_prompt_sending_attack(self, mock_objective_target, mock_adversarial_target):
        """Test that _get_attack creates PromptSendingAttack for default case."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )
        
        attack = scenario._get_attack(attack_type=PromptSendingAttack)
        
        assert isinstance(attack, PromptSendingAttack)

    def test_get_attack_creates_crescendo_attack(self, mock_objective_target, mock_adversarial_target):
        """Test that _get_attack creates CrescendoAttack when requested."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )
        
        attack = scenario._get_attack(attack_type=CrescendoAttack)
        
        assert isinstance(attack, CrescendoAttack)

    def test_get_attack_creates_multi_prompt_sending_attack(
        self, mock_objective_target, mock_adversarial_target
    ):
        """Test that _get_attack creates MultiPromptSendingAttack when requested."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )
        
        attack = scenario._get_attack(attack_type=MultiPromptSendingAttack)
        
        assert isinstance(attack, MultiPromptSendingAttack)

    def test_get_attack_raises_error_without_adversarial_target(self, mock_objective_target):
        """Test that _get_attack raises error for multi-turn attacks without adversarial target."""
        # Don't provide adversarial_chat
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=None,
        )
        scenario._adversarial_chat = None  # Ensure it's None
        
        with pytest.raises(ValueError, match="requires an adversarial target"):
            scenario._get_attack(attack_type=CrescendoAttack)


@pytest.mark.usefixtures("patch_central_database")
class TestRapidResponseHarmScenarioAttackFromStrategy:
    """Tests for creating atomic attacks from strategies."""

    @patch("pyrit.memory.central_memory.CentralMemory.get_seed_groups")
    def test_get_attack_from_strategy_with_hate_strategy(
        self, mock_get_seed_groups, mock_objective_target, mock_adversarial_target
    ):
        """Test creating attack from hate strategy."""
        # Mock seed groups with objectives
        mock_objective = Mock()
        mock_objective.objective.value = "Test hate objective"
        mock_get_seed_groups.return_value = [mock_objective]
        
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )
        
        composite_strategy = ScenarioCompositeStrategy(
            name="hate_test",
            strategies=[RapidResponseHarmStrategy.HateFictionalStory]
        )
        
        atomic_attack = scenario._get_attack_from_strategy(composite_strategy=composite_strategy)
        
        assert isinstance(atomic_attack, AtomicAttack)
        assert atomic_attack.atomic_attack_name == "hate_test"
        assert len(atomic_attack.objectives) == 1
        assert atomic_attack.objectives[0] == "Test hate objective"

    @patch("pyrit.memory.central_memory.CentralMemory.get_seed_groups")
    def test_get_attack_from_strategy_with_multi_turn_attack(
        self, mock_get_seed_groups, mock_objective_target, mock_adversarial_target
    ):
        """Test creating attack with MultiTurn strategy."""
        # Mock seed groups with objectives
        mock_objective = Mock()
        mock_objective.objective.value = "Test objective"
        mock_get_seed_groups.return_value = [mock_objective]
        
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )
        
        composite_strategy = ScenarioCompositeStrategy(
            name="multi_turn_test",
            strategies=[
                RapidResponseHarmStrategy.MultiTurn,
                RapidResponseHarmStrategy.HateFictionalStory
            ]
        )
        
        atomic_attack = scenario._get_attack_from_strategy(composite_strategy=composite_strategy)
        
        assert isinstance(atomic_attack.attack, MultiPromptSendingAttack)

    @patch("pyrit.memory.central_memory.CentralMemory.get_seed_groups")
    def test_get_attack_from_strategy_with_crescendo_attack(
        self, mock_get_seed_groups, mock_objective_target, mock_adversarial_target
    ):
        """Test creating attack with Crescendo strategy."""
        # Mock seed groups with objectives
        mock_objective = Mock()
        mock_objective.objective.value = "Test objective"
        mock_get_seed_groups.return_value = [mock_objective]
        
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )
        
        composite_strategy = ScenarioCompositeStrategy(
            name="crescendo_test",
            strategies=[
                RapidResponseHarmStrategy.Crescendo,
                RapidResponseHarmStrategy.HateFictionalStory
            ]
        )
        
        atomic_attack = scenario._get_attack_from_strategy(composite_strategy=composite_strategy)
        
        assert isinstance(atomic_attack.attack, CrescendoAttack)

    @patch("pyrit.memory.central_memory.CentralMemory.get_seed_groups")
    def test_get_attack_from_strategy_raises_error_with_no_harm(
        self, mock_get_seed_groups, mock_objective_target, mock_adversarial_target
    ):
        """Test that error is raised when no harm strategy is provided."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )
        
        composite_strategy = ScenarioCompositeStrategy(
            name="attack_only",
            strategies=[RapidResponseHarmStrategy.MultiTurn]
        )
        
        with pytest.raises(ValueError, match="No harm strategy found"):
            scenario._get_attack_from_strategy(composite_strategy=composite_strategy)

    @patch("pyrit.memory.central_memory.CentralMemory.get_seed_groups")
    def test_get_attack_from_strategy_raises_error_with_no_objectives(
        self, mock_get_seed_groups, mock_objective_target, mock_adversarial_target
    ):
        """Test that error is raised when no objectives are found in memory."""
        # Mock empty seed groups
        mock_get_seed_groups.return_value = []
        
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )
        
        composite_strategy = ScenarioCompositeStrategy(
            name="hate_test",
            strategies=[RapidResponseHarmStrategy.HateFictionalStory]
        )
        
        with pytest.raises(ValueError, match="No objectives found in the dataset"):
            scenario._get_attack_from_strategy(composite_strategy=composite_strategy)

    @patch("pyrit.memory.central_memory.CentralMemory.get_seed_groups")
    def test_get_attack_from_strategy_with_custom_dataset_path(
        self, mock_get_seed_groups, mock_objective_target, mock_adversarial_target
    ):
        """Test that custom dataset path is used when retrieving objectives."""
        # Mock seed groups with objectives
        mock_objective = Mock()
        mock_objective.objective.value = "Test objective"
        mock_get_seed_groups.return_value = [mock_objective]
        
        custom_path = "custom_path_"
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            objective_dataset_path=custom_path,
        )
        
        composite_strategy = ScenarioCompositeStrategy(
            name="hate_test",
            strategies=[RapidResponseHarmStrategy.HateFictionalStory]
        )
        
        scenario._get_attack_from_strategy(composite_strategy=composite_strategy)
        
        # Verify the correct dataset name was used
        expected_dataset_name = f"{custom_path}hate_fictional_story"
        mock_get_seed_groups.assert_called_once_with(dataset_name=expected_dataset_name)


@pytest.mark.usefixtures("patch_central_database")
class TestRapidResponseHarmScenarioGetAtomicAttacks:
    """Tests for getting atomic attacks list."""

    @patch("pyrit.scenarios.scenarios.ai_rt.rapid_response_harm_scenario.RapidResponseHarmScenario._get_attack_from_strategy")
    def test_get_rapid_response_harm_attacks(
        self, mock_get_attack_from_strategy, mock_objective_target, mock_adversarial_target
    ):
        """Test that _get_rapid_response_harm_attacks creates attacks for each strategy."""
        mock_atomic_attack = Mock(spec=AtomicAttack)
        mock_get_attack_from_strategy.return_value = mock_atomic_attack
        
        strategies = [
            RapidResponseHarmStrategy.HateFictionalStory,
            RapidResponseHarmStrategy.FairnessEthnicityInference,
        ]
        
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            scenario_strategies=strategies,
        )
        
        atomic_attacks = scenario._get_rapid_response_harm_attacks()
        
        assert len(atomic_attacks) == 2
        assert mock_get_attack_from_strategy.call_count == 2

    @patch("pyrit.scenarios.scenarios.ai_rt.rapid_response_harm_scenario.RapidResponseHarmScenario._get_rapid_response_harm_attacks")
    async def test_get_atomic_attacks_async_calls_harm_attacks(
        self, mock_get_harm_attacks, mock_objective_target, mock_adversarial_target
    ):
        """Test that _get_atomic_attacks_async delegates to _get_rapid_response_harm_attacks."""
        mock_atomic_attack = Mock(spec=AtomicAttack)
        mock_get_harm_attacks.return_value = [mock_atomic_attack]
        
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )
        
        result = await scenario._get_atomic_attacks_async()
        
        assert result == [mock_atomic_attack]
        mock_get_harm_attacks.assert_called_once()


@pytest.mark.usefixtures("patch_central_database")
class TestRapidResponseHarmScenarioStrategyExpansion:
    """Tests for strategy expansion and composition."""

    def test_all_strategy_expands_to_multiple_strategies(self, mock_objective_target, mock_adversarial_target):
        """Test that ALL strategy expands to include all harm strategies."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            scenario_strategies=[RapidResponseHarmStrategy.ALL],
        )
        
        # ALL should expand to multiple strategies
        assert len(scenario._rapid_response_harm_strategy_compositiion) > 1

    def test_hate_strategy_expands_to_hate_specific_strategies(
        self, mock_objective_target, mock_adversarial_target
    ):
        """Test that HATE aggregate strategy expands to hate-specific strategies."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            scenario_strategies=[RapidResponseHarmStrategy.HATE],
        )
        
        # HATE should expand to multiple hate strategies
        assert len(scenario._rapid_response_harm_strategy_compositiion) >= 1
        
        # All expanded strategies should have "hate" tag
        for composite_strategy in scenario._rapid_response_harm_strategy_compositiion:
            strategy_list = [s for s in composite_strategy.strategies if isinstance(s, RapidResponseHarmStrategy)]
            harm_tags = [s for s in strategy_list if "harm" in s.tags]
            if harm_tags:
                assert "hate" in harm_tags[0].tags

    def test_composite_strategy_with_attack_and_harm(
        self, mock_objective_target, mock_adversarial_target
    ):
        """Test that composite strategies can combine attack and harm strategies."""
        composite = ScenarioCompositeStrategy(
            name="test_composite",
            strategies=[
                RapidResponseHarmStrategy.MultiTurn,
                RapidResponseHarmStrategy.HateFictionalStory
            ]
        )
        
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            scenario_strategies=[composite],
        )
        
        assert len(scenario._rapid_response_harm_strategy_compositiion) == 1
        assert scenario._rapid_response_harm_strategy_compositiion[0].name == "test_composite"


@pytest.mark.usefixtures("patch_central_database")
class TestRapidResponseHarmScenarioEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_attack_strategy_raises_error(self, mock_objective_target, mock_adversarial_target):
        """Test that unknown attack strategy raises ValueError."""
        # Create a mock strategy with attack tag but not recognized
        mock_strategy = MagicMock(spec=RapidResponseHarmStrategy)
        mock_strategy.value = "unknown_attack"
        mock_strategy.tags = {"attack"}
        
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
        )
        
        # Mock the composite strategy with unknown attack
        composite_strategy = ScenarioCompositeStrategy(
            name="unknown_test",
            strategies=[mock_strategy, RapidResponseHarmStrategy.HateFictionalStory]
        )
        
        with pytest.raises(ValueError, match="Unknown attack strategy"):
            scenario._get_attack_from_strategy(composite_strategy=composite_strategy)

    def test_include_baseline_parameter(self, mock_objective_target, mock_adversarial_target):
        """Test that include_baseline parameter is passed correctly."""
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            include_baseline=True,
        )
        
        assert scenario._include_baseline is True

    def test_memory_labels_are_passed_to_atomic_attacks(
        self, mock_objective_target, mock_adversarial_target
    ):
        """Test that memory labels are passed to atomic attacks."""
        memory_labels = {"test_run": "123", "category": "harm"}
        
        scenario = RapidResponseHarmScenario(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_target,
            memory_labels=memory_labels,
        )
        
        assert scenario._memory_labels == memory_labels
