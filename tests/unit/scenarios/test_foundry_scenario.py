# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the FoundryScenario class."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack
from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.prompt_converter import Base64Converter
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.scenarios import AtomicAttack, FoundryAttackStrategy, FoundryScenario
from pyrit.score import TrueFalseScorer


@pytest.fixture
def mock_objective_target():
    """Create a mock objective target for testing."""
    return MagicMock(spec=PromptTarget)


@pytest.fixture
def mock_adversarial_target():
    """Create a mock adversarial target for testing."""
    return MagicMock(spec=PromptChatTarget)


@pytest.fixture
def mock_objective_scorer():
    """Create a mock objective scorer for testing."""
    return MagicMock(spec=TrueFalseScorer)


@pytest.fixture
def sample_objectives():
    """Create sample objectives for testing."""
    return ["objective1", "objective2", "objective3"]


@pytest.mark.usefixtures("patch_central_database")
class TestFoundryScenarioInitialization:
    """Tests for FoundryScenario initialization."""

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    @pytest.mark.asyncio
    async def test_init_with_single_strategy(self, mock_harmbench, mock_objective_target, mock_objective_scorer):
        """Test initialization with a single attack strategy."""
        mock_dataset = Mock()
        mock_dataset.get_random_values.return_value = ["test objective"]
        mock_harmbench.return_value = mock_dataset

        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.Base64]],
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async()
        assert scenario.atomic_attack_count > 0
        assert scenario.name == "Foundry Scenario"

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    @pytest.mark.asyncio
    async def test_init_with_multiple_strategies(self, mock_harmbench, mock_objective_target, mock_objective_scorer):
        """Test initialization with multiple attack strategies."""
        mock_dataset = Mock()
        mock_dataset.get_random_values.return_value = ["test objective"]
        mock_harmbench.return_value = mock_dataset

        strategies = [
            [FoundryAttackStrategy.Base64],
            [FoundryAttackStrategy.ROT13],
            [FoundryAttackStrategy.Leetspeak],
        ]

        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=strategies,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async()
        assert scenario.atomic_attack_count >= len(strategies)

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    def test_init_with_custom_objectives(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test initialization with custom objectives."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.Base64]],
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario._objectives == sample_objectives
        # Should not call fetch_harmbench_dataset when objectives provided
        mock_harmbench.assert_not_called()

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    def test_init_with_custom_adversarial_target(
        self, mock_harmbench, mock_objective_target, mock_adversarial_target, mock_objective_scorer, sample_objectives
    ):
        """Test initialization with custom adversarial target."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.Base64]],
            adversarial_chat=mock_adversarial_target,
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario._adversarial_chat == mock_adversarial_target

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    def test_init_with_custom_scorer(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test initialization with custom objective scorer."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.Base64]],
            objective_scorer=mock_objective_scorer,
            objectives=sample_objectives,
        )

        assert scenario._objective_scorer == mock_objective_scorer

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    def test_init_with_memory_labels(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test initialization with memory labels."""
        memory_labels = {"test": "foundry", "category": "attack"}

        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.Base64]],
            memory_labels=memory_labels,
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario._memory_labels == memory_labels

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch("pyrit.scenarios.scenarios.foundry_scenario.TrueFalseCompositeScorer")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
            "AZURE_CONTENT_SAFETY_API_ENDPOINT": "https://test.content.azure.com/",
            "AZURE_CONTENT_SAFETY_API_KEY": "test-content-key",
        },
    )
    def test_init_creates_default_scorer_when_not_provided(self, mock_composite, mock_harmbench, mock_objective_target):
        """Test that initialization creates default scorer when not provided."""
        # Mock the composite scorer
        mock_composite_instance = MagicMock(spec=TrueFalseScorer)
        mock_composite.return_value = mock_composite_instance

        # Mock HarmBench dataset
        mock_dataset = Mock()
        mock_dataset.get_random_values.return_value = ["harmbench1", "harmbench2", "harmbench3", "harmbench4"]
        mock_harmbench.return_value = mock_dataset

        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.Base64]],
        )

        # Verify default scorer was created
        mock_composite.assert_called_once()

        # Verify HarmBench was used for objectives
        mock_harmbench.assert_called_once()
        assert len(scenario._objectives) == 4


@pytest.mark.usefixtures("patch_central_database")
class TestFoundryScenarioStrategyNormalization:
    """Tests for attack strategy normalization."""

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    @pytest.mark.asyncio
    async def test_normalize_easy_strategies(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test that EASY strategy expands to easy attack strategies."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.EASY]],
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async()
        # EASY should expand to multiple attack strategies
        assert scenario.atomic_attack_count > 1

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    @pytest.mark.asyncio
    async def test_normalize_moderate_strategies(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test that MODERATE strategy expands to moderate attack strategies."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.MODERATE]],
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async()
        # MODERATE should expand to moderate attack strategies (currently only 1: Tense)
        assert scenario.atomic_attack_count >= 1

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    @pytest.mark.asyncio
    async def test_normalize_difficult_strategies(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test that DIFFICULT strategy expands to difficult attack strategies."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.DIFFICULT]],
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async()
        # DIFFICULT should expand to multiple attack strategies
        assert scenario.atomic_attack_count > 1

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    @pytest.mark.asyncio
    async def test_normalize_mixed_difficulty_levels(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test that multiple difficulty levels expand correctly."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.EASY], [FoundryAttackStrategy.MODERATE]],
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async()
        # Combined difficulty levels should expand to multiple strategies
        assert scenario.atomic_attack_count > 5  # EASY has 20, MODERATE has 1, combined should have more

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    @pytest.mark.asyncio
    async def test_normalize_with_specific_and_difficulty_levels(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test that specific strategies combined with difficulty levels work correctly."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[
                [FoundryAttackStrategy.EASY],
                [FoundryAttackStrategy.Base64],  # Specific strategy
            ],
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async()
        # EASY expands to 20 strategies, but Base64 might already be in EASY, so at least 20
        assert scenario.atomic_attack_count >= 20


@pytest.mark.usefixtures("patch_central_database")
class TestFoundryScenarioAttackCreation:
    """Tests for attack creation from strategies."""

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    def test_get_attack_from_single_turn_strategy(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test creating an attack from a single-turn strategy."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.Base64]],
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        atomic_attack = scenario._get_attack_from_strategy([FoundryAttackStrategy.Base64])

        assert isinstance(atomic_attack, AtomicAttack)
        assert atomic_attack._objectives == sample_objectives

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    def test_get_attack_from_multi_turn_strategy(
        self, mock_harmbench, mock_objective_target, mock_adversarial_target, mock_objective_scorer, sample_objectives
    ):
        """Test creating a multi-turn attack strategy."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.Crescendo]],
            adversarial_chat=mock_adversarial_target,
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        atomic_attack = scenario._get_attack_from_strategy([FoundryAttackStrategy.Crescendo])

        assert isinstance(atomic_attack, AtomicAttack)
        assert atomic_attack._objectives == sample_objectives


@pytest.mark.usefixtures("patch_central_database")
class TestFoundryScenarioGetAttack:
    """Tests for the _get_attack method."""

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    def test_get_attack_single_turn_with_converters(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test creating a single-turn attack with converters."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.Base64]],
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        attack = scenario._get_attack(
            attack_type=PromptSendingAttack,
            converters=[Base64Converter()],
        )

        assert isinstance(attack, PromptSendingAttack)

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    def test_get_attack_multi_turn_with_adversarial_target(
        self, mock_harmbench, mock_objective_target, mock_adversarial_target, mock_objective_scorer, sample_objectives
    ):
        """Test creating a multi-turn attack."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.Crescendo]],
            adversarial_chat=mock_adversarial_target,
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        attack = scenario._get_attack(
            attack_type=CrescendoAttack,
            converters=[],
        )

        assert isinstance(attack, CrescendoAttack)


@pytest.mark.usefixtures("patch_central_database")
class TestFoundryScenarioAllStrategies:
    """Tests that all strategies can be instantiated."""

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    @pytest.mark.parametrize(
        "strategy",
        [
            FoundryAttackStrategy.AnsiAttack,
            FoundryAttackStrategy.AsciiArt,
            FoundryAttackStrategy.AsciiSmuggler,
            FoundryAttackStrategy.Atbash,
            FoundryAttackStrategy.Base64,
            FoundryAttackStrategy.Binary,
            FoundryAttackStrategy.Caesar,
            FoundryAttackStrategy.CharacterSpace,
            FoundryAttackStrategy.CharSwap,
            FoundryAttackStrategy.Diacritic,
            FoundryAttackStrategy.Flip,
            FoundryAttackStrategy.Leetspeak,
            FoundryAttackStrategy.Morse,
            FoundryAttackStrategy.ROT13,
            FoundryAttackStrategy.SuffixAppend,
            FoundryAttackStrategy.StringJoin,
            FoundryAttackStrategy.Tense,
            FoundryAttackStrategy.UnicodeConfusable,
            FoundryAttackStrategy.UnicodeSubstitution,
            FoundryAttackStrategy.Url,
            FoundryAttackStrategy.Jailbreak,
        ],
    )
    def test_all_single_turn_strategies_create_attack_runs(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives, strategy
    ):
        """Test that all single-turn strategies can create attack runs."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[strategy]],
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        atomic_attack = scenario._get_attack_from_strategy([strategy])
        assert isinstance(atomic_attack, AtomicAttack)

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    @pytest.mark.parametrize(
        "strategy",
        [
            FoundryAttackStrategy.MultiTurn,
            FoundryAttackStrategy.Crescendo,
        ],
    )
    def test_all_multi_turn_strategies_create_attack_runs(
        self,
        mock_harmbench,
        mock_objective_target,
        mock_adversarial_target,
        mock_objective_scorer,
        sample_objectives,
        strategy,
    ):
        """Test that all multi-turn strategies can create attack runs."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[strategy]],
            adversarial_chat=mock_adversarial_target,
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        atomic_attack = scenario._get_attack_from_strategy([strategy])
        assert isinstance(atomic_attack, AtomicAttack)


@pytest.mark.usefixtures("patch_central_database")
class TestFoundryScenarioProperties:
    """Tests for FoundryScenario properties and attributes."""

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    def test_scenario_name_includes_strategies(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test that scenario name is set correctly."""
        strategies = [[FoundryAttackStrategy.Base64], [FoundryAttackStrategy.ROT13]]

        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=strategies,
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario.name == "Foundry Scenario"
        # Verify the scenario was initialized with the attack strategies
        assert len(scenario._foundry_strategy_compositions) == 2
        assert any("base64" in str(comp).lower() for comp in scenario._foundry_strategy_compositions)
        assert any("rot13" in str(comp).lower() for comp in scenario._foundry_strategy_compositions)

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    def test_scenario_version_is_set(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test that scenario version is properly set."""
        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=[[FoundryAttackStrategy.Base64]],
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario.version == 1

    @patch("pyrit.scenarios.scenarios.foundry_scenario.fetch_harmbench_dataset")
    @patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
        },
    )
    @pytest.mark.asyncio
    async def test_scenario_atomic_attack_count_matches_strategies(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test that atomic attack count is reasonable for the number of strategies."""
        strategies = [
            [FoundryAttackStrategy.Base64],
            [FoundryAttackStrategy.ROT13],
            [FoundryAttackStrategy.Leetspeak],
        ]

        scenario = FoundryScenario(
            objective_target=mock_objective_target,
            attack_strategies=strategies,
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async()
        # Should have at least as many runs as specific strategies provided
        assert scenario.atomic_attack_count >= len(strategies)
