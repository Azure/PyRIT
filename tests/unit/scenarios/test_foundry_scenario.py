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
from pyrit.scenarios import AtomicAttack, FoundryScenario, FoundryStrategy
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
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=[FoundryStrategy.Base64],
        )
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
            FoundryStrategy.Base64,
            FoundryStrategy.ROT13,
            FoundryStrategy.Leetspeak,
        ]

        scenario = FoundryScenario(
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=strategies,
        )
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
    @pytest.mark.asyncio
    async def test_init_with_memory_labels(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test initialization with memory labels."""
        memory_labels = {"test": "foundry", "category": "attack"}

        scenario = FoundryScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario._memory_labels == {}

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            memory_labels=memory_labels,
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

        scenario = FoundryScenario()

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
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=[FoundryStrategy.EASY],
        )
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
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=[FoundryStrategy.MODERATE],
        )
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
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=[FoundryStrategy.DIFFICULT],
        )
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
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=[FoundryStrategy.EASY, FoundryStrategy.MODERATE],
        )
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
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=[
                FoundryStrategy.EASY,
                FoundryStrategy.Base64,  # Specific strategy
            ],
        )
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
    @pytest.mark.asyncio
    async def test_get_attack_from_single_turn_strategy(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test creating an attack from a single-turn strategy."""
        scenario = FoundryScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=[FoundryStrategy.Base64],
        )

        # Get the composite strategy that was created during initialization
        composite_strategy = scenario._scenario_composites[0]
        atomic_attack = scenario._get_attack_from_strategy(composite_strategy)

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
    @pytest.mark.asyncio
    async def test_get_attack_from_multi_turn_strategy(
        self, mock_harmbench, mock_objective_target, mock_adversarial_target, mock_objective_scorer, sample_objectives
    ):
        """Test creating a multi-turn attack strategy."""
        scenario = FoundryScenario(
            adversarial_chat=mock_adversarial_target,
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=[FoundryStrategy.Crescendo],
        )

        # Get the composite strategy that was created during initialization
        composite_strategy = scenario._scenario_composites[0]
        atomic_attack = scenario._get_attack_from_strategy(composite_strategy)

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
            FoundryStrategy.AnsiAttack,
            FoundryStrategy.AsciiArt,
            FoundryStrategy.AsciiSmuggler,
            FoundryStrategy.Atbash,
            FoundryStrategy.Base64,
            FoundryStrategy.Binary,
            FoundryStrategy.Caesar,
            FoundryStrategy.CharacterSpace,
            FoundryStrategy.CharSwap,
            FoundryStrategy.Diacritic,
            FoundryStrategy.Flip,
            FoundryStrategy.Leetspeak,
            FoundryStrategy.Morse,
            FoundryStrategy.ROT13,
            FoundryStrategy.SuffixAppend,
            FoundryStrategy.StringJoin,
            FoundryStrategy.Tense,
            FoundryStrategy.UnicodeConfusable,
            FoundryStrategy.UnicodeSubstitution,
            FoundryStrategy.Url,
            FoundryStrategy.Jailbreak,
        ],
    )
    @pytest.mark.asyncio
    async def test_all_single_turn_strategies_create_attack_runs(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives, strategy
    ):
        """Test that all single-turn strategies can create attack runs."""
        scenario = FoundryScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=[strategy],
        )

        # Get the composite strategy that was created during initialization
        composite_strategy = scenario._scenario_composites[0]
        atomic_attack = scenario._get_attack_from_strategy(composite_strategy)
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
            FoundryStrategy.MultiTurn,
            FoundryStrategy.Crescendo,
        ],
    )
    @pytest.mark.asyncio
    async def test_all_multi_turn_strategies_create_attack_runs(
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
            adversarial_chat=mock_adversarial_target,
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=[strategy],
        )

        # Get the composite strategy that was created during initialization
        composite_strategy = scenario._scenario_composites[0]
        atomic_attack = scenario._get_attack_from_strategy(composite_strategy)
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
    @pytest.mark.asyncio
    async def test_scenario_composites_set_after_initialize(
        self, mock_harmbench, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test that scenario composites are set after initialize_async."""
        strategies = [FoundryStrategy.Base64, FoundryStrategy.ROT13]

        scenario = FoundryScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        # Before initialize_async, composites should be empty
        assert len(scenario._scenario_composites) == 0

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=strategies,
        )

        # After initialize_async, composites should be set
        assert len(scenario._scenario_composites) == len(strategies)
        assert scenario.atomic_attack_count == len(strategies)

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
            FoundryStrategy.Base64,
            FoundryStrategy.ROT13,
            FoundryStrategy.Leetspeak,
        ]

        scenario = FoundryScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target,
            scenario_strategies=strategies,
        )
        # Should have at least as many runs as specific strategies provided
        assert scenario.atomic_attack_count >= len(strategies)
