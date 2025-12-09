# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the CyberScenario class."""
import pathlib
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from pyrit.common.path import DATASETS_PATH
from pyrit.executor.attack import PromptSendingAttack, RedTeamingAttack
from pyrit.executor.attack.core.attack_config import AttackScoringConfig
from pyrit.models import SeedDataset, SeedObjective
from pyrit.prompt_target import OpenAIChatTarget, PromptChatTarget, PromptTarget
from pyrit.scenario import CyberScenario, CyberStrategy
from pyrit.score import TrueFalseCompositeScorer


@pytest.fixture
def mock_memory_seeds():
    """Create mock seed objectives that memory.get_seeds() would return."""
    malware_path = pathlib.Path(DATASETS_PATH) / "seed_datasets" / "local" / "airt"
    seed_prompts = list(SeedDataset.from_yaml_file(malware_path / "malware.prompt").get_values())
    return [SeedObjective(value=prompt, data_type="text") for prompt in seed_prompts]


@pytest.fixture
def fast_cyberstrategy():
    return CyberStrategy.SINGLE_TURN


@pytest.fixture
def slow_cyberstrategy():
    return CyberStrategy.MULTI_TURN


@pytest.fixture
def malware_prompts():
    """The default malware prompts."""
    malware_path = pathlib.Path(DATASETS_PATH) / "seed_datasets" / "local" / "airt"
    seed_prompts = list(SeedDataset.from_yaml_file(malware_path / "malware.prompt").get_values())
    return seed_prompts


@pytest.fixture
def mock_runtime_env():
    with patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
            "OPENAI_CHAT_ENDPOINT": "https://test.openai.azure.com/",
            "OPENAI_CHAT_KEY": "test-key",
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
    mock = MagicMock(spec=TrueFalseCompositeScorer)
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

    def test_init_with_custom_objectives(self, mock_objective_scorer, sample_objectives):
        """Test initialization with custom objectives."""

        scenario = CyberScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert len(scenario._objectives) == len(sample_objectives)
        assert scenario.name == "Cyber Scenario"
        assert scenario.version == 1

    def test_init_with_default_objectives(self, mock_objective_scorer, malware_prompts, mock_memory_seeds):
        """Test initialization with default objectives."""

        with patch.object(CyberScenario, "_get_default_objectives", return_value=malware_prompts):
            scenario = CyberScenario(objective_scorer=mock_objective_scorer)

            assert scenario._objectives == malware_prompts
            assert scenario.name == "Cyber Scenario"
            assert scenario.version == 1

    def test_init_with_default_scorer(self, mock_memory_seeds):
        """Test initialization with default scorer."""
        with patch.object(
            CyberScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = CyberScenario()
            assert scenario._objective_scorer_identifier

    def test_init_with_custom_scorer(self, mock_objective_scorer, mock_memory_seeds):
        """Test initialization with custom scorer."""
        scorer = MagicMock(TrueFalseCompositeScorer)
        with patch.object(
            CyberScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = CyberScenario(objective_scorer=scorer)
            assert isinstance(scenario._scorer_config, AttackScoringConfig)

    def test_init_default_adversarial_chat(self, mock_objective_scorer, mock_memory_seeds):
        """Test initialization with default adversarial chat."""
        with patch.object(
            CyberScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = CyberScenario(
                objective_scorer=mock_objective_scorer,
            )

            assert isinstance(scenario._adversarial_chat, OpenAIChatTarget)
            assert scenario._adversarial_chat._temperature == 1.2

    def test_init_with_adversarial_chat(self, mock_objective_scorer, mock_memory_seeds):
        """Test initialization with adversarial chat (for red teaming attack variation)."""
        adversarial_chat = MagicMock(OpenAIChatTarget)
        adversarial_chat.get_identifier.return_value = {"type": "CustomAdversary"}

        with patch.object(
            CyberScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = CyberScenario(
                adversarial_chat=adversarial_chat,
                objective_scorer=mock_objective_scorer,
            )
            assert scenario._adversarial_chat == adversarial_chat
            assert scenario._adversarial_config.target == adversarial_chat

    def test_init_raises_exception_when_no_datasets_available(self, mock_objective_scorer):
        """Test that initialization raises ValueError when datasets are not available in memory."""
        # Don't mock _get_default_objectives, let it try to load from empty memory
        with pytest.raises(ValueError, match="Dataset is not available or failed to load"):
            CyberScenario(objective_scorer=mock_objective_scorer)


@pytest.mark.usefixtures(*FIXTURES)
class TestCyberScenarioAttackGeneration:
    """Tests for CyberScenario attack generation."""

    @pytest.mark.asyncio
    async def test_attack_generation_for_all(self, mock_objective_target, mock_objective_scorer, mock_memory_seeds):
        """Test that _get_atomic_attacks_async returns atomic attacks."""
        with patch.object(
            CyberScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = CyberScenario(objective_scorer=mock_objective_scorer)

            await scenario.initialize_async(objective_target=mock_objective_target)
            atomic_attacks = await scenario._get_atomic_attacks_async()

            assert len(atomic_attacks) > 0
            assert all(hasattr(run, "_attack") for run in atomic_attacks)

    @pytest.mark.asyncio
    async def test_attack_generation_for_singleturn(
        self, mock_objective_target, mock_objective_scorer, sample_objectives, fast_cyberstrategy
    ):
        """Test that the single turn attack generation works."""
        scenario = CyberScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target, scenario_strategies=[fast_cyberstrategy]
        )
        atomic_attacks = await scenario._get_atomic_attacks_async()
        for run in atomic_attacks:
            assert isinstance(run._attack, PromptSendingAttack)

    @pytest.mark.asyncio
    async def test_attack_generation_for_multiturn(
        self, mock_objective_target, mock_objective_scorer, sample_objectives, slow_cyberstrategy
    ):
        """Test that the multi turn attack generation works."""
        scenario = CyberScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target, scenario_strategies=[slow_cyberstrategy]
        )
        atomic_attacks = await scenario._get_atomic_attacks_async()

        for run in atomic_attacks:
            assert isinstance(run._attack, RedTeamingAttack)

    @pytest.mark.asyncio
    async def test_attack_runs_include_objectives(
        self, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test that attack runs include objectives for each seed prompt."""
        scenario = CyberScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(objective_target=mock_objective_target)
        atomic_attacks = await scenario._get_atomic_attacks_async()

        # Check that objectives are created for each seed prompt
        for run in atomic_attacks:
            assert len(run._objectives) == len(sample_objectives)
            for i, objective in enumerate(run._objectives):
                assert sample_objectives[i] in objective

    @pytest.mark.asyncio
    async def test_get_atomic_attacks_async_returns_attacks(
        self, mock_objective_target, mock_objective_scorer, sample_objectives
    ):
        """Test that _get_atomic_attacks_async returns atomic attacks."""
        scenario = CyberScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(objective_target=mock_objective_target)
        atomic_attacks = await scenario._get_atomic_attacks_async()
        assert len(atomic_attacks) > 0
        assert all(hasattr(run, "_attack") for run in atomic_attacks)


@pytest.mark.usefixtures(*FIXTURES)
class TestCyberScenarioLifecycle:
    """
    Tests for CyberScenario lifecycle, including initialize_async and execution.
    """

    async def test_initialize_async_with_max_concurrency(
        self, mock_objective_target, mock_objective_scorer, mock_memory_seeds
    ):
        """Test initialization with custom max_concurrency."""
        with patch.object(
            CyberScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = CyberScenario(objective_scorer=mock_objective_scorer)
            await scenario.initialize_async(objective_target=mock_objective_target, max_concurrency=20)
            assert scenario._max_concurrency == 20

    async def test_initialize_async_with_memory_labels(
        self, mock_objective_target, mock_objective_scorer, mock_memory_seeds
    ):
        """Test initialization with memory labels."""
        memory_labels = {"test": "cyber", "category": "scenario"}

        with patch.object(
            CyberScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = CyberScenario(
                objective_scorer=mock_objective_scorer,
            )
            await scenario.initialize_async(
                memory_labels=memory_labels,
                objective_target=mock_objective_target,
            )

            assert scenario._memory_labels == memory_labels


@pytest.mark.usefixtures(*FIXTURES)
class TestCyberScenarioProperties:
    """
    Tests for CyberScenario properties and attributes.
    """

    def test_scenario_version_is_set(self, mock_objective_scorer, sample_objectives):
        """Test that scenario version is properly set."""
        scenario = CyberScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario.version == 1

    @pytest.mark.asyncio
    async def test_no_target_duplication(self, mock_objective_target, mock_memory_seeds):
        """Test that all three targets (adversarial, object, scorer) are distinct."""
        with patch.object(
            CyberScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = CyberScenario()
            await scenario.initialize_async(objective_target=mock_objective_target)

            objective_target = scenario._objective_target

            # this works because TrueFalseCompositeScorer subclasses TrueFalseScorer,
            # but TrueFalseScorer itself (the type for ScorerConfig) does not have ._scorers.
            scorer_target = scenario._scorer_config.objective_scorer._scorers[0]  # type: ignore
            adversarial_target = scenario._adversarial_chat

            assert objective_target != scorer_target
            assert objective_target != adversarial_target
            assert scorer_target != adversarial_target
