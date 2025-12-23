# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the ScamScenario class."""

import pathlib
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from pyrit.common.path import DATASETS_PATH
from pyrit.executor.attack import RolePlayAttack
from pyrit.executor.attack.core.attack_config import AttackScoringConfig
from pyrit.models import SeedDataset, SeedObjective
from pyrit.prompt_target import OpenAIChatTarget, PromptChatTarget, PromptTarget
from pyrit.scenario import ScamScenario, ScamStrategy
from pyrit.score import SelfAskTrueFalseScorer

SEED_DATASETS_PATH = pathlib.Path(DATASETS_PATH) / "seed_datasets" / "local" / "airt"
SEED_PROMPT_LIST = list(SeedDataset.from_yaml_file(SEED_DATASETS_PATH / "scams.prompt").get_values())


@pytest.fixture
def mock_memory_seeds() -> List[SeedObjective]:
    """Create mock seed objectives that memory.get_seeds() would return."""
    return [SeedObjective(value=prompt, data_type="text") for prompt in SEED_PROMPT_LIST]


@pytest.fixture
def impersonation_strategy() -> ScamStrategy:
    return ScamStrategy.ALL


@pytest.fixture
def scam_prompts() -> List[str]:
    return SEED_PROMPT_LIST


@pytest.fixture
def mock_runtime_env():
    with patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL": "gpt-4",
            "OPENAI_CHAT_ENDPOINT": "https://test.openai.azure.com/",
            "OPENAI_CHAT_KEY": "test-key",
            "OPENAI_CHAT_MODEL": "gpt-4",
        },
    ):
        yield


@pytest.fixture
def mock_objective_target() -> PromptTarget:
    mock = MagicMock(spec=PromptTarget)
    mock.get_identifier.return_value = {"__type__": "MockObjectiveTarget", "__module__": "test"}
    return mock


@pytest.fixture
def mock_objective_scorer() -> SelfAskTrueFalseScorer:
    mock = MagicMock(spec=SelfAskTrueFalseScorer)
    mock.get_identifier.return_value = {"__type__": "MockObjectiveScorer", "__module__": "test"}
    return mock


@pytest.fixture
def mock_adversarial_target() -> PromptChatTarget:
    mock = MagicMock(spec=PromptChatTarget)
    mock.get_identifier.return_value = {"__type__": "MockAdversarialTarget", "__module__": "test"}
    return mock


@pytest.fixture
def sample_objectives() -> List[str]:
    return ["scam prompt 1", "scam prompt 2"]


FIXTURES = ["patch_central_database", "mock_runtime_env"]


@pytest.mark.usefixtures(*FIXTURES)
class TestScamScenarioInitialization:
    """Tests for ScamScenario initialization."""

    def test_init_with_custom_objectives(
        self,
        *,
        mock_objective_scorer: SelfAskTrueFalseScorer,
        sample_objectives: List[str],
    ) -> None:
        scenario = ScamScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert len(scenario._objectives) == len(sample_objectives)
        assert scenario.name == "Scam Scenario"
        assert scenario.version == 1

    def test_init_with_default_objectives(
        self,
        *,
        mock_objective_scorer: SelfAskTrueFalseScorer,
        scam_prompts: List[str],
        mock_memory_seeds: List[SeedObjective],
    ) -> None:
        with patch.object(ScamScenario, "_get_default_objectives", return_value=scam_prompts):
            scenario = ScamScenario(objective_scorer=mock_objective_scorer)

            assert scenario._objectives == scam_prompts
            assert scenario.name == "Scam Scenario"
            assert scenario.version == 1

    def test_init_with_default_scorer(self, mock_memory_seeds) -> None:
        """Test initialization with default scorer."""
        with patch.object(
            ScamScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = ScamScenario()
            assert scenario._objective_scorer_identifier

    def test_init_with_custom_scorer(
        self, *, mock_objective_scorer: SelfAskTrueFalseScorer, mock_memory_seeds: list[SeedObjective]
    ) -> None:
        """Test initialization with custom scorer."""
        scorer = MagicMock(spec=SelfAskTrueFalseScorer)

        with patch.object(
            ScamScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = ScamScenario(objective_scorer=scorer)
            assert isinstance(scenario._scorer_config, AttackScoringConfig)

    def test_init_default_adversarial_chat(
        self, *, mock_objective_scorer: SelfAskTrueFalseScorer, mock_memory_seeds: list[SeedObjective]
    ) -> None:
        with patch.object(
            ScamScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = ScamScenario(objective_scorer=mock_objective_scorer)

            assert isinstance(scenario._adversarial_chat, OpenAIChatTarget)
            assert scenario._adversarial_chat._temperature == 1.2

    def test_init_with_adversarial_chat(
        self, *, mock_objective_scorer: SelfAskTrueFalseScorer, mock_memory_seeds: list[SeedObjective]
    ) -> None:
        adversarial_chat = MagicMock(OpenAIChatTarget)
        adversarial_chat.get_identifier.return_value = {"type": "CustomAdversary"}

        with patch.object(
            ScamScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = ScamScenario(
                adversarial_chat=adversarial_chat,
                objective_scorer=mock_objective_scorer,
            )
            assert scenario._adversarial_chat == adversarial_chat
            assert scenario._adversarial_config.target == adversarial_chat

    def test_init_raises_exception_when_no_datasets_available(self, mock_objective_scorer):
        """Test that initialization raises ValueError when datasets are not available in memory."""
        # Don't mock _get_default_objectives, let it try to load from empty memory
        with pytest.raises(ValueError, match="Dataset is not available or failed to load"):
            ScamScenario(objective_scorer=mock_objective_scorer)


@pytest.mark.usefixtures(*FIXTURES)
class TestScamScenarioAttackGeneration:
    """Tests for ScamScenario attack generation."""

    @pytest.mark.asyncio
    async def test_attack_generation_for_all(self, mock_objective_target, mock_objective_scorer, mock_memory_seeds):
        """Test that _get_atomic_attacks_async returns atomic attacks."""
        with patch.object(
            ScamScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = ScamScenario(objective_scorer=mock_objective_scorer)

            await scenario.initialize_async(objective_target=mock_objective_target)
            atomic_attacks = await scenario._get_atomic_attacks_async()

            assert len(atomic_attacks) > 0
            assert all(hasattr(run, "_attack") for run in atomic_attacks)

    @pytest.mark.asyncio
    async def test_attack_generation_for_impersonation_async(
        self,
        *,
        mock_objective_target: PromptTarget,
        mock_objective_scorer: SelfAskTrueFalseScorer,
        impersonation_strategy: ScamStrategy,
        sample_objectives: List[str],
    ) -> None:
        """Test that the impersonation strategy attack generation works."""
        scenario = ScamScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(
            objective_target=mock_objective_target, scenario_strategies=[impersonation_strategy]
        )
        atomic_attacks = await scenario._get_atomic_attacks_async()
        for run in atomic_attacks:
            assert isinstance(run._attack, RolePlayAttack)

    @pytest.mark.asyncio
    async def test_attack_runs_include_objectives_async(
        self,
        *,
        mock_objective_target: PromptTarget,
        mock_objective_scorer: SelfAskTrueFalseScorer,
        sample_objectives: List[str],
    ) -> None:
        """Test that attack runs include objectives for each seed prompt."""
        scenario = ScamScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(objective_target=mock_objective_target)
        atomic_attacks = await scenario._get_atomic_attacks_async()

        for run in atomic_attacks:
            assert len(run._objectives) == len(sample_objectives)
            for index, objective in enumerate(run._objectives):
                assert sample_objectives[index] in objective

    @pytest.mark.asyncio
    async def test_get_atomic_attacks_async_returns_attacks(
        self,
        *,
        mock_objective_target: PromptTarget,
        mock_objective_scorer: SelfAskTrueFalseScorer,
        sample_objectives: List[str],
    ) -> None:
        """Test that _get_atomic_attacks_async returns atomic attacks."""
        scenario = ScamScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async(objective_target=mock_objective_target)
        atomic_attacks = await scenario._get_atomic_attacks_async()
        assert len(atomic_attacks) > 0
        assert all(hasattr(run, "_attack") for run in atomic_attacks)


@pytest.mark.usefixtures(*FIXTURES)
class TestScamScenarioLifecycle:
    """Tests for ScamScenario lifecycle behavior."""

    @pytest.mark.asyncio
    async def test_initialize_async_with_max_concurrency(
        self,
        *,
        mock_objective_target: PromptTarget,
        mock_objective_scorer: SelfAskTrueFalseScorer,
        mock_memory_seeds: List[SeedObjective],
    ) -> None:
        """Test initialization with custom max_concurrency."""
        with patch.object(
            ScamScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = ScamScenario(objective_scorer=mock_objective_scorer)
            await scenario.initialize_async(objective_target=mock_objective_target, max_concurrency=20)
            assert scenario._max_concurrency == 20

    @pytest.mark.asyncio
    async def test_initialize_async_with_memory_labels(
        self,
        *,
        mock_objective_target: PromptTarget,
        mock_objective_scorer: SelfAskTrueFalseScorer,
        mock_memory_seeds: List[SeedObjective],
    ) -> None:
        """Test initialization with memory labels."""
        memory_labels = {"type": "scam", "category": "scenario"}

        with patch.object(
            ScamScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = ScamScenario(objective_scorer=mock_objective_scorer)
            await scenario.initialize_async(
                memory_labels=memory_labels,
                objective_target=mock_objective_target,
            )
            assert scenario._memory_labels == memory_labels


@pytest.mark.usefixtures(*FIXTURES)
class TestScamScenarioProperties:
    """Tests for ScamScenario properties."""

    def test_scenario_version_is_set(
        self,
        *,
        mock_objective_scorer: SelfAskTrueFalseScorer,
        sample_objectives: List[str],
    ) -> None:
        """Test that scenario version is properly set."""
        scenario = ScamScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario.version == 1

    @pytest.mark.asyncio
    async def test_no_target_duplication_async(
        self, *, mock_objective_target: PromptTarget, mock_memory_seeds: List[SeedObjective]
    ) -> None:
        """Test that all three targets (adversarial, object, scorer) are distinct."""
        with patch.object(
            ScamScenario, "_get_default_objectives", return_value=[seed.value for seed in mock_memory_seeds]
        ):
            scenario = ScamScenario()
            await scenario.initialize_async(objective_target=mock_objective_target)

            objective_target = scenario._objective_target
            scorer_target = scenario._scorer_config.objective_scorer  # type: ignore
            adversarial_target = scenario._adversarial_chat

            assert objective_target != scorer_target
            assert objective_target != adversarial_target
            assert scorer_target != adversarial_target
