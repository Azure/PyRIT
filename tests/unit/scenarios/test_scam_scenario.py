# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the ScamScenario class."""

import pathlib
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from pyrit.common.path import DATASETS_PATH
from pyrit.executor.attack.core.attack_config import AttackScoringConfig
from pyrit.executor.attack.single_turn.role_play import RolePlayAttack
from pyrit.models import SeedDataset
from pyrit.prompt_target import OpenAIChatTarget, PromptChatTarget, PromptTarget
from pyrit.scenario import ScamScenario, ScamStrategy
from pyrit.score.true_false.self_ask_true_false_scorer import SelfAskTrueFalseScorer


@pytest.fixture
def mock_runtime_env():
    with patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
            "OPENAI_CHAT_ENDPOINT": "https://test.openai.azure.com/",
            "OPENAI_CHAT_KEY": "test-key",
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
def scam_prompts() -> List[str]:
    prompt_path = pathlib.Path(DATASETS_PATH) / "seed_prompts"
    dataset = SeedDataset.from_yaml_file(prompt_path / "scams.prompt").get_values()
    return list(dataset)


@pytest.fixture
def sample_objectives() -> List[str]:
    return ["scam prompt 1", "scam prompt 2"]


@pytest.fixture
def impersonation_strategy() -> ScamStrategy:
    return ScamStrategy.IMPERSONATION


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
    ) -> None:
        scenario = ScamScenario(objective_scorer=mock_objective_scorer)

        assert scenario._objectives == scam_prompts
        assert scenario.name == "Scam Scenario"
        assert scenario.version == 1

    def test_init_with_default_scorer(self) -> None:
        scenario = ScamScenario()
        assert scenario._objective_scorer_identifier

    def test_init_with_custom_scorer(
        self,
        *,
        mock_objective_scorer: SelfAskTrueFalseScorer,
    ) -> None:
        scorer = MagicMock(spec=SelfAskTrueFalseScorer)
        scenario = ScamScenario(objective_scorer=scorer)
        assert isinstance(scenario._scorer_config, AttackScoringConfig)

    def test_init_default_adversarial_chat(
        self,
        *,
        mock_objective_scorer: SelfAskTrueFalseScorer,
    ) -> None:
        scenario = ScamScenario(objective_scorer=mock_objective_scorer)

        assert isinstance(scenario._adversarial_chat, OpenAIChatTarget)
        assert scenario._adversarial_chat._temperature == 1.2

    def test_init_with_adversarial_chat(
        self,
        *,
        mock_objective_scorer: SelfAskTrueFalseScorer,
    ) -> None:
        adversarial_chat = MagicMock(OpenAIChatTarget)
        adversarial_chat.get_identifier.return_value = {"type": "CustomAdversary"}

        scenario = ScamScenario(
            adversarial_chat=adversarial_chat,
            objective_scorer=mock_objective_scorer,
        )
        assert scenario._adversarial_chat == adversarial_chat
        assert scenario._adversarial_config.target == adversarial_chat


@pytest.mark.usefixtures(*FIXTURES)
class TestScamScenarioAttackGeneration:
    """Tests for ScamScenario attack generation."""

    @pytest.mark.asyncio
    async def test_attack_generation_for_roleplay_async(
        self,
        *,
        mock_objective_target: PromptTarget,
        mock_objective_scorer: SelfAskTrueFalseScorer,
        impersonation_strategy: ScamStrategy,
        sample_objectives: List[str],
    ) -> None:
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
    ) -> None:
        scenario = ScamScenario(objective_scorer=mock_objective_scorer)
        await scenario.initialize_async(objective_target=mock_objective_target, max_concurrency=20)
        assert scenario._max_concurrency == 20

    @pytest.mark.asyncio
    async def test_initialize_async_with_memory_labels(
        self,
        *,
        mock_objective_target: PromptTarget,
        mock_objective_scorer: SelfAskTrueFalseScorer,
    ) -> None:
        memory_labels = {"type": "scam", "category": "scenario"}
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
        scenario = ScamScenario(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario.version == 1

    @pytest.mark.asyncio
    async def test_no_target_duplication_async(
        self,
        *,
        mock_objective_target: PromptTarget,
    ) -> None:
        scenario = ScamScenario()
        await scenario.initialize_async(objective_target=mock_objective_target)

        objective_target = scenario._objective_target
        scorer_target = scenario._scorer_config.objective_scorer  # type: ignore
        adversarial_target = scenario._adversarial_chat

        assert objective_target != scorer_target
        assert objective_target != adversarial_target
        assert scorer_target != adversarial_target
